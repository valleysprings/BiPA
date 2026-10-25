"""SkillAdam: upstream ``FeedbackLoop`` driven by ``SkillAdamRunner`` on our batches, executor and LLM client."""

import copy, json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from harness_bench.eval.traces import SKILL_FRAME
from skilladam.core.acceptance_gate import AcceptanceGate, ImprovementRule
from skilladam.core.case_sampler import IterationCases
from skilladam.core.feedback_loop import CheckpointStore, EditBudgetConfig, FeedbackLoop, FeedbackLoopConfig
from skilladam.core.momentum import MomentumTracker
from skilladam.core.stage0 import Stage0PromptContext
from skilladam.core.type_taxonomy import load_taxonomy, render_taxonomy_section
from skilladam.cost.ledger import UsageLedger
from skilladam.execution.aggregation import AggregationResult, aggregate_metric_results
from skilladam.execution.contracts import GenerationResult, ToolCall
from skilladam.execution.evaluate import CaseEvaluation, EvaluationResult
from skilladam.execution.store import RunStore
from skilladam.methods.skilladam import SkillAdamRunner, _skill_hash, _validate_momentum_tool_calls
from skilladam.methods.skilladam_prompts import CORE_PROMPTS_DIR, BenchmarkPromptContext, IterationGeneration
from skilladam.types import BenchmarkCase, MetricResult, RolloutResult, UsageRecord

from src import executor
from src.llm import GatewayRejected

PROMPTS = Path(__file__).resolve().parents[1] / "prompts/skilladam"
# max/min_iterations are set to T; 99 consecutive failures means no early stop
LOOP = FeedbackLoopConfig(max_iterations=10, min_iterations=10, max_consecutive_failures=99, patch_attempts=3,
                          edit_budget=EditBudgetConfig("hard", v_max=0.1, base=4, minimum=1, beta=0.9))
GATE = AcceptanceGate(improvements=(ImprovementRule("primary", 0.0),))
SETTINGS = {"scope": None, "trajectory_compression": "deterministic", "optimizer_context_limit": 110000,
            "optimizer_history_turns": None, "max_tokens": 16384, "temperature": 1.0}


class UnknownVerdict(BaseException):
    """A verdict still unknown after the executor's retries; a BaseException so that upstream does not catch it."""


class Runner(SkillAdamRunner):
    """``SkillAdamRunner`` without its benchmark registry, backend and Stage0."""

    def __init__(self, cell, llm, out):
        self.cell, self.llm, self.output_dir, self.momentum = cell, llm, out / "skilladam", MomentumTracker()
        self.config = SimpleNamespace(benchmark=cell.benchmark, **SETTINGS, feedback_loop=LOOP, to_dict=lambda: {
            "benchmark": cell.benchmark, **SETTINGS, "feedback_loop": asdict(self.config.feedback_loop)})
        self.context = SimpleNamespace(benchmark=cell.benchmark, method="skilladam", model=llm.model,
                                       reasoning_effort="provider-default", workers=1, seed=0)
        self._run_store = self._usage_ledger = self._prompt_context = self._active_iteration = None
        self._cases_by_id, self._trajectory_cache, self._optimizer_history, self._validation_rollout_counts = \
            {}, {}, [], {}

    def optimize(self, seed, schedule):
        """Upstream ``SkillAdamRunner.run`` from ``seed`` on our batches, gated on the training batch; resumes."""
        checkpoint, T = CheckpointStore(self.output_dir), len(schedule)
        resume, self.config.feedback_loop = checkpoint.exists, replace(LOOP, max_iterations=T, min_iterations=T)
        self._run_store = RunStore(self.output_dir, resume=self.output_dir.exists())
        self._store.write_text("seed.md", seed)
        self._usage_ledger = UsageLedger(self.output_dir / "usage.jsonl")
        self.ledger = self.output_dir / "episodes.json"  # distinct (skill hash, task, split, role) rollouts
        self.seen = {tuple(row) for row in json.loads(self.ledger.read_text())} if self.ledger.exists() else set()
        self._prompt_context = prompt_context(self.cell.benchmark, len(schedule[0]))
        # upstream's same_batch cases (the published profiles' validation): the gate rolls out the training ids
        batches = [IterationCases(tuple(b), tuple(b), validation_mode="same_batch") for b in schedule]
        self._cases_by_id = {task: BenchmarkCase(task, {}) for b in batches
                             for task in b.training_case_ids + b.validation_case_ids}
        if resume:
            payload = checkpoint.load_payload()
            self._restore_optimizer_history(payload)
            k, stale = payload["next_iteration"], self.output_dir / "iterations-stale"
            # upstream's store never overwrites: set aside an aborted attempt of the next iteration
            if (part := self.output_dir / f"iterations/iter_{k:03d}").is_dir():
                stale.mkdir(exist_ok=True)
                part.rename(stale / f"iter_{k:03d}-{len(list(stale.iterdir()))}")
        loop = FeedbackLoop(self.config.feedback_loop, gate=GATE, checkpoint_store=checkpoint, momentum=self.momentum)
        result = loop.run(initial_skill=seed, resume=resume, batches=batches, propose_patch=self._propose_patch,
                          evaluate=self._evaluate_validation,
                          post_decision_hook=self._post_decision_hook(_skill_hash(seed)))
        self._store.write_text("current_skill.md", result.final_skill)
        self._store.write_text("final_skill.md", result.final_skill)
        return result

    def _evaluate_skill(self, skill, cases, *, stage):
        found = executor.evaluate(self.cell, skill, [c.case_id for c in cases], "train")  # training rollout and gate
        rows, step = {row["id"]: row for row in found}, self._active_iteration + 1
        unknown = [c.case_id for c in cases if rows.get(c.case_id, {"success": None})["success"] is None]
        if unknown:
            raise UnknownVerdict(f"{stage} step {step}: {len(unknown)} unknown outcomes")
        # upstream counts its validation rollouts after this call: the baseline first, then the candidate
        role = stage if stage == "training_rollout" else (
            "validation_candidate" if self._validation_rollout_counts.get(step - 1) else "validation_baseline")
        executor.record_round(self.cell.root, arm="skilladam", step=step, role=role, skill_text=skill,
                              evaluation_path=found[0]["evaluation_path"])
        role = "training" if stage == "training_rollout" else "gate"
        self.seen.update((_skill_hash(skill), case.case_id, "train", role) for case in cases)
        self.ledger.write_text(json.dumps(sorted(self.seen)))
        evaluated = []
        for case in cases:
            messages, hard = rows[case.case_id]["messages"], float(rows[case.case_id]["success"])
            task = next((m["content"] for m in messages if m["role"] == "user"), "")
            case.payload["task"] = case.payload.get("task") or SKILL_FRAME.sub(r"\1(the evaluated skill)\3", task)
            answer = next((m["content"] for m in reversed(messages) if m["role"] == "assistant" and m["content"]), "")
            metric = MetricResult(primary=hard, metrics={"hard": hard}, case_metrics={case.case_id: {"hard": hard}})
            evaluated.append(CaseEvaluation(case.case_id, RolloutResult(case.case_id, answer, messages), metric))
        overall = aggregate_metric_results([item.metric for item in evaluated])
        return EvaluationResult(stage, tuple(evaluated), AggregationResult(overall, {}), ())

    def _generate(self, *, stage, messages, metadata):
        exchange = (self.output_dir / f"iterations/iter_{metadata.get('iteration', -1):03d}"
                    / f"patch_attempt_{metadata.get('patch_attempt', 0):02d}_exchange.json")
        if stage == "iteration_patch" and exchange.is_file():  # a resumed iteration reuses its persisted reply
            return GenerationResult(text=json.loads(exchange.read_text(encoding="utf-8"))["assistant"])
        for attempt in range(1, LOOP.patch_attempts + 1):
            try:
                text, calls, usage = self.llm.complete([dict(m) for m in messages], tools=metadata.get("tools"),
                                                       temperature=SETTINGS["temperature"],
                                                       max_tokens=SETTINGS["max_tokens"])
            except GatewayRejected as error:  # a reply upstream rejects itself: a malformed patch, else no content
                reply = {"reasoning": f"gateway rejected the request ({error})", "patch": "(unparseable)"}
                return GenerationResult(text=json.dumps(reply) if stage == "iteration_patch" else "(empty reply)")
            try:
                tool_calls = tuple(ToolCall(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
                                   for c in calls)
                if stage == "momentum_update":  # a cut or failing tracker update is asked again; tried on a copy
                    _validate_momentum_tool_calls(tool_calls, metadata["tools"])
                    probe = copy.deepcopy(self.momentum)
                    failed = [r for r in (probe.execute_tool_call(c.name, dict(c.arguments)) for c in tool_calls)
                              if r.startswith("Unknown tool:") or " call failed:" in r]
                    if failed:
                        raise ValueError(failed[0])
                break
            except ValueError:
                if attempt == LOOP.patch_attempts:
                    raise
        if stage == "iteration_patch":
            try:
                IterationGeneration.from_raw_text(text)
            except ValueError as error:  # a malformed patch goes to the loop's own retry
                text = json.dumps({"reasoning": f"unparseable reply ({error}): {text}", "patch": "(unparseable)"})
        usage = UsageRecord(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"))
        return GenerationResult(text=text or ("" if calls else "(empty reply)"), usage=(usage,), tool_calls=tool_calls)


def prompt_context(benchmark, batch_size):
    """What upstream's ``load_benchmark_prompt_context`` builds for a packaged benchmark, from our 5 files."""
    read = lambda name: (PROMPTS / name).read_text(encoding="utf-8").strip().replace("{benchmark}", benchmark)
    taxonomy = load_taxonomy(PROMPTS)
    stage0 = Stage0PromptContext(trajectory_count=batch_size, vendor_prompt=read("vendor.md"),
                                 benchmark_domain_info_md="Training trajectories are supplied per iteration.",
                                 metric_interpretation=read("metric_interpretation.md"),
                                 outcome_label_format="outcome=success | hard=1\noutcome=failure | hard=0")
    return BenchmarkPromptContext(
        benchmark=benchmark, scope=None, stage0=stage0, taxonomy=taxonomy,
        skill_writing_policy=(CORE_PROMPTS_DIR / "skill_writing_policy.md").read_text(encoding="utf-8").strip(),
        taxonomy_section=render_taxonomy_section(taxonomy), iteration_example=read("iteration_example.md"),
        prompt_profile="v2", iteration_template_path=CORE_PROMPTS_DIR / "iteration_system_prompt.md",
        stage0_system_prompt_path=None, trajectory_compress_system_prompt=None)


def run(cell, panels, batches, llm, out):
    """Each candidate is gated on the training batch it was written from; D_sel is never read."""
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner(cell, llm, out)
    result = runner.optimize(executor.SEED, batches)
    # distinct (skill, task) rollouts: the training ones, then the gate's beyond them
    training = {(skill, task, split) for skill, task, split, role in runner.seen if role == "training"}
    gated = {(skill, task, split) for skill, task, split, role in runner.seen if role == "gate"} - training
    return {"final_skill": result.final_skill, "episodes": {"train": len(training), "gate_train": len(gated)},
            "llm": {"calls": llm.usage["calls"], "tokens": llm.usage["prompt_tokens"] + llm.usage["completion_tokens"]},
            "extra": {"stop_reason": result.stop_reason, "accepted": sum(r.accepted for r in result.iterations),
                      "gate": {"mode": "train-batch", "validation_size": None},
                      "iterations": [{"t": r.iteration + 1, "accepted": r.accepted, "reason": r.reason,
                                      "patch_mode": r.patch_mode, "attempts": r.patch_attempts,
                                      "budget": r.edit_budget, "v_ema": r.v_ema_after,
                                      "baseline": r.baseline.primary, "candidate": r.candidate.primary}
                                     for r in result.iterations]}}
