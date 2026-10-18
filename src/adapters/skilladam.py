"""SkillAdam, the original: upstream ``FeedbackLoop.run`` driven by the paper method's ``SkillAdamRunner``
(paper prompts ``core/prompts/*.md``, issue tracker, volatility-scaled edit budget, multi-turn optimizer history,
disjoint-validation gate on D_sel, last accepted skill returned). Only the executor, the batch sampler and the LLM are ours.

``make_runner`` builds the runner; a wrapper subclasses ``Runner`` and overrides ``_propose_patch`` (the proposal)
or ``judge`` (the gate decision)."""

import copy, json, os, random
from dataclasses import asdict, replace
from types import SimpleNamespace

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

from src.utils import executor
from src.utils.llm import GatewayRejected
from src.core.panels import ROOT

PROMPTS = ROOT.parent / "prompts/skilladam"  # ROOT is src/core; the prompts live at src/prompts
# Upstream defaults (paper profile, main_results.json / ALFWorld; the product defaults coincide): beta 0.9, base 4,
# min 1, V_max 0.1 on the binary metric, 3 patch attempts; 99 = no early stop. max/min_iterations are set to T.
LOOP = FeedbackLoopConfig(max_iterations=10, min_iterations=10, max_consecutive_failures=99, patch_attempts=3,
                          edit_budget=EditBudgetConfig("hard", v_max=0.1, base=4, minimum=1, beta=0.9))
GATE = AcceptanceGate(improvements=(ImprovementRule("primary", 0.0),))  # product default: strict batch gain > 0
SEED, VALIDATION_SIZE = 42, 10  # upstream cli: --seed 42, --validation-size 10
SETTINGS = {"scope": None, "trajectory_compression": "deterministic", "optimizer_context_limit": 110000,
            "optimizer_history_turns": None, "max_tokens": 16384, "temperature": 1.0}

DEVIATIONS = [
    "Seed: our fixed seed SKILL.md replaces Stage0 (no Stage0 call); same seed for every arm.",
    "Batches: our stratified B_1..B_T at upstream's --train-size b=10 under the shared 160-train-episode budget (one pass: "
    "τ³ T=16 over its 155 train tasks, AutoB T=15 over 148; TB2 T=18 from three reshuffled passes over its 62) replace "
    "upstream's random sampler.",
    "Validation: upstream's default disjoint mode (validation_mode='disjoint', separate pool, --seed 42, random "
    "strategy): every iteration draws validation_size = min(10, |D_sel|) ids afresh, without replacement, from the "
    "shared D_sel panel (evaluated as split 'dev') with random.Random(42); the incumbent and the candidate are both "
    "rolled out on that batch and the gate compares them (baseline = incumbent's validation rollout, not the training "
    "rollout). Upstream draws train and validation from one rng stream; ours are drawn from a validation-only stream "
    "because the training batches are ours. When |D_sel| < 10 (TB2, 9 tasks; upstream's rng.sample would raise) the whole "
    "pool is drawn each iteration, in a permuted order. Validation rollouts count as dev episodes.",
    "max_iterations = min_iterations = T (upstream CLI: 10 iterations, min 3), max_consecutive_failures = 99 (ALFWorld "
    "profile): no early stop, every budgeted batch is one iteration.",
    "Gate: upstream's product default AcceptanceGate (primary gain > 0, strict) for every benchmark; the "
    "ALFWorld profile's non-regression gate (>= 0) is not used.",
    "Executor: _evaluate_skill calls our harness_bench executor (train split for training rollouts, dev for validation) "
    "instead of upstream's EvaluationRunner/backend; per task primary = metrics.hard = success.",
    "One rollout per (skill, task, split): upstream re-rolls the gate baseline and any repeated skill; our executor's "
    "cache serves repeats, so an unchanged incumbent's validation baseline on a repeated id and an empty patch's "
    "candidate equal their earlier rollout (an empty patch is never accepted, zero volatility).",
    "Trajectories: the executor's last request (+response, observed tool results) as chat messages, condensed by "
    "upstream's deterministic condense_trajectory_md (no packaged LLM compressor); query = first user message "
    "with the delivered skill elided; final answer = last non-empty assistant text.",
    "Prompt resources: our 5 short benchmark-neutral files in prompts/skilladam/; vendor.md is a one-line stand-in "
    "for the agent system prompt upstream puts there; core skill_writing_policy and templates unchanged.",
    "LLM: our optimizer LLM over chat completions (function-calling tools for the tracker update), temperature "
    "1.0, max_tokens 16384, provider-default reasoning effort (paper profile: medium); no prompt cache.",
    "A proposal reply that is not strict {reasoning, patch} JSON counts as a malformed patch (the loop's own retry, "
    "up to 3) instead of aborting the run; an empty reply with no tool calls is stored as '(empty reply)'.",
    "A tracker (momentum) reply whose tool-call arguments do not parse (a max_tokens cut), fail upstream's schema check "
    "or fail on a copy of the tracker is asked again, up to 3 times (skilladam-reply-retries.jsonl), then the run aborts: "
    "upstream aborts at once and its resume asks again (a failed apply would replay its cached generation forever).",
    "Resume: an interrupted iteration reuses upstream's persisted patch_attempt_NN_exchange.json reply instead of a new "
    "call (upstream's RunStore refuses to rewrite it with a different reply).",
    "Feedback: HARNESS_FEEDBACK_MODE (condensed|full) does not apply; upstream's deterministic condense_trajectory_md "
    "always receives the complete messages, and its own per-field truncation is the paper's representation.",
    "optimizer_context_limit = 110000, history_turns unset (full history until the reset), as the DocVQA and "
    "OfficeQA profiles; ALFWorld's history_turns = 2 is not used.",
]


class Runner(SkillAdamRunner):
    """``SkillAdamRunner`` without its benchmark registry, backend and Stage0; its loop methods run verbatim."""

    def __init__(self, cell, llm, out):  # upstream's __init__ only validates its own adapter/backend objects
        self.cell, self.llm, self.gate = cell, llm, GATE
        self.config = SimpleNamespace(benchmark=cell.benchmark, **SETTINGS, feedback_loop=LOOP, to_dict=lambda: {
            "benchmark": cell.benchmark, **SETTINGS, "feedback_loop": asdict(self.config.feedback_loop)})
        self.context = SimpleNamespace(benchmark=cell.benchmark, method="skilladam", model=llm.model,
                                       reasoning_effort="provider-default", workers=1, seed=0)
        self.output_dir, self.momentum = out / "skilladam", MomentumTracker()
        self.strict = os.environ.get("HARNESS_STRICT_OUTCOMES") == "1"
        self.native = os.environ.get("HARNESS_PROTOCOL") == "native-v1"
        self._run_store = self._usage_ledger = self._prompt_context = self._active_iteration = None
        self._cases_by_id, self._trajectory_cache, self._optimizer_history, self._validation_rollout_counts = \
            {}, {}, [], {}

    def judge(self, baseline: MetricResult, candidate: MetricResult):
        """The gate decision on the batch (upstream ``AcceptanceGate.judge``)."""
        return self.gate.judge(baseline, candidate)

    def optimize(self, seed: str, schedule: list[list[str]], pool: list[str]):
        """Upstream ``SkillAdamRunner.run`` with an explicit initial skill, on our batches and validation pool; resumes from disk."""
        checkpoint, T = CheckpointStore(self.output_dir), len(schedule)
        resume, self.config.feedback_loop = checkpoint.exists, replace(LOOP, max_iterations=T, min_iterations=T)
        self._run_store = RunStore(self.output_dir, resume=self.output_dir.exists())
        self._store.write_text("seed.md", seed)  # adapter file: the initial skill, round 0 of the audited history
        self._usage_ledger = UsageLedger(self.output_dir / "usage.jsonl")
        self.ledger = self.output_dir / "episodes.json"  # (skill sha256, task) pairs evaluated, kept across resumes
        self.seen = {tuple(pair) for pair in json.loads(self.ledger.read_text())} if self.ledger.exists() else set()
        self._prompt_context = prompt_context(self.cell.benchmark, len(schedule[0]))
        rng, size = random.Random(SEED), min(VALIDATION_SIZE, len(pool))  # upstream's random strategy: a fresh sample per iteration
        batches = [IterationCases(tuple(b), tuple(rng.sample(pool, size))) for b in schedule]
        self._cases_by_id = {task: BenchmarkCase(task, {}) for b in batches for task in b.training_case_ids + b.validation_case_ids}
        if resume:
            self._restore_optimizer_history(checkpoint.load_payload())
            k, stale = checkpoint.load_payload()["next_iteration"], self.output_dir / "iterations-stale"
            if (part := self.output_dir / f"iterations/iter_{k:03d}").is_dir():  # an aborted attempt of the next iteration: upstream's store never overwrites its artifacts
                stale.mkdir(exist_ok=True); part.rename(stale / f"iter_{k:03d}-{len(list(stale.iterdir()))}")
        gate = SimpleNamespace(judge=self.judge, to_dict=self.gate.to_dict)
        loop = FeedbackLoop(self.config.feedback_loop, gate=gate, checkpoint_store=checkpoint, momentum=self.momentum)
        result = loop.run(initial_skill=seed, resume=resume,
                          batches=batches,
                          propose_patch=self._propose_patch, evaluate=self._evaluate_validation,
                          post_decision_hook=self._post_decision_hook(_skill_hash(seed)))
        self._store.write_text("current_skill.md", result.final_skill)
        self._store.write_text("final_skill.md", result.final_skill)
        return result

    def _evaluate_skill(self, skill: str, cases, *, stage: str) -> EvaluationResult:
        split = "train" if stage == "training_rollout" else "dev"
        found = executor.evaluate(self.cell, skill, [c.case_id for c in cases], split)
        rows, step = {row["id"]: row for row in found}, self._active_iteration + 1
        unknown = [c.case_id for c in cases if rows.get(c.case_id, {"success": None})["success"] is None]
        if self.strict and unknown:  # SkillOpt/GEPA policy: no score from a batch with an unknown verdict
            (self.cell.root / "skilladam-incomplete.json").write_text(json.dumps(
                {"status": "incomplete", "step": step, "stage": stage, "skill_sha256": _skill_hash(skill),
                 "unknown": unknown, "evaluation_path": found[0]["evaluation_path"] if found else None,
                 "semantics": "whole batch aborted before scoring; resume reruns the iteration"}, indent=2) + "\n")
            raise IncompleteSkillAdamEvaluation(f"{stage} step {step}: {len(unknown)} unknown outcomes")
        if self.native:
            from src.core.native_trace import record_round_evaluation
            role = stage if stage == "training_rollout" else (  # upstream's own count: baseline first, then candidate
                "validation_candidate" if self._validation_rollout_counts.get(step - 1) else "validation_baseline")
            record_round_evaluation(self.cell.root, arm="skilladam", step=step, role=role, skill_text=skill,
                                    evaluation_path=found[0]["evaluation_path"])
        self.seen.update((_skill_hash(skill), case.case_id, split) for case in cases)
        self.ledger.write_text(json.dumps(sorted(self.seen)))
        evaluated = []
        for case in cases:
            row = rows[case.case_id]
            hard = float(row["success"] or 0)
            task = next((m["content"] for m in row["messages"] if m["role"] == "user"), "")
            case.payload["task"] = case.payload.get("task") or executor._SKILL.sub(r"\1(the evaluated skill)\3", task)
            answer = next((m["content"] for m in reversed(row["messages"]) if m["role"] == "assistant"
                           and m["content"]), "")
            metric = MetricResult(primary=hard, metrics={"hard": hard}, case_metrics={case.case_id: {"hard": hard}})
            evaluated.append(CaseEvaluation(case.case_id, RolloutResult(case.case_id, answer, row["messages"]), metric))
        overall = aggregate_metric_results([item.metric for item in evaluated])
        return EvaluationResult(stage, tuple(evaluated), AggregationResult(overall, {}), ())

    def _generate(self, *, stage: str, messages, metadata) -> GenerationResult:
        exchange = (self.output_dir / f"iterations/iter_{metadata.get('iteration', -1):03d}"
                    / f"patch_attempt_{metadata.get('patch_attempt', 0):02d}_exchange.json")
        if stage == "iteration_patch" and exchange.is_file():  # resumed iteration: its persisted reply (DEVIATIONS)
            return GenerationResult(text=json.loads(exchange.read_text(encoding="utf-8"))["assistant"])
        for attempt in range(1, LOOP.patch_attempts + 1):
            try:
                text, calls, usage = self.llm.complete([dict(m) for m in messages], tools=metadata.get("tools"),
                                                       temperature=SETTINGS["temperature"], max_tokens=SETTINGS["max_tokens"])
            except GatewayRejected as error:  # handed back to upstream as a reply it rejects itself: an unparseable patch (its own retry), else no content
                with (self.cell.root / "skilladam-reply-retries.jsonl").open("a") as log:
                    log.write(json.dumps({"stage": stage, "iteration": metadata.get("iteration"), "attempt": attempt, "error": f"gateway: {error}"}) + "\n")
                return GenerationResult(text=json.dumps({"reasoning": f"gateway rejected the request ({error})", "patch": "(unparseable)"})
                                        if stage == "iteration_patch" else "(empty reply)")
            try:
                tool_calls = tuple(ToolCall(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
                                   for c in calls)
                if stage == "momentum_update":  # upstream's checks, and its apply on a copy of the tracker (DEVIATIONS)
                    _validate_momentum_tool_calls(tool_calls, metadata["tools"])
                    probe = copy.deepcopy(self.momentum)
                    failed = [r for r in (probe.execute_tool_call(c.name, dict(c.arguments)) for c in tool_calls)
                              if r.startswith("Unknown tool:") or " call failed:" in r]
                    if failed:
                        raise ValueError(failed[0])
                break
            except ValueError as error:  # json.JSONDecodeError included
                with (self.cell.root / "skilladam-reply-retries.jsonl").open("a") as log:
                    log.write(json.dumps({"stage": stage, "iteration": metadata.get("iteration"), "attempt": attempt,
                                          "error": str(error)[:500]}) + "\n")
                if attempt == LOOP.patch_attempts:
                    raise
        if stage == "iteration_patch":
            try:
                IterationGeneration.from_raw_text(text)
            except ValueError as error:  # a malformed patch for the loop's own retry (DEVIATIONS)
                text = json.dumps({"reasoning": f"unparseable reply ({error}): {text}", "patch": "(unparseable)"})
        return GenerationResult(
            text=text or ("" if calls else "(empty reply)"),  # upstream keeps "" beside tool calls
            usage=(UsageRecord(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens")),),
            tool_calls=tool_calls)


class IncompleteSkillAdamEvaluation(BaseException):
    """Strict abort past any upstream exception handler; the checkpoint stays at the last completed iteration."""


def prompt_context(benchmark: str, batch_size: int) -> BenchmarkPromptContext:
    """What upstream's ``load_benchmark_prompt_context`` builds from a packaged benchmark, from our 5 files."""
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


def make_runner(cell, llm, out) -> Runner:
    return Runner(cell, llm, out)


def run(cell, panels, batches, llm, out, runner=None) -> dict:
    """The gate's validation batches are drawn from ``panels['sel']`` (D_sel, rolled out as split 'dev')."""
    runner = runner or make_runner(cell, llm, out)
    policy = {"strict_outcomes": runner.strict,
              "unknown_outcome": "abort_and_rerun_iteration" if runner.strict else "score_zero"}
    DEVIATIONS[:] = [item for item in DEVIATIONS if not item.startswith("Unknown outcomes:")] + [
        "Unknown outcomes: a train or dev verdict still unknown after the executor's retries aborts the whole batch before any "
        "score (strict, as SkillOpt/GEPA); resume reruns the iteration from the last checkpoint" if runner.strict else
        "Unknown outcomes: an unknown train or dev verdict is scored hard = 0 (legacy)"]
    out.mkdir(parents=True, exist_ok=True)
    policy_path = out / "skilladam-outcome-policy.json"
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy:  # keep the history of policy changes
        with (out / "skilladam-outcome-policy-changes.jsonl").open("a") as changes:
            changes.write(policy_path.read_text().replace("\n", " ") + "\n")
    policy_path.write_text(json.dumps(policy, indent=2) + "\n")
    result = runner.optimize(executor.SEED, batches, panels["sel"])
    (out / "skilladam-incomplete.json").unlink(missing_ok=True)
    episodes = {split: sum(p[2] == split for p in runner.seen) for split in ("train", "dev")}
    return {"final_skill": result.final_skill, "episodes": episodes,
            "llm": {"calls": llm.usage["calls"], "tokens": llm.usage["prompt_tokens"] + llm.usage["completion_tokens"]},
            "extra": {"stop_reason": result.stop_reason, "accepted": sum(r.accepted for r in result.iterations),
                      "outcome_policy": policy,
                      "iterations": [{"t": r.iteration + 1, "accepted": r.accepted, "reason": r.reason,
                                      "patch_mode": r.patch_mode, "attempts": r.patch_attempts,
                                      "budget": r.edit_budget, "v_ema": r.v_ema_after,
                                      "baseline": r.baseline.primary, "candidate": r.candidate.primary}
                                     for r in result.iterations]}}
