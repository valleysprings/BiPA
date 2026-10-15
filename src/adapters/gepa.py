"""GEPA (gepa-ai/gepa @ d771eb21) with its own loop: ``optimize_anything`` on the seed skill as one text parameter, the
train panel as dataset (iteration t reflects on B_t) and D_sel as valset. Upstream samples each parent from the D_sel
Pareto front, accepts a child on a strict minibatch-sum improvement, scores every accepted child on all of D_sel and
returns the pool's best D_sel mean. Its state, log and candidates live in ``<out>/gepa``; a rerun resumes from there."""

from collections import defaultdict
from itertools import accumulate
import json
import os
import hashlib

from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

from src.utils.executor import SEED, evaluate

OBJECTIVE = (  # slim's GEPA objective, the only text the reflection prompt adds to upstream's template
    "Improve this agent skill: a Markdown procedure file given to an LLM agent harness as guidance while it solves "
    "{benchmark} tasks. A better skill lets the agent pass more tasks, as judged by each task's own verifier, on unseen"
    " tasks of the same benchmark.\nIf any older skills are still helpful, include them as well. However, always keep "
    "the overall skills concise and comprehensive.\nAdding skills blindly will not help the agent, so only add skills "
    "that are actually helpful.")

DEVIATIONS = [
    "rollouts: a batch_evaluator over the shared harness_bench executor replaces gskill's SWE-smith Docker fitness",
    "minibatch: iteration t reflects on the shared batch B_t of b=8 tasks instead of upstream's 3 (lead 2026-10-08: 20 iterations instead of 54 for the same 160 train episodes; "
    "8 full trajectories still fit one reflection prompt under the 512k cap, 16 would not), stratified folds (one pass over 160, TB2 reshuffled passes over 62), "
    "through a stateless BatchSampler; upstream's default is epoch-shuffled batches of 3",
    "valset: the shared D_sel (30 dev tasks; TB2 all 9) instead of gskill's 50",
    "budget: T iterations, one per batch B_t (skipped ones included), i.e. the same 160 train episodes as every arm, instead of gskill's 600 metric calls; "
    "no early stopping (until 2026-10-08 a cap of 20 iterations plus D_sel early stopping left GEPA ~60 of the 160)",
    "skip_perfect_score=True, perfect_score=1.0 as in gepa.optimize and gskill (the launcher dataclass default is off)",
    "prompt: upstream's optimize_anything template with our objective sentence and no background; gskill's "
    "SWE-specific LoggingProposer prompt is not used",
    "side_info: gskill's layout, but Agent Trace is our condensed trajectory (12k chars water-filled plus the task "
    "statement up to 4k; skill and system prompt elided) instead of gskill's full '[ROLE]\\ncontent' dump, which on "
    "our cached episodes runs to a median 45k-140k and p90 300k-440k chars each, six per prompt; no Problem field (the "
    "task statement leads the trace) and no Patch; Test Output is the rendered verifier (under 200 chars on cached "
    "episodes, so gskill's head/tail-500 cut is moot)",
    "score: binary task success; an unknown verdict scores 0.0 with Status 'unknown' (gskill's setup_error convention)",
    "rollouts are memoized by (skill text, split, task): a child identical to its parent repeats the parent's episodes "
    "instead of drawing fresh ones, and the seed's episodes are the cache every arm shares",
    "reflection_lm: our recorded OpenAI-SDK client (temperature 1.0, max_tokens 32768, 6 retries) as a plain callable "
    "instead of litellm's LM (no temperature or max_tokens sent, 3 retries)",
]


class Schedule:
    """Iteration t (``state.i`` = t - 1) gets B_t: the dataset is B_1 … B_T concatenated."""

    def __init__(self, batches):
        self.step = 0
        self.ids = [list(range(end - len(b), end)) for b, end in zip(batches, accumulate(map(len, batches)))]

    def next_minibatch_ids(self, loader, state):
        self.step = state.i + 1
        return self.ids[state.i]


class IncompleteGEPAEvaluation(BaseException):
    """Abort without upstream's Exception handler advancing the proposal checkpoint."""


def side_info(row: dict, strict=False, full=False) -> tuple[float, dict]:
    """(score, side_info) in gskill's layout for one episode row of ``executor.evaluate``."""
    if strict and row["success"] is None:
        raise IncompleteGEPAEvaluation(f"Unknown GEPA outcome for task {row['id']}")
    score = float(row["success"] or 0)
    feedback = {"Status": {1: "success", 0: "failure", None: "unknown"}[row["success"]]}
    if row["verifier"]:
        feedback["Test Output"] = row["verifier"]
    if full:
        # gskill swe_fitness_fn.py: full ROLE trace; benchmark-specific Problem/Patch
        # are unavailable on these tasks, so no fictitious SWE patch is supplied.
        trace = "\n".join(f"[{m['role'].upper()}]\n{m['content']}" +
                          "".join(f"\n{c['function']['name']}({c['function']['arguments']})"
                                  for c in m.get("tool_calls", [])) for m in row["messages"])
        problem = next((m["content"] for m in row["messages"] if m["role"] == "user"), "")
        return score, {"Input": {"Task ID": row["id"], "Problem": problem},
                       "Generated Outputs": {"Agent Trace": trace}, "Feedback": feedback,
                       "scores": {"correctness": score}}
    return score, {"Input": {"Task ID": row["id"]}, "Generated Outputs": {"Agent Trace": row["trajectory"]},
                   "Feedback": feedback, "scores": {"correctness": score}}


def run(cell, panels, batches, llm, out) -> dict:
    strict = os.environ.get("HARNESS_STRICT_OUTCOMES") == "1"
    full = os.environ.get("HARNESS_FEEDBACK_MODE") == "full"
    if full:
        DEVIATIONS[:] = [item for item in DEVIATIONS if not item.startswith("side_info:")]
        DEVIATIONS.append("side_info: complete ROLE/content/tool trace and Problem; binary benchmark verifier; no SWE Patch exists")
    policy = {"strict_outcomes": strict, "unknown_outcome": "abort_and_retry_checkpoint" if strict else "score_zero",
              "reflection_failure": "upstream_default"}
    out.mkdir(parents=True, exist_ok=True)
    policy_path = out / "gepa-outcome-policy.json"
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy:  # keep the history of policy changes
        with (out / "gepa-outcome-policy-changes.jsonl").open("a") as changes:
            changes.write(policy_path.read_text().replace("\n", " ") + "\n")
    policy_path.write_text(json.dumps(policy, indent=2) + "\n")
    strict_deviation = "strict outcomes: unknown train/dev verdicts abort; resume retries the last saved proposal slot; a failed reflection call is left to upstream (iteration skipped)"
    DEVIATIONS[:] = [item for item in DEVIATIONS if not item.startswith("strict outcomes:")]
    for index, item in enumerate(DEVIATIONS):
        if item.startswith("score: binary task success;"):
            DEVIATIONS[index] = ("score: binary task success; unknown verdicts abort before entering GEPA" if strict else
                                 "score: binary task success; an unknown verdict scores 0.0 with Status 'unknown' (gskill's setup_error convention)")
    if strict:
        DEVIATIONS.append(strict_deviation)

    def abort(reason, evaluations=None):
        (out / "gepa-incomplete.json").write_text(json.dumps({"status": "incomplete", "reason": reason,
                                                             "evaluations": evaluations or [],
                                                             **policy}, indent=2) + "\n")
        raise IncompleteGEPAEvaluation(reason)

    def reflection_lm(*args, **kwargs):  # a failed reflection call raises into upstream, which skips the iteration
        return llm(*args, **kwargs)

    schedule = Schedule(batches)

    def batch_evaluator(pairs):
        groups, rows = defaultdict(list), {}
        for text, example in pairs:
            groups[text, example["split"]].append(example["id"])
        for (text, split), ids in groups.items():
            evaluated = evaluate(cell, text, ids, split)
            if strict and (len(evaluated) != len(ids) or sorted(r["id"] for r in evaluated) != sorted(ids)):
                abort("Executor returned an incomplete/mismatched panel")
            rows.update({(text, split, row["id"]): row for row in evaluated})
            if os.environ.get("HARNESS_PROTOCOL") == "native-v1":
                from src.native_trace import record_round_evaluation
                record_round_evaluation(out, arm="gepa", step=schedule.step,
                                        role="minibatch" if split == "train" else "full_selection",
                                        skill_text=text, evaluation_path=evaluated[0]["evaluation_path"])
        if strict:
            unknown = [(split, task) for (_, split, task), row in rows.items() if row["success"] is None]
            if unknown:
                affected = []
                for (text, split), ids in groups.items():
                    if any(rows[text, split, task]["success"] is None for task in ids):
                        files = {"SKILL.md": text}
                        version_id = hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False,
                                                              separators=(",", ":")).encode()).hexdigest()
                        affected.append({"version": {"version_id": version_id, "files": files},
                                         "split": split, "panel": ids})
                abort(f"Unknown GEPA outcomes after executor retries: {unknown}", affected)
        return [side_info(rows[text, example["split"], example["id"]], strict=strict, full=full) for text, example in pairs]

    train = [{"id": task, "split": "train"} for batch in batches for task in batch]
    val = [{"id": task, "split": "dev"} for task in panels["sel"]]
    config = GEPAConfig(engine=EngineConfig(run_dir=str(out / "gepa"), max_candidate_proposals=len(batches)),
                        reflection=ReflectionConfig(skip_perfect_score=True, perfect_score=1.0,
                                                    batch_sampler=schedule, reflection_lm=reflection_lm if strict else llm))
    result = optimize_anything(SEED, batch_evaluator=batch_evaluator, dataset=train, valset=val,
                               objective=OBJECTIVE.format(benchmark=cell.benchmark), config=config)
    (out / "gepa-incomplete.json").unlink(missing_ok=True)
    dev = result.num_candidates * len(val)  # full_eval: every pool member is scored once on all of D_sel
    return {"final_skill": result.best_candidate,
            "episodes": {"train": result.total_metric_calls - dev, "dev": dev},
            "llm": {"calls": llm.usage["calls"], "tokens": llm.usage["prompt_tokens"] + llm.usage["completion_tokens"]},
            "extra": {"best_idx": result.best_idx, "val_scores": result.val_aggregate_scores,
                      "parents": result.parents, "accepted": result.num_candidates - 1,
                      "metric_calls": result.total_metric_calls, "outcome_policy": policy}}
