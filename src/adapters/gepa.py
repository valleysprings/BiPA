"""GEPA (gepa-ai/gepa @ d771eb21) on the seed skill: iteration t reflects on B_t, D_sel is the valset."""

import json
from collections import defaultdict
from itertools import accumulate

from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

from src.utils import executor
from src.utils.executor import SEED, evaluate

OBJECTIVE = (
    "Improve this agent skill: a Markdown procedure file given to an LLM agent harness as guidance while it solves "
    "{benchmark} tasks. A better skill lets the agent pass more tasks, as judged by each task's own verifier, on unseen"
    " tasks of the same benchmark.\nIf any older skills are still helpful, include them as well. However, always keep "
    "the overall skills concise and comprehensive.\nAdding skills blindly will not help the agent, so only add skills "
    "that are actually helpful.")


class Schedule:
    """Upstream batch sampler: iteration t (state.i = t - 1) gets B_t of the concatenated dataset."""

    def __init__(self, batches):
        self.step = 0
        self.ids = [list(range(end - len(b), end)) for b, end in zip(batches, accumulate(map(len, batches)))]

    def next_minibatch_ids(self, loader, state):
        self.step = state.i + 1
        return self.ids[state.i]


class IncompleteGEPAEvaluation(BaseException):
    """Not an Exception, so upstream's handler cannot skip the iteration and advance the checkpoint."""


def side_info(row):
    """(score, side_info) for one episode in gskill's layout, with the whole conversation as the trace."""
    score = float(row["success"])
    feedback = {"Status": "success" if row["success"] else "failure"}
    if row["verifier"]:
        feedback["Test Output"] = row["verifier"]
    trace = "\n".join(f"[{m['role'].upper()}]\n{m['content']}" +
                      "".join(f"\n{c['function']['name']}({c['function']['arguments']})" for c in m.get("tool_calls", []))
                      for m in row["messages"])
    problem = next((m["content"] for m in row["messages"] if m["role"] == "user"), "")
    return score, {"Input": {"Task ID": row["id"], "Problem": problem}, "Generated Outputs": {"Agent Trace": trace},
                   "Feedback": feedback, "scores": {"correctness": score}}


def run(cell, panels, batches, llm, out) -> dict:
    stop = out / "gepa" / "gepa.stop"  # upstream's FileStopper ends the run whenever this file exists
    if stop.exists():
        raise RuntimeError(f"{stop} exists: a stale stop signal would end the run early; remove it to (re)start")
    schedule = Schedule(batches)

    def batch_evaluator(pairs):
        groups, rows = defaultdict(list), {}
        for text, example in pairs:
            groups[text, example["split"]].append(example["id"])
        for (text, split), ids in groups.items():
            evaluated = evaluate(cell, text, ids, split)
            rows.update({(text, split, row["id"]): row for row in evaluated})
            role = "minibatch" if split == "train" else "full_selection"
            executor.record_round(out, arm="gepa", step=schedule.step, role=role, skill_text=text,
                                  evaluation_path=evaluated[0]["evaluation_path"])
        unknown = [(split, task) for (_, split, task), row in rows.items() if row["success"] is None]
        if unknown:
            raise IncompleteGEPAEvaluation(f"Unknown GEPA outcomes after executor retries: {unknown}")
        return [side_info(rows[text, example["split"], example["id"]]) for text, example in pairs]

    train = [{"id": task, "split": "train"} for batch in batches for task in batch]
    val = [{"id": task, "split": "dev"} for task in panels["sel"]]

    def reflection_lm(prompt):  # a plain callable; when it raises, upstream skips the iteration
        return llm.complete([{"role": "user", "content": prompt}])[0]

    config = GEPAConfig(engine=EngineConfig(run_dir=str(out / "gepa"), max_candidate_proposals=len(batches)),
                        reflection=ReflectionConfig(skip_perfect_score=True, perfect_score=1.0,
                                                    batch_sampler=schedule, reflection_lm=reflection_lm))
    result = optimize_anything(SEED, batch_evaluator=batch_evaluator, dataset=train, valset=val,
                               objective=OBJECTIVE.format(benchmark=cell.benchmark), config=config)
    log = out / "gepa/run_log.json"  # one entry per iteration started
    rounds_run = len(json.loads(log.read_text())) if log.exists() else 0
    if rounds_run < len(batches):
        reason = "gepa.stop appeared during the run" if stop.exists() else "upstream loop ended early (see gepa/run_log.txt)"
        raise IncompleteGEPAEvaluation(f"GEPA ran {rounds_run} of T={len(batches)} iterations: {reason}")
    dev = result.num_candidates * len(val)  # every pool member is scored once on all of D_sel
    return {"final_skill": result.best_candidate,
            "episodes": {"train": result.total_metric_calls - dev, "dev": dev},
            "llm": {"calls": llm.usage["calls"], "tokens": llm.usage["prompt_tokens"] + llm.usage["completion_tokens"]},
            "extra": {"best_idx": result.best_idx, "val_scores": result.val_aggregate_scores,
                      "parents": result.parents, "accepted": result.num_candidates - 1,
                      "metric_calls": result.total_metric_calls, "rounds_run": rounds_run, "T": len(batches)}}
