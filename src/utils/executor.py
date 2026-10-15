"""A skill text on a list of tasks -> per-task results, through harness-bench's one-call skill evaluation
(harness_bench.eval.run) with this campaign's backend settings (configs/tasks.yaml) and cache."""

import dataclasses, json, os
from dataclasses import dataclass
from pathlib import Path

from harness_bench.eval import Evaluator, NATIVE_V1_POLICY, run, evaluator_for, verifier_detail, version_from_files
from harness_bench.eval.traces import SKILL_FRAME as _SKILL, chat_messages as _chat, content_text as _text
from harness_bench.eval.traces import trajectory_text as _trajectory  # shared formatting for optimizer adapters
from src.panels import CONFIG
from src.layout import CACHE, SKILLS, relocate

SEED = (SKILLS / "seed/SKILL.md").read_text(encoding="utf-8")


@dataclass(frozen=True)
class Cell:
    benchmark: str
    model: str
    harness: str
    root: Path  # the run directory (absolute): evaluations, slot executions and slot logs
    cache_root: Path = CACHE.resolve()
    slots: int = 16


# gpt-6-luna (lead 2026-10-12): the gateway now and then fails an episode's requests (HTTP 400 invalid_encrypted_content when one
# conversation is served by different deployments), so a slot can stay unknown after native-v1's two retries.  An unknown slot may get
# one more retry round, granted outside the run and recorded in the evaluation's extra_rounds; the resumed run accepts that lineage.
# Known outcomes are never re-run.  Every other native-v1 rule is unchanged.
LUNA_NATIVE_POLICY = dataclasses.replace(NATIVE_V1_POLICY, reject_expanded_lineage=False)


def _settings(cell: Cell) -> dict:
    """This campaign's settings for harness_bench.eval.run / evaluator_for."""
    backend = CONFIG["backend"]
    # native-v1: unknown slots retried twice on every split, no extra round on resume, exhaustion report
    policy = None
    if os.environ.get("HARNESS_PROTOCOL") == "native-v1":
        policy = LUNA_NATIVE_POLICY if cell.model == "gpt-6-luna" else NATIVE_V1_POLICY
    return {"model": cell.model, "harness": cell.harness, "out": cell.root, "slots": cell.slots,
            "cache_root": cell.cache_root, "bench_root": backend["bench_root"], "injection": backend["injection"],
            "infrastructure_retries": backend["infrastructure_retries"],
            "timeout_seconds": CONFIG["benchmarks"][cell.benchmark]["timeout_seconds"], "policy": policy}


def evaluator(cell: Cell) -> Evaluator:
    """The cell's Evaluator, the same one ``evaluate`` runs through (shared per process by harness-bench)."""
    return evaluator_for(cell.benchmark, **_settings(cell))


def evaluate(cell: Cell, text: str | None, task_ids: list, split: str) -> list[dict]:
    """``split``: train | dev | test; ``task_ids``: ids, or rows {task_id, repetition} for a fresh sample;
    ``text=None``: no skill file is delivered (the main table's "no skill" row)."""
    skill_version = version_from_files({} if text is None else {"SKILL.md": text})  # sha256 of canonical JSON
    result = run(skill_version, cell.benchmark, split, tasks=list(task_ids), **_settings(cell))
    rows = []
    for row in result.rows:
        episode = result.episode(row["task_id"], row["repetition"])
        messages = [m for r in episode.get("requests", [])[-1:] for m in _chat(json.loads(Path(relocate(r["record_path"])).read_text("utf-8")))]  # the final request's whole conversation; relocate: records of runs moved into execution/ keep their old paths
        rows.append({"id": row["task_id"], "repetition": row["repetition"], "evaluation_path": str(result.path.resolve()),
                     "success": None if row["success"] is None else int(row["success"]),
                     "cost": row["cost"], "n_tools": row["n_tools"], "input_tokens": row["input_tokens"],
                     "messages": messages, "trajectory": _trajectory(messages),
                     "verifier": "\n".join("\n- ".join([f"{k}:", *map(str, v)]) if isinstance(v, list) else f"{k}: {v}"
                                           for k, v in (verifier_detail(episode) or {}).items()
                                           if v not in (None, "", [], {})) or None})
    return rows
