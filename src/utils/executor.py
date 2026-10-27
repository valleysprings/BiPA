"""A skill text on a list of tasks -> per-task results, through harness-bench with the campaign's backend settings."""
import dataclasses, json, uuid
from pathlib import Path

from harness_bench.eval import NATIVE_V1_POLICY, run, verifier_detail, version_from_files
from harness_bench.eval.traces import chat_messages, trajectory_text

from src.utils.layout import CACHE, HARNESS_BENCH, SEED_FILE, relocate
from src.panels import BENCHMARKS, CONFIG

SEED = SEED_FILE.read_text(encoding="utf-8")
# Luna's gateway now and then leaves a slot unknown after both retries; a retry round granted outside the run is accepted
LUNA_POLICY = dataclasses.replace(NATIVE_V1_POLICY, reject_expanded_lineage=False)


@dataclasses.dataclass(frozen=True)
class Cell:
    benchmark: str
    model: str
    harness: str
    root: Path  # the run directory: evaluations, slot executions and logs
    cache_root: Path = CACHE.resolve()
    slots: int = 16


def settings(cell):
    """The keyword settings of harness_bench.eval.run for this cell."""
    return {"model": cell.model, "harness": cell.harness, "out": cell.root, "slots": cell.slots,
            "cache_root": cell.cache_root, "bench_root": str(HARNESS_BENCH), "injection": CONFIG["backend"]["injection"],
            "infrastructure_retries": CONFIG["backend"]["infrastructure_retries"],
            "timeout_seconds": BENCHMARKS[cell.benchmark]["timeout_seconds"],
            "policy": LUNA_POLICY if cell.model == "gpt-6-luna" else NATIVE_V1_POLICY}


def _verifier(episode):
    detail = verifier_detail(episode) or {}
    return "\n".join("\n- ".join([f"{k}:", *map(str, v)]) if isinstance(v, list) else f"{k}: {v}"
                     for k, v in detail.items() if v not in (None, "", [], {})) or None


def evaluate(cell, text, task_ids, split):
    """``split``: train | dev | test; ``task_ids``: ids, or rows {task_id, repetition}; ``text=None``: no skill."""
    result = run(version_from_files({} if text is None else {"SKILL.md": text}), cell.benchmark, split,
                 tasks=list(task_ids), **settings(cell))
    rows = []
    for row in result.rows:
        episode = result.episode(row["task_id"], row["repetition"])
        # the whole conversation of the episode's last request
        messages = [m for r in episode.get("requests", [])[-1:]
                    for m in chat_messages(json.loads(relocate(r["record_path"]).read_text("utf-8")))]
        rows.append({"id": row["task_id"], "repetition": row["repetition"], "evaluation_path": str(result.path.resolve()),
                     "success": None if row["success"] is None else int(row["success"]), "cost": row["cost"],
                     "n_tools": row["n_tools"], "input_tokens": row["input_tokens"], "messages": messages,
                     "trajectory": trajectory_text(messages), "verifier": _verifier(episode)})
    return rows


def record_round(out, arm, step, role, skill_text, evaluation_path, panel=None):
    """Write <out>/round-evidence/<id>.json: which skill a round evaluated, on which panel, and where the result is."""
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    if panel is not None and evaluation["panel"] != panel:
        raise ValueError(f"{evaluation_path} is not an evaluation of the given panel")
    folder = Path(out) / "round-evidence"
    folder.mkdir(parents=True, exist_ok=True)
    record = {"kind": "round_evaluation", "arm": arm, "step": step, "role": role,
              "skill": None if skill_text is None else {"text": skill_text},
              "evaluation_path": str(Path(evaluation_path).resolve()), "panel": evaluation["panel"],
              "split": evaluation["split"]}
    (folder / f"{uuid.uuid4().hex}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
