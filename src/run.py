"""Run one arm on one cell and test its final skill.

    python run.py --arm {skillopt,skilladam,gepa} --benchmark B --model M --harness H [--scale main|smoke]
                  [--slots N] [--out DIR]

Every arm gets the same sample (seed skill, train panel, B_1..B_T, D_sel, test panel), executor and optimizer LLM;
``adapters/<arm>.py`` ``run(cell, panels, batches, llm, out)`` runs the optimizer's own loop and returns
{final_skill, episodes, llm, extra}. This script then evaluates the seed and the final skill on the test panel and writes ``<out>/summary.json``."""

import argparse, importlib, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.executor import SEED, Cell, evaluate
from src.utils.llm import LLM
from src.core.panels import CONFIG, batches, panels, sensitive
from src.core.layout import run_path


def episodes(out: Path) -> dict:
    """Per split: episodes the evaluations consumed and of them executed (not cache hits), retries included."""
    counts = {}
    for path in sorted((out / "results/evaluations").rglob("evaluation.json")):
        accounting = json.loads(path.read_text(encoding="utf-8"))["slot_accounting"]
        split = path.relative_to(out / "results/evaluations").parts[0]
        entry = counts.setdefault(split, {"episodes": 0, "executed": 0})
        entry["episodes"] += accounting["slots"]
        entry["executed"] += accounting["executed"]
    return counts


def score(rows: list[dict]) -> dict:
    """Mean success and mean cost over the tasks whose value is known."""
    known = [row["success"] for row in rows if row["success"] is not None]
    costs = [row["cost"] for row in rows if row["cost"] is not None]
    return {"success": sum(known) / len(known) if known else None, "known": len(known), "tasks": len(rows),
            "outcomes_complete": bool(rows) and len(known) == len(rows),
            "cost": sum(costs) / len(costs) if costs else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, choices=("none", "seed", "skillopt", "skilladam", "gepa"))
    parser.add_argument("--benchmark", required=True, choices=sorted(CONFIG["benchmarks"]))
    parser.add_argument("--model", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--scale", default="main")
    parser.add_argument("--slots", type=int, default=16, help="slot subprocesses at once")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cache-root", type=Path, help="isolated episode cache for a new execution protocol")
    parser.add_argument("--protocol", choices=("legacy", "native-v1"), default="native-v1")
    parser.add_argument("--feedback-mode", choices=("condensed", "full"), default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full-train", action="store_true", help="no gateway-sensitive train exclusion (new runs from 2026-10-05, as run_bach.py)")
    args = parser.parse_args()
    if args.scale not in CONFIG["benchmarks"][args.benchmark]["scales"]:
        parser.error(f"unknown scale {args.scale!r} for {args.benchmark}: "
                     f"have {sorted(CONFIG['benchmarks'][args.benchmark]['scales'])}")
    arm_name = args.arm
    out = (args.out or run_path(args.arm, args.model, args.benchmark, args.scale)).resolve()
    recorded = out / "native_protocol.json"  # a resumed run keeps the train exclusion it started with (none before 10-04)
    args.train_exclude = (json.loads(recorded.read_text()).get("train_exclude", []) if args.resume and recorded.is_file()
                          else [] if args.full_train else list(sensitive(args.benchmark)))
    sample = panels(args.benchmark, args.scale, tuple(args.train_exclude))
    schedule = batches(args.benchmark, args.scale, args.arm, tuple(args.train_exclude))
    protocol = None
    if args.protocol == "native-v1":
        if args.feedback_mode != "full" or args.cache_root is None or arm_name not in ("none", "seed", "skillopt", "gepa", "skilladam"):
            parser.error("native-v1 requires --feedback-mode full, --cache-root, and a native arm")
        from src.core.native_protocol import prepare, require_complete
        protocol = prepare(args, out, sample, schedule, SEED)
        sample, schedule = protocol["panels"], protocol["batches"]
        os.environ.update(HARNESS_PROTOCOL="native-v1", HARNESS_FEEDBACK_MODE="full",
                          HARNESS_STRICT_OUTCOMES="1", SKILLOPT_STRICT_OUTCOMES="1",
                          HARNESS_LLM_ATTEMPT_LEDGER="1", HARNESS_CAPTURE_PROTOCOL="native-v1",
                          HARNESS_LLM_ANSWER_REPLAY="0")
    out.mkdir(parents=True, exist_ok=True)
    cache_options = {"cache_root": args.cache_root.resolve()} if args.cache_root else {}
    cell = Cell(args.benchmark, args.model, args.harness, out, slots=args.slots, **cache_options)
    llm = LLM(args.model, out)
    arm = importlib.import_module(f"src.adapters.{arm_name}") if arm_name not in ("none", "seed") else None
    result = arm.run(cell, sample, schedule, llm, out) if arm else {
        "final_skill": None if args.arm == "none" else SEED, "episodes": {}, "extra": {}}

    if protocol and arm_name in ("skillopt", "gepa", "skilladam"):
        from src.core.native_trace import materialize_round_states
        materialize_round_states(out, arm_name)
    final = result["final_skill"]
    consumed = episodes(out)  # the arm's own evaluations, before the test evaluations below
    (out / "final").mkdir(exist_ok=True)
    (out / "final/SKILL.md").write_text(final or "", encoding="utf-8")
    test = {}
    splits = (("train", "train"), ("sel", "dev"), ("test", "test")) if args.arm == "seed" else (("test", "test"),)
    baseline_panels = {}
    for panel, split in splits:
        for name, text in ({"final": final} if arm is None else {"seed": SEED, "final": final}).items():
            rows = evaluate(cell, text, sample[panel], split)
            if protocol:
                require_complete(rows, sample[panel])
                from src.core.native_trace import record_round_evaluation
                record_round_evaluation(out, arm=arm_name, step=0 if arm is None else len(schedule),
                                        role=f"{name}_{split}", skill_text=text,
                                        evaluation_path=rows[0]["evaluation_path"])
            baseline_panels.setdefault(split, {})[name] = score(rows)
    test = baseline_panels["test"]
    summary = {"arm": args.arm, "benchmark": args.benchmark, "model": args.model, "harness": args.harness,
               "scale": args.scale, "panels": {split: len(ids) for split, ids in sample.items()}, "T": len(schedule),
               "episodes": consumed, "arm_episodes": result["episodes"], "llm": dict(llm.usage),
               "extra": result["extra"], "deviations": arm.DEVIATIONS if arm else [], "final_skill": final,
               "final_words": len((final or "").split()), "seed_words": len(SEED.split()), "test": test}
    summary.update(protocol=args.protocol, config_hash=protocol["config_hash"] if protocol else None,
                   feedback_mode=args.feedback_mode, evaluated_panels=baseline_panels)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if protocol:
        from src.dynamics.audit_native_run import certify
        certificate = certify(out)
        (out / "completion_audit.json").write_text(json.dumps(certificate, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("final_skill", "deviations", "extra")}, indent=1))


if __name__ == "__main__":
    main()
