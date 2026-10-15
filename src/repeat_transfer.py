"""Complete repetitions of a frozen transfer (OOD) evaluation, recovering only missing slots.

transfer_model.py output dirs (execution/<model>/<method>/<bench>/ood-from-<train>, ood-seed) carry no
summary.json / native_protocol.json / final/SKILL.md, so repeat.py's native-v1 source() does not apply
and fabricating those files would misstate the protocol. Instead the frozen inputs are recovered from
the run's own repetition-0 test evaluation (panel and version_id) and its hash-verified skill delivery;
repeat.collect then plans coverage and only missing (task, repetition) slots are executed, with
supplemental evidence written to test-reps/<bench>__<method>__<tag>. Known verdicts are never re-run.

Usage: python bipa/src/repeat_transfer.py --run <run dir> [--slots 8] [--test-runs 3] [--execute]
Without --execute it only prints the coverage plan (which tasks each repetition still needs).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import experiment_config as config
from src.repeat import collect, slot, valid_episode_slots, version


def source(run):
    """The transfer run's frozen rep-0 test evaluation and the exact skill text it evaluated."""
    evaluations = []
    for path in sorted((run / "results/evaluations/test").glob("*/evaluation.json")):
        data = json.loads(path.read_text())
        if data.get("split") == "test" and valid_episode_slots(data):
            evaluations.append((path, data))
    if len(evaluations) != 1:
        raise ValueError(f"Expected exactly one source test evaluation, found {len(evaluations)}")
    path, data = evaluations[0]
    tasks = [slot(row)[0] for row in data["panel"]]
    if not tasks or len(set(tasks)) != len(tasks) or {slot(row)[1] for row in data["panel"]} != {0}:
        raise ValueError("Source panel must be unique repetition-0 task IDs")
    delivery = run / "results/skills/deliveries/evaluations/test" / path.parent.name / "native/SKILL.md"
    text = delivery.read_text()
    identity = json.loads((delivery.parent.parent / "identity.json").read_text())
    if hashlib.sha256(text.encode()).hexdigest() != identity["files"]["SKILL.md"]:
        raise ValueError("Delivered skill differs from its identity record")
    if version(text) != data["version_id"]:
        raise ValueError("Delivered skill differs from the evaluated version")
    model, _, bench, _ = run.parts[-4:]
    return model, bench, data["version_id"], text, tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--test-runs", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.test_runs < 1 or args.slots < 1:
        parser.error("test-runs and slots must be positive")
    run = args.run.resolve()
    try:
        model, bench, vid, text, tasks = source(run)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"Source run is not ready: {exc}")
    out = config.repetition_dir(run)
    result = collect(run, out, vid, tasks, args.test_runs)
    if args.execute:
        config.configure_environment()
        from src.utils.executor import Cell, evaluate
        cell = Cell(bench, model, config.HARNESS, out.resolve(), slots=args.slots,
                    cache_root=config.baseline_cache(model))
        for repetition in range(args.test_runs):
            if result["runs"][repetition]["outcomes_complete"]:
                continue
            rows = [{"task_id": task, "repetition": repetition, "weight": 1.0}
                    for task in result["runs"][repetition]["missing_tasks"]]
            if not rows:
                continue
            # A returned seed follows the same path; harness-bench reuses only matching cached slots.
            evaluate(cell, text, rows, "test")
        result = collect(run, out, vid, tasks, args.test_runs)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
