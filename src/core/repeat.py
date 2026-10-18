"""Complete frozen-skill repetitions, recovering only unknown slots; never optimize."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # bipa root

from src.core import experiment_config as config
from src.core.layout import relocate


def version(text):
    files = {} if text is None else {"SKILL.md": text}
    return hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def slot(row):
    return (row, 0) if isinstance(row, str) else (row["task_id"], row.get("repetition", 0))


def source(run):
    """Use the source run's frozen panel, cache and skill, never today's sampling defaults."""
    summary = json.loads((run / "summary.json").read_text())
    protocol = json.loads((run / "native_protocol.json").read_text())
    for key in ("model", "benchmark", "arm", "harness", "scale", "config_hash"):
        if summary.get(key) != protocol.get(key):
            raise ValueError(f"Summary/protocol mismatch: {key}")
    if protocol.get("protocol") != "native-v1":
        raise ValueError("Repetitions require a frozen native-v1 run")
    text = summary["final_skill"]
    if (run / "final/SKILL.md").read_text() != (text or ""):
        raise ValueError("Final skill differs from summary")
    tasks = [slot(row)[0] for row in protocol["panels"]["test"]]
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Frozen test panel must contain unique task IDs")
    return protocol, text, tasks


def evaluation_for(directories, vid, tasks, repetition):
    """Find an exact panel/version, including legacy rep-N output, without mixing repetitions."""
    expected = collections.Counter((task, repetition) for task in tasks)
    for directory in directories:
        for path in sorted((directory / "results/evaluations/test").glob("*/evaluation.json")):
            data = json.loads(path.read_text())
            if (data.get("version_id") == vid and data.get("split") == "test"
                    and collections.Counter(slot(row) for row in data.get("panel", [])) == expected
                    and valid_episode_slots(data)):
                return path, data
    return None, None


def valid_episode_slots(data):
    """Incomplete episodes are allowed; duplicate or unplanned evidence is not."""
    panel = [slot(row) for row in data.get("panel", [])]
    episodes = [slot(row) for row in data.get("episodes", [])]
    return len(set(panel)) == len(panel) and len(set(episodes)) == len(episodes) and set(episodes) <= set(panel)


def coverage(data, tasks, repetition):
    expected = collections.Counter((task, repetition) for task in tasks)
    episodes = data.get("episodes", []) if data else []
    panel_complete = bool(data) and collections.Counter(slot(row) for row in episodes) == expected
    known = [row["success"] for row in episodes
             if type(row.get("success")) is bool and row.get("outcome_valid", True) is True]
    outcomes_complete = panel_complete and len(known) == len(tasks)
    return {"repetition": repetition, "tasks": len(tasks), "known": len(known),
            "panel_complete": panel_complete, "outcomes_complete": outcomes_complete,
            "Q_known": sum(known) / len(known) if known else None}


def collect(run, out, vid, tasks, test_runs):
    runs = []
    for repetition in range(test_runs):
        # Legacy controls used rep-1/rep-2 next to the source run. Match the full panel and skill.
        directories = [run, out] if repetition == 0 else [out, run.parent / f"rep-{repetition}"]
        path, data = evaluation_for(directories, vid, tasks, repetition)
        # Supplemental evaluations are genuine unknown-only executions. Keep the original
        # known verdicts and report their provenance, without editing the source evaluation.
        episodes = list(data.get("episodes", [])) if data else []
        evidence = {slot(row): row for row in episodes}
        paths = [str(path)] if path else []
        expected = {(task, repetition) for task in tasks}
        for supplement in sorted((out / "results/evaluations/test").glob("*/evaluation.json")):
            extra = json.loads(supplement.read_text())
            panel = [slot(row) for row in extra.get("panel", [])]
            if (extra.get("version_id") != vid or extra.get("split") != "test" or not panel
                    or not valid_episode_slots(extra) or not set(panel) <= expected):
                continue
            added = False
            for row in extra.get("episodes", []):
                key = slot(row)
                if key not in panel:
                    continue
                old = evidence.get(key)
                if old is not None and type(old.get("success")) is bool and old.get("outcome_valid", True) is True:
                    continue  # an original known verdict is never overridden
                evidence[key] = row
                added = True
            if added and str(supplement) not in paths:
                paths.append(str(supplement))
        known = {key: row for key, row in evidence.items()
                 if type(row.get("success")) is bool and row.get("outcome_valid", True) is True}
        missing = [task for task in tasks if (task, repetition) not in known]
        effective = {"episodes": list(known.values())} if known else None
        runs.append({**coverage(effective, tasks, repetition),
                     "panel_complete": coverage(data, tasks, repetition)["panel_complete"] or not missing,
                     "evaluation_path": str(path) if path else None, "evaluation_paths": paths,
                     "missing_tasks": missing,
                     "_successes": sum(row["success"] is True for row in known.values()),
                     "_known_tasks": {task for task, _ in known},
                     "_unresolved": {key: (row.get("outcome_reason") or "unresolved")
                                     for key, row in evidence.items() if key not in known}})
    # A task whose episodes ran but never produced a verdict in ANY requested repetition is adjudicated
    # as a failure (counts 0, never retried). Slots without episode evidence still block and recover.
    adjudicated = {}
    if test_runs >= 2:
        for task in tasks:
            reasons = [row["_unresolved"].get((task, repetition)) if task not in row["_known_tasks"] else None
                       for repetition, row in enumerate(runs)]
            if all(type(reason) is str for reason in reasons):
                adjudicated[task] = {str(rep): reason for rep, reason in enumerate(reasons)}
    for row in runs:
        row["adjudicated_failures"] = sorted(task for task in adjudicated if task not in row["_known_tasks"])
        row["missing_tasks"] = [task for task in row["missing_tasks"] if task not in adjudicated]
        row["Q"] = row["_successes"] / len(tasks) if not row["missing_tasks"] else None
        del row["_successes"], row["_known_tasks"], row["_unresolved"]
    complete = all(row["Q"] is not None for row in runs)
    return {"source_run": str(run), "version_id": vid, "requested_repetitions": list(range(test_runs)),
            "runs": runs, "repetitions_complete": all(row["outcomes_complete"] for row in runs),
            "adjudicated_failures": adjudicated,
            "avg": sum(row["Q"] for row in runs) / test_runs if complete else None,
            "semantics": "Mean Q over all requested full panels; episodes without a verdict in every requested "
                         "repetition are adjudicated as failures; raw traces require a separate audit."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--test-runs", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.test_runs < 1 or args.slots < 1:
        parser.error("test-runs and slots must be positive")
    run = args.run.resolve()
    try:
        protocol, text, tasks = source(run)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"Source run is not ready: {exc}")
    out = config.repetition_dir(run)
    vid = version(text)
    result = collect(run, out, vid, tasks, args.test_runs)
    if args.execute:
        config.configure_environment()
        from src.utils.executor import Cell, evaluate
        cell = Cell(protocol["benchmark"], protocol["model"], protocol["harness"], out.resolve(),
                    slots=args.slots, cache_root=relocate(protocol["cache_root"]).resolve())
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
