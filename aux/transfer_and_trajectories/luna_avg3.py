"""Avg@3 status + metrics for GPT-6-Luna Table 1 cells, from test-reps artifacts.

Replicates the coverage logic of src.run repeat (base full-panel evaluation + unknown-only
supplements merged) and the stats module's metric definitions:
  Q          = 100 * mean over reps of per-rep success rate among verified outcomes (table1_q)
  rho        = geometric mean over common tasks of per-task (mean over reps) ratios vs native-seed,
               for X = input tokens, N = tool calls, R = turns (distinct requests)
  common set = both sides present with per-task-mean N, X > 0 (stats.py common_tasks)
A cell is avg@3-fillable iff all 3 reps are outcomes_complete.
Read-only wrt artifacts; writes OUT/transfer_and_trajectories/luna_avg3.json.
Run:  python -m aux.transfer_and_trajectories.luna_avg3
"""
import collections
import json
import math
from pathlib import Path

from src import layout
from src.utils.dataloader import SEED_TEXT, vid


def luna_row(ep):
    """Per-episode row of the Luna extraction:
    X = input tokens summed over the distinct completed requests (provider-failed requests carry no
    usage and were retried), N = executed tool calls of those requests, R = distinct completed requests
    (turns; stats.py's nreq analog). Episodes with no completed request are cost-missing (None) —
    the analog of stats.py dropping runs with unknown N/X. Success follows src.run.known:
    a boolean success with outcome_valid not False."""
    reqs = [r for r in {r["request_id"]: r for r in ep.get("requests", [])}.values() if r["input_tokens"] is not None]
    calls = [c for r in reqs for c in r.get("tool_calls", []) if c.get("executed") is not False]
    ok = type(ep.get("success")) is bool and ep.get("outcome_valid", True) is True
    cost_known = len(reqs) > 0
    return {"success": bool(ep["success"]) if ok else None,
            "N": len(calls) if cost_known else None,
            "X": sum(r["input_tokens"] for r in reqs) if cost_known else None,
            "R": len(reqs) if cost_known else None}

EX = layout.EXECUTION / "gpt-6-luna"
TR = EX / "test-reps"
BENCHES = ["terminal-bench-2-1", "tau3-bench", "automationbench"]
ARMS = [("native-seed", "native-20261011"), ("native-noseed", "native-20261011"),
        ("skillopt", "native-20261012"), ("gepa", "native-20261012"), ("skilladam", "native-20261012")]


def slot(row):
    return (row, 0) if isinstance(row, str) else (row["task_id"], row.get("repetition", 0))


def valid_episode_slots(data):
    panel = [slot(r) for r in data.get("panel", [])]
    episodes = [slot(r) for r in data.get("episodes", [])]
    return len(set(panel)) == len(panel) and len(set(episodes)) == len(episodes) and set(episodes) <= set(panel)


def known_row(row):
    return type(row.get("success")) is bool and row.get("outcome_valid", True) is True


def evaluation_for(directories, v, tasks, repetition):
    """As src.run.evaluations: exact full panel + version match."""
    expected = collections.Counter((t, repetition) for t in tasks)
    for directory in directories:
        if not directory or not (directory / "results/evaluations/test").is_dir():
            continue
        for path in sorted((directory / "results/evaluations/test").glob("*/evaluation.json")):
            data = json.loads(path.read_text())
            if (data.get("version_id") == v and data.get("split") == "test"
                    and collections.Counter(slot(r) for r in data.get("panel", [])) == expected
                    and valid_episode_slots(data)):
                return path, data
    return None, None


def collect_rep(run, out, v, tasks, repetition):
    """src.run.collect for one repetition: base eval + unknown-only supplements.

    Returns (base_path, [paths], {task_id: episode} known-outcome effective episodes)."""
    directories = [run, out] if repetition == 0 else [out, run.parent / f"rep-{repetition}"]
    path, data = evaluation_for(directories, v, tasks, repetition)
    episodes = list(data.get("episodes", [])) if data else []
    known = {slot(r): r for r in episodes if known_row(r)}
    paths = [str(path)] if path else []
    expected = {(t, repetition) for t in tasks}
    if out and (out / "results/evaluations/test").is_dir():
        for supplement in sorted((out / "results/evaluations/test").glob("*/evaluation.json")):
            extra = json.loads(supplement.read_text())
            panel = [slot(r) for r in extra.get("panel", [])]
            if (extra.get("version_id") != v or extra.get("split") != "test" or not panel
                    or not valid_episode_slots(extra) or not set(panel) <= expected):
                continue
            added = False
            for row in extra.get("episodes", []):
                key = slot(row)
                if key in panel and key not in known and known_row(row):
                    known[key] = row
                    added = True
            if added and str(supplement) not in paths:
                paths.append(str(supplement))
    return path, paths, {task: ep for (task, _), ep in known.items()}


def cell_data(arm, tag, bench):
    run = EX / arm / bench / tag
    out = TR / f"{bench}__{arm}__{tag}"
    out = out if out.is_dir() else None
    summary = json.loads((run / "summary.json").read_text())
    text = summary["final_skill"]
    v = vid(text)
    tasks = [slot(r)[0] for r in json.loads((run / "native_protocol.json").read_text())["panels"]["test"]]
    reps = {}
    for k in range(3):
        base, paths, known = collect_rep(run, out, v, tasks, k)
        rows = {task: luna_row(ep) for task, ep in known.items()}
        rows = {task: r for task, r in rows.items() if r["success"] is not None}
        reps[k] = {
            "found": base is not None,
            "base": str(base) if base else None,
            "paths": paths,
            "known": len(rows),
            "tasks": len(tasks),
            "outcomes_complete": len(rows) == len(tasks) and base is not None,
            "Q": 100 * sum(r["success"] for r in rows.values()) / len(rows) if rows else None,
            "cost_known": sum(1 for r in rows.values() if r["X"] is not None),
            "rows": rows,
        }
    return {"run": str(run), "out": str(out) if out else None, "vid": v, "is_seed_skill": text == SEED_TEXT,
            "n_tasks": len(tasks), "reps": reps}


def task_means(reps):
    """stats.py per_task_runs + task_means: runs need success, N, X known; S/N/X/R mean over runs."""
    per = {}
    for k in sorted(reps):
        for task, r in reps[k]["rows"].items():
            if r["N"] is None or r["X"] is None:
                continue
            per.setdefault(task, []).append(r)
    out = {}
    for task, runs in per.items():
        out[task] = {key: sum((float(r["success"]) if key == "S" else r[key]) for r in runs) / len(runs)
                     for key in ("S", "N", "X", "R")}
    return out


def rho_geo(ref, own, key):
    """stats.py common_tasks + geometric mean of per-task ratios (means over runs)."""
    logs = []
    for task in sorted(ref):
        if task not in own:
            continue
        vals = (ref[task]["N"], ref[task]["X"], own[task]["N"], own[task]["X"])
        if all(x > 0 for x in vals):
            logs.append(math.log(own[task][key] / ref[task][key]))
    return (math.exp(sum(logs) / len(logs)), len(logs)) if logs else (None, 0)


def main():
    cells = {}
    for bench in BENCHES:
        for arm, tag in ARMS:
            cells[(arm, bench)] = cell_data(arm, tag, bench)

    # cross-check against the repetitions' results.json where present
    stale = []
    for (arm, bench), c in cells.items():
        out = c["out"]
        if not out:
            continue
        rj = Path(out) / "results.json"
        if not rj.is_file():
            stale.append(f"{bench}__{arm}: no results.json")
            continue
        rep_rj = {r["repetition"]: r for r in json.loads(rj.read_text())["runs"]}
        for k, rep in c["reps"].items():
            r = rep_rj.get(k)
            if r and (r["known"] != rep["known"] or r["outcomes_complete"] != rep["outcomes_complete"]):
                stale.append(f"{bench}__{arm} rep{k}: results.json known={r['known']}/complete={r['outcomes_complete']}"
                             f" vs recomputed {rep['known']}/{rep['outcomes_complete']} (stale results.json)")

    result = {}
    for bench in BENCHES:
        seed = cells[("native-seed", bench)]
        seed_means = task_means(seed["reps"])
        seed_complete = all(seed["reps"][k]["outcomes_complete"] for k in range(3))
        row_out = {}
        for arm, _ in ARMS:
            c = cells[(arm, bench)]
            complete = [c["reps"][k]["outcomes_complete"] for k in range(3)]
            n_complete = sum(complete)
            reps_found = [k for k in range(3) if c["reps"][k]["found"] and c["reps"][k]["rows"]]
            qs = [c["reps"][k]["Q"] for k in reps_found]
            q = sum(qs) / len(qs) if qs else None
            means = task_means(c["reps"])
            if arm == "native-seed":
                rhos = {key: (1.0, len(means)) for key in ("X", "N", "R")}
            else:
                rhos = {key: rho_geo(seed_means, means, key) for key in ("X", "N", "R")}
            row_out[arm] = {
                "rep_complete": complete, "n_complete": n_complete,
                "rep_known": [f"{c['reps'][k]['known']}/{c['reps'][k]['tasks']}(cost {c['reps'][k]['cost_known']})" for k in range(3)],
                "reps_used": reps_found, "rep_Q": [None if c["reps"][k]["Q"] is None else round(c["reps"][k]["Q"], 2) for k in range(3)],
                "Q_avg3": q if n_complete == 3 else None, "Q_provisional": q,
                "seed_complete": seed_complete,
                "rho": {key: (rhos[key][0] if (n_complete == 3 and seed_complete) else None) for key in ("X", "N", "R")},
                "rho_provisional": {key: rhos[key][0] for key in ("X", "N", "R")},
                "n_common": rhos["X"][1],
                "is_seed_skill": c["is_seed_skill"],
            }
        result[bench] = row_out

    out_json = layout.OUT / "transfer_and_trajectories/luna_avg3.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"cells": result, "stale_notes": stale}, indent=1))
    print("stale/coverage notes:")
    for s in stale:
        print(" -", s)
    for bench in BENCHES:
        print(f"\n== {bench} ==")
        for arm, _ in ARMS:
            r = result[bench][arm]
            comp = "".join("Y" if x else "." for x in r["rep_complete"])
            q = f"{r['Q_avg3']:.1f}" if r["Q_avg3"] is not None else f"~{r['Q_provisional']:.1f}" if r["Q_provisional"] is not None else "-"
            rh = {k: (f"{v:.3f}" if v is not None else "-") for k, v in r["rho"].items()}
            rp = {k: (f"~{v:.3f}" if v is not None else "-") for k, v in r["rho_provisional"].items()}
            print(f"{arm:13s} reps[0,1,2]={comp} known={r['rep_known']} Q={q} rho={rh['X']} rho^turns={rh['R']} rho^calls={rh['N']} "
                  f"(prov Q={r['Q_provisional'] and round(r['Q_provisional'],1)} rho={rp['X']} R={rp['R']} N={rp['N']} n={r['n_common']}) seed_skill={r['is_seed_skill']}")


if __name__ == "__main__":
    main()
