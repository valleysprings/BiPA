"""Prefix ablation analysis: bare (no "AHE evolved harness guidance:" prefix) seed run on AutoBench test
vs the three prefixed reps (rep0 = native-20261011, rep1/rep2 = test-reps). Rep pairs give the noise scale.

Run:  python -m aux.contamination_and_loading.prefix_ablation   -> OUT/contamination_and_loading/prefix-ablation-report.json
"""
import json
import math
import random
import statistics
from pathlib import Path

from src import layout

EXEC = layout.EXECUTION / "gpt-6-luna"
REP0 = EXEC / "native-seed/automationbench/native-20261011/results/evaluations/test/4fd5ca441c55-dfaebc25fa4f/evaluation.json"
REPS = EXEC / "test-reps/automationbench__native-seed__native-20261011/results/evaluations/test"
REP1 = REPS / "4fd5ca441c55-86b7f6a85ac2/evaluation.json"
REP2 = [REPS / "4fd5ca441c55-54a0f8f537c6/evaluation.json",
        REPS / "4fd5ca441c55-491f1309abd1/evaluation.json"]  # main + retry of exactly its unknowns
NEW = EXEC / "prefix-ablation/automationbench/bare-seed-20261021"


def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and v >= 0


def episode_row(ep):
    """Per-episode measurements: success, turns (unique requests), X (input tokens), first-request input tokens."""
    cap = ep.get("capture") or {}
    seen, reqs = set(), []
    for r in ep.get("requests", []):
        if r["request_id"] not in seen:
            seen.add(r["request_id"])
            reqs.append(r)
    turns = len(reqs)
    x = sum(r["input_tokens"] for r in reqs) if cap.get("requests_complete") is True \
        and all(finite(r["input_tokens"]) for r in reqs) else None
    first = reqs[0]["input_tokens"] if reqs and finite(reqs[0]["input_tokens"]) else None
    return {"success": ep.get("success"), "turns": turns, "X": x, "first": first}


def load_run(paths):
    """task_id -> episode row; later files only fill unknown/missing episodes (rep2 main + retry)."""
    rows = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        for e in data["episodes"]:
            if not isinstance(e, dict):
                continue
            t = e["task_id"]
            if t not in rows or rows[t]["success"] is None:
                rows[t] = episode_row(e)
    return rows


def load_new():
    files = sorted((NEW / "results/evaluations/test").glob("*/evaluation.json"),
                   key=lambda f: f.stat().st_mtime)
    rows = load_run(files)
    return rows, [str(f) for f in files]


def q(rows):
    known = [r for r in rows.values() if type(r["success"]) is bool]
    return (100 * sum(r["success"] is True for r in known) / len(known) if known else None,
            sum(r["success"] is True for r in known), len(known))


def paired(a, b, key):
    """Per-task paired differences a - b on tasks where both sides have the measurement."""
    out = {}
    for t, ra in a.items():
        rb = b.get(t)
        if rb is None:
            continue
        if key == "success":
            if type(ra["success"]) is bool and type(rb["success"]) is bool:
                out[t] = int(ra["success"]) - int(rb["success"])
        elif ra[key] is not None and rb[key] is not None:
            out[t] = ra[key] - rb[key]
    return out


def describe(diffs):
    vals = list(diffs.values())
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": statistics.fmean(vals), "median": statistics.median(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals), "abs_mean": statistics.fmean(abs(v) for v in vals)}


def agreement(a, b):
    both = [(ra["success"], rb["success"]) for t, ra in a.items()
            if (rb := b.get(t)) and type(ra["success"]) is bool and type(rb["success"]) is bool]
    same = sum(x == y for x, y in both)
    return {"n": len(both), "same": same, "rate": same / len(both) if both else None,
            "flips_to_success": sum((not x) and y for x, y in both),
            "flips_to_failure": sum(x and not y for x, y in both)}


def bootstrap_dq(a, b, n_boot=10000, seed=1):
    """95% percentile CI of Q(a) - Q(b), paired over tasks known on both sides."""
    tasks = [t for t, ra in a.items()
             if (rb := b.get(t)) and type(ra["success"]) is bool and type(rb["success"]) is bool]
    rng = random.Random(seed)
    n = len(tasks)
    diffs = []
    for _ in range(n_boot):
        s = sum(int(a[t]["success"]) - int(b[t]["success"]) for t in (tasks[rng.randrange(n)] for _ in range(n)))
        diffs.append(100 * s / n)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def compare(a, b):
    """Full paired comparison of run a against run b."""
    out = {"agreement": agreement(a, b)}
    qa, qb = q(a), q(b)
    out["Q"] = {"a": qa[0], "b": qb[0], "diff": (qa[0] - qb[0]) if qa[0] is not None and qb[0] is not None else None,
                "a_succ_known": f"{qa[1]}/{qa[2]}", "b_succ_known": f"{qb[1]}/{qb[2]}"}
    out["Q_diff_boot95"] = bootstrap_dq(a, b)
    for key in ("turns", "X", "first"):
        out[f"d_{key}"] = describe(paired(a, b, key))
    return out


def main():
    rep0, rep1, rep2 = load_run([REP0]), load_run([REP1]), load_run(REP2)
    new, new_files = load_new()
    reps = {"rep0": rep0, "rep1": rep1, "rep2": rep2}

    report = {"new_evaluations": new_files,
              "runs": {name: dict(zip(("Q", "successes", "known"), q(rows))) for name, rows in
                       ([*reps.items(), ("new", new)])},
              "new_vs_rep0": compare(new, rep0),
              "noise": {f"{x}_vs_{y}": compare(reps[x], reps[y])
                        for x, y in (("rep0", "rep1"), ("rep0", "rep2"), ("rep1", "rep2"))}}

    # mechanical token delta: per-task first-request input-token difference (prefix is the only prompt change),
    # and its projection over all requests of an episode (the prefix rides in every request's history)
    d_first = paired(new, rep0, "first")
    deltas = [-v for v in d_first.values()]
    report["mechanical"] = {
        "per_request_token_delta": describe({t: -v for t, v in d_first.items()}),
        "delta_hat": statistics.median(deltas) if deltas else None,
    }
    if deltas:
        d_hat = statistics.median(deltas)
        obs = paired(new, rep0, "X")
        resid = {t: v + d_hat * new[t]["turns"] for t, v in obs.items()}
        report["mechanical"]["d_X_observed_mean"] = statistics.fmean(obs.values()) if obs else None
        report["mechanical"]["d_X_residual_after_mechanical"] = describe(resid)

    out = layout.OUT / "contamination_and_loading/prefix-ablation-report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
