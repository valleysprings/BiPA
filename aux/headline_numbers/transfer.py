"""Skills tested unchanged on another model or benchmark: GPT-6 Luna avg@3 on tau3 (luna) and DeepSeek TB2 <-> AutoBench (ood)."""
import collections
import json
import math
import sys

from src.analysis.common import luna_row
from aux.headline_numbers.dynamics import LUNA, collect, known, slot
from src.utils.dataloader import DS, SEED_TEXT, vid, write_json

ARMS = [("native-seed", "native-20261011"), ("native-noseed", "native-20261011"),
        ("skillopt", "native-20261012"), ("gepa", "native-20261012"), ("skilladam", "native-20261012"),
        ("skillopt", "ood-from-deepseek-flash"), ("gepa", "ood-from-deepseek-flash"),
        ("skilladam", "ood-from-deepseek-flash"), ("native-seed", "ood-seed")]
DS_RUN = "native-20261008-https"
SEED_RUNS = ("native-20261003-trace-final", "rep-1", "rep-2")
PAIRS = [("terminal-bench-2-1", "automationbench"), ("automationbench", "terminal-bench-2-1")]


def cell_data(arm, tag, bench):
    """A Luna cell's three repetitions: known outcomes only, per repetition {Q, outcomes_complete, rows}."""
    run = LUNA / arm / bench / tag
    out = LUNA / "test-reps" / f"{bench}__{arm}__{tag}"
    out = out if out.is_dir() else None
    if (run / "summary.json").is_file():
        text = json.loads((run / "summary.json").read_text())["final_skill"]
    else:  # a transfer run tests the seed or the skill DeepSeek returned
        text = SEED_TEXT if arm == "native-seed" else \
            json.loads((DS / arm / bench / DS_RUN / "summary.json").read_text())["final_skill"]
    proto = run if (run / "native_protocol.json").is_file() else LUNA / "native-seed" / bench / "native-20261011"
    tasks = [slot(r)[0] for r in json.loads((proto / "native_protocol.json").read_text())["panels"]["test"]]
    reps = {}
    for k in range(3):
        found, merged, done = collect(run, out, vid(text), tasks, k)
        rows = {t: luna_row(merged[t]) for t in done}
        reps[k] = {"found": found, "outcomes_complete": found and len(rows) == len(tasks), "rows": rows,
                   "Q": 100 * sum(r["success"] for r in rows.values()) / len(rows) if rows else None}
    return {"is_seed_skill": text == SEED_TEXT, "reps": reps}


def task_means(reps):
    """Per task: means of S, N, X, R over the repetitions whose cost is known."""
    per = collections.defaultdict(list)
    for k in sorted(reps):
        for t, r in reps[k]["rows"].items():
            if r["N"] is not None and r["X"] is not None:
                per[t].append(r)
    return {t: {key: sum(float(r["success"]) if key == "S" else r[key] for r in runs) / len(runs) for key in ("S", "N", "X", "R")}
            for t, runs in per.items()}


def rho_geo(ref, own, key):
    """Geometric mean of own / ref over the tasks both have with positive calls and input tokens."""
    logs = [math.log(own[t][key] / ref[t][key]) for t in sorted(ref)
            if t in own and all(x > 0 for x in (ref[t]["N"], ref[t]["X"], own[t]["N"], own[t]["X"]))]
    return (math.exp(sum(logs) / len(logs)), len(logs)) if logs else (None, 0)


def luna(bench="tau3-bench"):
    """Q and rho (X, N, R) of every Luna arm against the Luna seed; the avg@3 values only when all runs are complete."""
    cells = {(arm, tag): cell_data(arm, tag, bench) for arm, tag in ARMS}
    seed = cells[ARMS[0]]
    seed_means = task_means(seed["reps"])
    seed_complete = all(r["outcomes_complete"] for r in seed["reps"].values())
    rows = {}
    for (arm, tag), c in cells.items():
        complete = [c["reps"][k]["outcomes_complete"] for k in range(3)]
        qs = [r["Q"] for r in c["reps"].values() if r["found"] and r["rows"]]
        means = task_means(c["reps"])
        rhos = {key: (1.0, len(means)) if (arm, tag) == ARMS[0] else rho_geo(seed_means, means, key) for key in "XNR"}
        q = sum(qs) / len(qs) if qs else None
        rows[f"{arm}@{tag}"] = r = {
            "rep_complete": complete, "rep_Q": [None if r["Q"] is None else round(r["Q"], 2) for r in c["reps"].values()],
            "Q_avg3": q if all(complete) else None, "Q_provisional": q,
            "rho": {key: rhos[key][0] if all(complete) and seed_complete else None for key in "XNR"},
            "rho_provisional": {key: rhos[key][0] for key in "XNR"}, "n_common": rhos["X"][1], "is_seed_skill": c["is_seed_skill"]}
        print(f"{arm}@{tag:24s} Q={r['Q_provisional']} rho={r['rho_provisional']} complete={complete} n={r['n_common']}")
    write_json("luna_transfer", {bench: rows})


def ood_row(ep):
    reqs = [r for r in {r["request_id"]: r for r in ep.get("requests", [])}.values() if r["input_tokens"] is not None]
    return {"S": float(ep["success"]), "X": sum(int(r["input_tokens"]) for r in reqs),
            "N": sum(1 for r in reqs for c in r.get("tool_calls", []) if c.get("executed") is not False), "R": len(reqs)}


def reps_of(dirs):
    """{repetition: {task: row}} of known outcomes; a shallower evaluation file wins a slot."""
    reps = collections.defaultdict(dict)
    for d in dirs:
        for p in sorted(d.glob("results/evaluations/test/**/evaluation.json"), key=lambda p: len(p.parts)):
            for ep in json.loads(p.read_text())["episodes"]:
                if known(ep) and ep["task_id"] not in reps[ep.get("repetition", 0)]:
                    reps[ep.get("repetition", 0)][ep["task_id"]] = ood_row(ep)
    return reps


def means(reps):
    per = collections.defaultdict(list)
    for rows in reps.values():
        for t, r in rows.items():
            if r["N"] > 0 and r["X"] > 0:
                per[t].append(r)
    return {t: {k: sum(r[k] for r in rs) / len(rs) for k in "SXNR"} for t, rs in per.items()}


def gm(own, ref, key):
    common = [t for t in own if t in ref]
    return math.exp(sum(math.log(own[t][key] / ref[t][key]) for t in common) / len(common)), len(common)


def summarize(name, reps):
    qs = {k: 100 * sum(r["S"] for r in rows.values()) / len(rows) for k, rows in sorted(reps.items())}
    print(f"  {name}: reps={ {k: len(v) for k, v in sorted(reps.items())} } Q={ {k: round(v, 1) for k, v in qs.items()} }")


def ood():
    """Q per repetition and per-task geometric-mean ratios of each DeepSeek skill on the other benchmark."""
    seed_dirs = lambda tgt, runs=SEED_RUNS: [DS / "native-seed" / tgt / r for r in runs]
    for src, tgt in PAIRS:
        print(f"== {src} -> {tgt}")
        seed = reps_of(seed_dirs(tgt))
        summarize("seed (Table 1)", seed)
        summarize("seed (ood-bench-seed)", reps_of(seed_dirs(tgt, ["ood-bench-seed"])))
        sm = means(seed)
        for arm in ("skillopt", "gepa", "skilladam"):
            tag = f"ood-from-bench-{src}"
            reps = reps_of([DS / arm / tgt / tag, DS / f"test-reps/{tgt}__{arm}__{tag}"])
            summarize(arm, reps)
            m, m0 = means(reps), means({0: reps[0]})
            r1, r0 = {k: gm(m, sm, k) for k in "XRN"}, {k: gm(m0, sm, k) for k in "XRN"}
            print(f"    vs seed avg3: X={r1['X'][0]:.3f} R={r1['R'][0]:.3f} N={r1['N'][0]:.3f} n={r1['X'][1]} | rep0 only: "
                  f"X={r0['X'][0]:.3f} R={r0['R'][0]:.3f} N={r0['N'][0]:.3f}")
    print("== noise reference and ratios against the ood-bench-seed rerun")
    for src, tgt in PAIRS:
        seed = means(reps_of(seed_dirs(tgt)))
        single = [means(reps_of(seed_dirs(tgt, [r]))) for r in (*SEED_RUNS, "ood-bench-seed", "native-20261003")]
        oseed, old = single[3], single[4]
        print(tgt, "single seed runs vs seed avg3 X:", [round(gm(x, seed, "X")[0], 3) for x in single])
        for arm in ("skillopt", "gepa", "skilladam"):
            m = means(reps_of([DS / arm / tgt / f"ood-from-bench-{src}"]))
            print(f"  {arm}: vs ood-bench-seed X={gm(m, oseed, 'X')[0]:.3f} R={gm(m, oseed, 'R')[0]:.3f} N={gm(m, oseed, 'N')[0]:.3f}"
                  f" | vs native-20261003 X={gm(m, old, 'X')[0]:.3f} N={gm(m, old, 'N')[0]:.3f}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("luna", "all"):
        luna()
    if step in ("ood", "all"):
        ood()
