#!/usr/bin/env python
"""Step 0 of the path analysis: dump the seed's per-task dev executions (and per-task input tokens) that the paper's selection replay uses, so that
analyze_paths.py can rebuild the same task model without importing the paper's code and without touching assets/ again.

Read-only on the run artifacts; writes only OUT/skill_paths/inputs/seed_inputs.json (idempotent: contains no extraction time; the mtimes
recorded are those of the evaluation files, which change only if a run writes again).

Run (bounded, ~1-3 min on the slow file system):  python -m aux.skill_paths.prep_inputs

What it reuses from the paper, unchanged: selection.seed_files / executions / check_inputs (DeepSeek: the seed's distinct dev executions, deduplicated by
request ids, restricted to the paper's inputs; check_inputs asserts the paper's counts), selection.fit_prior (beta-binomial ML prior over dev+test
executions), selection_replay.luna.luna_seed_files / luna_check_inputs (Luna: one dev execution per task).  The fitted priors are compared with the paper's cache
assets/cache/analysis/main/selection{,_luna}.json (`prior_fitted`, `mean_success`) and the comparison is stored.

The near-empty seed ("Solve the task.", version cf7b932190ed): its single dev execution and its single test execution are read from the fromscratch
SkillOpt run directory (the GEPA fromscratch run holds the same cached episodes; both are compared).  Nothing in the paper fits a prior for it: that is an
assumption of the path analysis (see analyze_paths.py / mech_paths.md).
"""
import sys

sys.dont_write_bytecode = True

import glob
import json
import os
from collections import defaultdict

from harness_bench.eval import version_from_files

from src import layout
from src.analysis import selection as S
from aux.selection_replay import luna as L

OUT = layout.OUT / "skill_paths/inputs"
CACHE = layout.CACHE / "analysis/main"
BENCHES = ["tau3-bench", "automationbench", "terminal-bench-2-1"]
EXEC = layout.EXECUTION
SEED_MIN = layout.ASSETS / "skills/seed-minimal/SKILL.md"


def vid(text):
    return version_from_files({"SKILL.md": text})["version_id"]


def episode_tokens(e):
    return sum(r.get("input_tokens") or 0 for r in e["requests"])


def per_task(files):
    """({task: [0/1 per distinct verified execution]}, {task: [input tokens per distinct verified execution]}, files list with mtimes)."""
    seen = defaultdict(dict)
    meta = []
    for path in files:
        meta.append({"path": os.path.relpath(path, layout.ASSETS.parent), "mtime": os.path.getmtime(path)})
        for e in json.load(open(path))["episodes"]:
            if S.verified(e):
                seen[e["task_id"]][S.signature(e)] = (int(bool(e["success"])), episode_tokens(e))
    succ = {t: [v[0] for v in runs.values()] for t, runs in seen.items()}
    toks = {t: [v[1] for v in runs.values()] for t, runs in seen.items()}
    return succ, toks, meta


def compare(ours, theirs):
    return {"ours": ours, "paper_cache": theirs, "max_abs_diff": max(abs(a - b) for a, b in zip(ours, theirs))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"note": "per-task seed executions used by analyze_paths.py; see prep_inputs.py", "deepseek-flash": {}, "gpt-6-luna": {}, "gpt-6-luna-nearempty": {}}
    sel = json.load(open(CACHE / "selection.json"))
    sel_luna = json.load(open(CACHE / "selection_luna.json"))
    for bench in BENCHES:
        # ---- DeepSeek, hand-written seed (paper inputs, asserted)
        dev_files, test_files = S.seed_files(bench)
        dev, test = S.executions(dev_files), S.executions(test_files)
        S.check_inputs(bench, dev, test)   # AssertionError if the paper's counts are not reproduced
        succ, toks, meta = per_task(dev_files)
        assert {t: v for t, v in succ.items()} == {t: v for t, v in dev.items()} or all(sorted(succ[t]) == sorted(dev[t]) for t in dev)
        pooled = [(sum(v), len(v)) for v in list(dev.values()) + list(test.values())]
        a, b = S.fit_prior(pooled)
        mean = sum(k for k, _ in pooled) / sum(n for _, n in pooled)
        out["deepseek-flash"][bench] = {
            "counts_dev_tasks_dev_exec_test_exec": [len(dev), sum(map(len, dev.values())), sum(map(len, test.values()))],
            "paper_inputs_asserted": list(S.PAPER_INPUTS[bench]),
            "dev": succ, "dev_tokens": toks, "prior_fitted": [a, b], "mean_success": mean,
            "prior_check": compare([a, b, mean], sel[bench]["prior_fitted"] + [sel[bench]["mean_success"]]),
            "dev_files": meta, "n_test_files": len(test_files)}
        # ---- Luna, hand-written seed
        try:
            ldev_files, ltest_files = L.luna_seed_files(bench)
            ldev, ltest = S.executions(ldev_files), S.executions(ltest_files)
            L.luna_check_inputs(bench, ldev, ltest)
            lpooled = [(sum(v), len(v)) for v in list(ldev.values()) + list(ltest.values())]
            la, lb = S.fit_prior(lpooled)
            lmean = sum(k for k, _ in lpooled) / sum(n for _, n in lpooled)
            lsucc, ltoks, lmeta = per_task(ldev_files)
            source = "paper functions luna_seed_files/luna_check_inputs (counts asserted)"
            check = compare([la, lb, lmean], sel_luna[bench]["prior_fitted"] + [sel_luna[bench]["mean_success"]])
        except Exception as exc:   # fall back to the cache for the prior, read the dev files by glob
            ldev_files = sorted(glob.glob(str(L.LUNA_EXECUTION / f"*/{bench}/native-2026101[12]/results/evaluations/dev/{S.SEED_VERSION}-*/evaluation.json")))
            lsucc, ltoks, lmeta = per_task(ldev_files)
            la, lb = sel_luna[bench]["prior_fitted"]
            lmean = sel_luna[bench]["mean_success"]
            source = f"fallback: dev files by glob, prior from the paper cache selection_luna.json ({type(exc).__name__}: {exc})"
            check = None
        out["gpt-6-luna"][bench] = {"counts_dev_tasks_dev_exec": [len(lsucc), sum(map(len, lsucc.values()))],
                                    "paper_inputs": list(L.LUNA_PAPER_INPUTS[bench][:2]), "source": source,
                                    "dev": lsucc, "dev_tokens": ltoks, "prior_fitted": [la, lb], "mean_success": lmean,
                                    "prior_check": check, "dev_files": lmeta}
    # ---- near-empty seed on Luna / automationbench: dev and test draw of version cf7b932190ed
    seed_vid = vid(SEED_MIN.read_text())
    runs = {o: EXEC / f"gpt-6-luna/{o}/automationbench/fromscratch" for o in ("skillopt", "gepa")}
    ne = {"seed_version_id": seed_vid}
    for o, run in runs.items():
        dev_f = sorted(glob.glob(str(run / f"results/evaluations/dev/{seed_vid[:12]}-*/evaluation.json")))
        test_f = sorted(glob.glob(str(run / f"results/evaluations/test/{seed_vid[:12]}-*/evaluation.json")))
        ds, dt, dm = per_task(dev_f)
        ts, tt, tm = per_task(test_f)
        ne[o] = {"dev": ds, "dev_tokens": dt, "dev_files": dm, "test_n_tasks": len(ts), "test_success": sum(sum(v) for v in ts.values()),
                 "test_executions": sum(map(len, ts.values())), "test_files": tm}
    assert ne["skillopt"]["dev"] == ne["gepa"]["dev"], "the two fromscratch runs do not share the seed's dev execution"
    # every dev execution of every version in the two fromscratch runs (the paper's "null plug-in" model for Luna: if no candidate differs from the seed,
    # all of them are seed draws; selection.luna_null_executions does the same for the native runs)
    all_files = sorted(f for run in runs.values() for f in glob.glob(str(run / "results/evaluations/dev/*/evaluation.json")))
    al_s, al_t, al_m = per_task(all_files)
    ne["all_dev"] = al_s
    ne["all_dev_n_files"] = len(all_files)
    ne["all_dev_executions_per_task"] = [min(map(len, al_s.values())), max(map(len, al_s.values()))]
    pooled = ne["skillopt"]["test_success"] + sum(sum(v) for v in ne["skillopt"]["dev"].values())
    n_pooled = ne["skillopt"]["test_executions"] + sum(map(len, ne["skillopt"]["dev"].values()))
    ne["dev_mean_success"] = sum(sum(v) for v in ne["skillopt"]["dev"].values()) / sum(map(len, ne["skillopt"]["dev"].values()))
    ne["pooled_mean_success_dev_and_test"] = pooled / n_pooled
    ne["n_pooled_executions"] = n_pooled
    ne["gepa_test_equals_skillopt_test"] = ne["gepa"]["test_success"] == ne["skillopt"]["test_success"] and ne["gepa"]["test_executions"] == ne["skillopt"]["test_executions"]
    out["gpt-6-luna-nearempty"]["automationbench"] = ne
    (OUT / "seed_inputs.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    for m in ("deepseek-flash", "gpt-6-luna"):
        for b, v in out[m].items():
            print(m, b, v.get("counts_dev_tasks_dev_exec_test_exec") or v["counts_dev_tasks_dev_exec"], "prior", [round(x, 3) for x in v["prior_fitted"]],
                  "mean", round(v["mean_success"], 3), "check", None if not v["prior_check"] else round(v["prior_check"]["max_abs_diff"], 6), v.get("source", ""))
    print("near-empty", {k: v for k, v in ne.items() if k not in ("skillopt", "gepa")})


if __name__ == "__main__":
    main()
