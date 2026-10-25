"""Preliminary exploration analysis for RQ4 (attractors; does a gain come with more or less exploration?).

Per (model, bench, arm), on the avg@3 test runs (dynamics.load_cell, Table 1's runs):
  attractors(task) = connected components among the task's 3 runs, two runs joined when the edit similarity
                     of their executed tool-name sequences is >= THETA (1 = all runs follow one behavior)
  D(task)          = 1 - mean pairwise edit similarity (run-to-run divergence; exploration across runs)
  B(task)          = mean over runs of distinct tool-name bigrams (breadth searched within a run)
  S(task)          = mean success over the 3 runs
Contrasts: Seed vs No skill, each learned skill vs Seed, on tasks with 3 verified runs on both sides.
Output: OUT/transfer_and_trajectories/explore.json; a summary on stdout.
Run:  python -m aux.transfer_and_trajectories.explore
"""
import json
import numpy as np

from src.analysis import dynamics as D
from src import layout

THETA = 0.5
OUT = layout.OUT / "transfer_and_trajectories/explore.json"
MODELS = ("deepseek-flash", "gpt-6-luna")
BENCHES = ("tau3-bench", "automationbench")
ARMS = ("native-noseed", "native-seed", "gepa", "skillopt", "skilladam")


def attractors(seqs):
    n = len(seqs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if D.editsim(seqs[i], seqs[j]) >= THETA:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def task_rows(reps):
    if len(reps) != 3:
        return {}
    ks = sorted(reps)
    tasks = [t for t in set.intersection(*(set(reps[k]) for k in ks))
             if all(reps[k][t]["R"] > 0 and reps[k][t]["success"] is not None for k in ks)]
    out = {}
    for t in tasks:
        seqs = [reps[k][t]["seq"] for k in ks]
        sims = [D.editsim(seqs[i], seqs[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        out[t] = {"A": attractors(seqs), "D": 1 - float(np.mean(sims)),
                  "B": float(np.mean([len(set(zip(s, s[1:]))) for s in seqs])),
                  "S": float(np.mean([float(reps[k][t]["success"]) for k in ks]))}
    return out


def contrast(own, ref):
    tasks = [t for t in own if t in ref]
    dS = np.array([own[t]["S"] - ref[t]["S"] for t in tasks])
    dD = np.array([own[t]["D"] - ref[t]["D"] for t in tasks])
    dA = np.array([own[t]["A"] - ref[t]["A"] for t in tasks])
    gain, loss = dS > 0, dS < 0
    more, less = dD > 0, dD < 0
    return {"n": len(tasks), "gain": int(gain.sum()), "loss": int(loss.sum()),
            "gain_more": int((gain & more).sum()), "gain_less": int((gain & less).sum()),
            "loss_more": int((loss & more).sum()), "loss_less": int((loss & less).sum()),
            "tie_more": int((~gain & ~loss & more).sum()), "tie_less": int((~gain & ~loss & less).sum()),
            "share_more_all": float(more.mean()), "dD_mean": float(dD.mean()), "dA_mean": float(dA.mean()),
            "converged_gains": sorted(t for t in tasks if own[t]["S"] - ref[t]["S"] > 0 and own[t]["D"] < ref[t]["D"])}


def main():
    res = {}
    for model in MODELS:
        for bench in BENCHES:
            rows = {arm: task_rows(D.load_cell(model, arm, bench)) for arm in ARMS}
            cell = {"arms": {}, "contrasts": {}}
            for arm, r in rows.items():
                a = np.array([v["A"] for v in r.values()])
                cell["arms"][arm] = {"n": len(r), "A_share": [float((a == k).mean()) for k in (1, 2, 3)],
                                     "A_mean": float(a.mean()), "D_mean": float(np.mean([v["D"] for v in r.values()])),
                                     "B_mean": float(np.mean([v["B"] for v in r.values()]))}
            cell["contrasts"]["seed_vs_noskill"] = contrast(rows["native-seed"], rows["native-noseed"])
            for arm in ARMS[2:]:
                cell["contrasts"][f"{arm}_vs_seed"] = contrast(rows[arm], rows["native-seed"])
            res[f"{model}|{bench}"] = cell
            print(model, bench)
            for arm, v in cell["arms"].items():
                print(f"   {arm:14s} n={v['n']} attractors 1/2/3 = {' / '.join(f'{100*x:.0f}%' for x in v['A_share'])} "
                      f"mean A={v['A_mean']:.2f} D={v['D_mean']:.3f} B={v['B_mean']:.1f}")
            for k, c in cell["contrasts"].items():
                print(f"   {k:22s} n={c['n']} gains {c['gain']} (more expl {c['gain_more']}, less {c['gain_less']}) "
                      f"losses {c['loss']} (more {c['loss_more']}, less {c['loss_less']}) ties more/less {c['tie_more']}/{c['tie_less']} "
                      f"dD={c['dD_mean']:+.3f} dA={c['dA_mean']:+.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"theta": THETA, "cells": res}, indent=1))


if __name__ == "__main__":
    main()
