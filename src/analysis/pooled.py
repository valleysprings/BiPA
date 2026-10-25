"""Learned skills against the seed over all cells: pooled success difference and token ratio, and task coverage (pass@3)."""
import math
import sys

import numpy as np

from src.analysis import transfer as L
from src.analysis.dynamics import load_cell, luna_seed_skill
from src.utils.dataloader import write_json

B = 10000
OPT = {"gepa": "GEPA", "skillopt": "SkillOpt", "skilladam": "SkillAdam"}
BN = {"terminal-bench-2-1": "TB2", "tau3-bench": "tau3", "automationbench": "AutoB"}
ARMS = ("native-noseed", "native-seed", "gepa", "skillopt", "skilladam")


def ds_means(arm, bench):
    """Per task (mean success, mean X over the runs that read tokens) on tasks verified in all three runs."""
    reps = load_cell("deepseek-flash", arm, bench)
    ks = sorted(reps)
    tasks = set.intersection(*(set(reps[k]) for k in ks))
    xs = {t: [reps[k][t]["X"] for k in ks if reps[k][t]["X"] > 0] for t in tasks}
    return {t: (float(np.mean([reps[k][t]["success"] for k in ks])), float(np.mean(xs[t])))
            for t in tasks if all(reps[k][t]["success"] is not None for k in ks) and xs[t]}


def luna_means(arm, tag, bench):
    return {t: (m["S"], m["X"]) for t, m in L.task_means(L.cell_data(arm, tag, bench)["reps"]).items()
            if m.get("S") is not None and m.get("X")}


def block(seed, arms):
    """Tasks of the seed and all three optimizers: (tasks x 3) matrices of dQ in points and log rho."""
    tasks = sorted(set(seed).intersection(*map(set, arms.values())))
    return {"dq": np.array([[100 * (arms[o][t][0] - seed[t][0]) for o in OPT.values()] for t in tasks]),
            "lr": np.array([[math.log(arms[o][t][1] / seed[t][1]) for o in OPT.values()] for t in tasks])}


def load():
    blocks = {}
    for bench, b in BN.items():
        blocks["DeepSeek", b] = block(ds_means("native-seed", bench), {n: ds_means(a, bench) for a, n in OPT.items()})
    for bench in ("tau3-bench", "automationbench"):
        blocks["Luna", BN[bench]] = block(luna_means("native-seed", "native-20261011", bench),
                                          {n: luna_means(a, "native-20261012", bench) for a, n in OPT.items()})
    # Luna's three optimizers returned the seed on TB2: every difference is zero over its 17 known tasks
    blocks["Luna", "TB2"] = {"dq": np.zeros((17, 3)), "lr": np.zeros((17, 3))}
    return blocks


def cell_stats(blocks, rng):
    rows = []
    for (model, bench), b in blocks.items():
        for j, o in enumerate(OPT.values()):
            d, lr = b["dq"][:, j], b["lr"][:, j]
            n = len(d)
            idx = rng.integers(0, n, (B, n))
            bd, bl = d[idx].mean(1), lr[idx].mean(1)
            rows.append({"model": model, "bench": bench, "method": o, "n": n,
                         "dQ": float(d.mean()), "lo": float(np.percentile(bd, 2.5)), "hi": float(np.percentile(bd, 97.5)),
                         "dQ_se": float(d.std(ddof=1) / math.sqrt(n)),
                         "rho": float(math.exp(lr.mean())), "rho_lo": float(math.exp(np.percentile(bl, 2.5))),
                         "rho_hi": float(math.exp(np.percentile(bl, 97.5))), "lr": float(lr.mean()),
                         "lr_se": float(lr.std(ddof=1) / math.sqrt(n))})
    return rows


def cluster(blocks, keys, rng, cols=(0, 1, 2)):
    """Task- and cell-weighted dQ and log rho, bootstrapping each block's tasks with their optimizer columns together."""
    names = ("dq_task", "dq_cell", "lr_task", "lr_cell")

    def stat(sel):
        sq, sl, n, cq, cl = 0.0, 0.0, 0, [], []
        for k, idx in sel:
            dq, lr = blocks[k]["dq"][idx][:, cols], blocks[k]["lr"][idx][:, cols]
            sq += dq.sum()
            sl += lr.sum()
            n += dq.size
            cq += list(dq.mean(0))
            cl += list(lr.mean(0))
        return sq / n, np.mean(cq), sl / n, np.mean(cl)
    point = stat([(k, np.arange(len(blocks[k]["dq"]))) for k in keys])
    draws = np.array([stat([(k, rng.integers(0, len(blocks[k]["dq"]), len(blocks[k]["dq"]))) for k in keys]) for _ in range(B)])
    out = {name: {"point": float(point[i]), "lo95": float(np.percentile(draws[:, i], 2.5)),
                  "hi95": float(np.percentile(draws[:, i], 97.5)),
                  "one_sided": float(np.percentile(draws[:, i], 95 if name.startswith("dq") else 5))}
           for i, name in enumerate(names)}
    return out | {"n_tasks": int(sum(len(blocks[k]["dq"]) for k in keys)), "n_cells": len(keys) * len(cols)}


def meta(rows, est, se):
    """Fixed-effect and DerSimonian-Laird random-effects pooling over cells."""
    rows = [r for r in rows if r[se] > 0]
    y, v = np.array([r[est] for r in rows]), np.array([r[se] ** 2 for r in rows])
    w = 1 / v
    fe, fe_se = float((w * y).sum() / w.sum()), float(math.sqrt(1 / w.sum()))
    q, k = float((w * (y - fe) ** 2).sum()), len(y)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (v + tau2)
    re, re_se = float((wr * y).sum() / wr.sum()), float(math.sqrt(1 / wr.sum()))
    return {"k": k, "FE": fe, "FE_lo": fe - 1.96 * fe_se, "FE_hi": fe + 1.96 * fe_se,
            "FE_upper1s": fe + 1.645 * fe_se, "FE_lower1s": fe - 1.645 * fe_se,
            "RE": re, "RE_lo": re - 1.96 * re_se, "RE_hi": re + 1.96 * re_se,
            "RE_upper1s": re + 1.645 * re_se, "RE_lower1s": re - 1.645 * re_se, "tau2": tau2, "Q": q,
            "I2": max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0}


def pooled():
    rng = np.random.default_rng(0)
    blocks = load()
    rows = cell_stats(blocks, rng)
    ds = [k for k in blocks if k[0] == "DeepSeek"]
    lu = [("Luna", "tau3"), ("Luna", "AutoB")]
    tb2 = [("Luna", "TB2")]
    res = {"cells": rows, "n_common": {f"{m} {b}": len(blocks[m, b]["dq"]) for m, b in blocks}}
    res["cluster"] = {name: cluster(blocks, keys, rng) for name, keys in (
        ("DeepSeek", ds), ("Luna (tau3, AutoB)", lu), ("Luna (incl. TB2 = seed)", lu + tb2),
        ("DeepSeek + Luna (tau3, AutoB)", ds + lu), ("DeepSeek + Luna (incl. TB2)", ds + lu + tb2))}
    res["per_optimizer"] = {}
    for j, o in enumerate(OPT.values()):
        res["per_optimizer"][f"DeepSeek {o}"] = cluster(blocks, ds, rng, cols=(j,))
        res["per_optimizer"][f"DeepSeek + Luna {o}"] = cluster(blocks, ds + lu, rng, cols=(j,))
    dsr = [r for r in rows if r["model"] == "DeepSeek"]
    allr = dsr + [r for r in rows if r["model"] == "Luna" and r["bench"] != "TB2"]
    res["meta"] = {"DeepSeek dQ": meta(dsr, "dQ", "dQ_se"), "DeepSeek log rho": meta(dsr, "lr", "lr_se"),
                   "DeepSeek+Luna dQ": meta(allr, "dQ", "dQ_se"), "DeepSeek+Luna log rho": meta(allr, "lr", "lr_se")}
    write_json("pooled", res)
    for name, c in res["cluster"].items():
        q, lr = c["dq_task"], c["lr_task"]
        print(f"{name:32s} tasks={c['n_tasks']:4d} cells={c['n_cells']:2d} dQ {q['point']:+5.2f} [{q['lo95']:+5.2f},{q['hi95']:+5.2f}]"
              f" rho {math.exp(lr['point']):.3f} >= {math.exp(lr['one_sided']):.3f}")


def wins(reps):
    """Successes out of three per task, over the tasks verified in all three runs."""
    if len(reps) != 3:
        return {}
    ks = sorted(reps)
    tasks = [t for t in set.intersection(*(set(reps[k]) for k in ks)) if all(reps[k][t]["success"] is not None for k in ks)]
    return {t: sum(reps[k][t]["success"] is True for k in ks) for t in tasks}


def coverage(own, ref):
    """Tasks both arms verified, and those solved in some run by one side only."""
    tasks = sorted(set(own) & set(ref))
    o, r = np.array([own[t] for t in tasks]), np.array([ref[t] for t in tasks])
    return o, r, {"n": len(tasks), "only_own": int(((o >= 1) & (r == 0)).sum()), "only_ref": int(((o == 0) & (r >= 1)).sum())}


def contrast(own, ref, rng):
    """pass@3 and pass@1 differences with 95% paired bootstrap intervals over tasks."""
    o, r, out = coverage(own, ref)
    idx = rng.integers(0, out["n"], size=(B, out["n"]))
    for key, d in (("pass3", 100 * ((o >= 1).astype(float) - (r >= 1))), ("pass1", 100 * (o - r) / 3)):
        bs = d[idx].mean(axis=1)
        out[key] = {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return out


def pass3():
    """tau3 and AutoBench, both models: pass@1, pass@3 per arm and the contrasts seed - no skill, learned - seed."""
    rng = np.random.default_rng(0)
    res = {}
    for model in ("deepseek-flash", "gpt-6-luna"):
        for bench in ("tau3-bench", "automationbench"):
            w = {arm: wins(load_cell(model, arm, bench)) for arm in ARMS}
            cell = {"arms": {arm: {"n": len(v), "pass1": 100 * float(np.mean([x / 3 for x in v.values()])),
                                   "pass3": 100 * float(np.mean([x >= 1 for x in v.values()]))} for arm, v in w.items()},
                    "contrasts": {"seed_vs_noskill": contrast(w["native-seed"], w["native-noseed"], rng)}}
            for arm in ARMS[2:]:
                cell["contrasts"][f"{arm}_vs_seed"] = contrast(w[arm], w["native-seed"], rng)
            res[f"{model}|{bench}"] = cell
    write_json("pass3_ci", {"bootstrap": B, "cells": res})
    return res


def tb2():
    """TB2 coverage counts; Luna's optimizers returned the seed there, so only its seed - no skill contrast."""
    res = {}
    for model, arms in (("deepseek-flash", ARMS), ("gpt-6-luna", ARMS[:2])):
        w = {arm: wins(load_cell(model, arm, "terminal-bench-2-1")) for arm in arms}
        pairs = [("native-seed", "native-noseed", "seed_vs_noskill")] + [(a, "native-seed", f"{a}_vs_seed") for a in arms[2:]]
        res[model] = {"n_tasks": {arm: len(v) for arm, v in w.items()},
                      "contrasts": {key: coverage(w[own], w[ref])[2] for own, ref, key in pairs}}
    res["luna_learned_returned_seed"] = {a: bool(luna_seed_skill(a, "terminal-bench-2-1")) for a in ARMS[2:]}
    write_json("gains_tb2", res)
    return res


def gains():
    """Coverage counts summed over the five pairs with learned skills: seed against no skill, learned skills against the seed."""
    cells = list(pass3().values()) + [tb2()["deepseek-flash"]]
    seed = [c["contrasts"]["seed_vs_noskill"] for c in cells]
    learned = [c["contrasts"][f"{a}_vs_seed"] for c in cells for a in ARMS[2:]]
    print(f"seed over no skill: covers {sum(c['only_own'] for c in seed)}, misses {sum(c['only_ref'] for c in seed)}; "
          f"learned over seed: cover {sum(c['only_own'] for c in learned)}, lose {sum(c['only_ref'] for c in learned)}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("pooled", "all"):
        pooled()
    if step in ("gains", "all"):
        gains()
