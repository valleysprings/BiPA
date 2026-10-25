"""Independent independent replay of the best-of-K null (own code; same model as described in mech_paths.md and selection.py docs,
not a call into analyze_paths.py).  Inputs: paths/*.json (seed draw, K, observed gain) and inputs/seed_inputs.json (per-task seed executions, fitted prior)."""
import json, re
from pathlib import Path
import numpy as np
from src.layout import OUT


if __name__ == "__main__":
    SC = OUT / "skill_paths"
    runs = {}
    for p in sorted((SC / "paths").glob("*.json")):
        d = json.load(open(p)); runs[(d["model"], d["optimizer"], d["benchmark"], d["run_id"])] = d
    SI = json.load(open(SC / "inputs/seed_inputs.json"))
    rng = np.random.default_rng(777)
    DRAWS = 200_000
    def cands(d): return [s for s in d["steps"] if s["event"] == "candidate"]
    def scored(d): return [s for s in cands(d) if s.get("dev")]

    def replay(p, K, baseline, rng, chunk=20000):
        """p: (draws, tasks) success probabilities.  returns G_k matrix (draws, K): best-so-far gain over baseline (floored at 0)."""
        out = []
        for i in range(0, p.shape[0], chunk):
            pc = p[i:i + chunk]
            sc = (rng.random((pc.shape[0], K, pc.shape[1])) < pc[:, None, :]).sum(axis=2)   # (draws, K) dev scores of K fresh seed copies
            best = np.maximum.accumulate(sc, axis=1)
            out.append(np.maximum(best - baseline, 0))
        return np.concatenate(out)

    def post_p(dev, prior, draws):
        a, b = prior
        ks = sorted(dev)
        pa = np.array([a + sum(dev[t]) for t in ks]); pb = np.array([b + len(dev[t]) - sum(dev[t]) for t in ks])
        return rng.beta(pa, pb, size=(draws, len(ks)))

    bm = {"tau3-bench": "tau3-bench", "automationbench": "automationbench", "terminal-bench-2-1": "terminal-bench-2-1"}
    rec = {}   # parse record
    md = (SC / "mech_paths.md").read_text().splitlines()
    on = False
    for ln in md:
        if ln.startswith("| optimizer | run | seed draw"): on = True; continue
        if on:
            if not ln.startswith("|"): break
            if ln.startswith("|---"): continue
            c = [x.strip() for x in ln.strip("|").split("|")]
            rec[(c[0], c[1])] = (float(c[8].rstrip("*")), float(c[9].rstrip("*")))
    LAB = {"deepseek-flash": "DeepSeek", "gpt-6-luna": "Luna"}
    BL = {"tau3-bench": "tau3", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}
    OP = {"gepa": "GEPA", "skillopt": "SkillOpt"}
    print("run | K base obs_gain | P_fitted mine (record) | P_moderate mine (record) | band95 at k=K (tasks) | #k where observed best-so-far > q95")
    above = []
    for k in sorted(runs):
        if k[1] == "skilladam" or k[3] == "fromscratch": continue
        d = runs[k]
        sc = scored(d) if k[1] == "gepa" else cands(d)
        K = len(sc); base = d["seed_dev"]["n_success"]
        seq = [s["dev"]["n_success"] for s in sc]
        obs_best = np.maximum.accumulate(np.array(seq)); obs_gain = np.maximum(obs_best - base, 0)
        g = int(obs_gain[-1]); thr = max(g, 1)
        si = SI[k[0]][k[2]]
        dev = si["dev"]; prior_f = tuple(si["prior_fitted"]); m = si["mean_success"]
        res = []
        for name, prior in (("fitted", prior_f), ("moderate", (4 * m, 4 * (1 - m)))):
            p = post_p(dev, prior, DRAWS)
            G = replay(p, K, base, rng)
            P = float((G[:, -1] >= thr).mean())
            q95 = np.quantile(G, 0.95, axis=0, method="inverted_cdf") if hasattr(np, "quantile") else None
            res.append((P, G, q95))
            if name == "fitted":
                ab = [(j + 1) for j in range(K) if obs_gain[j] > q95[j]]
                above.append((LAB[k[0]], BL[k[2]], OP[k[1]], ab))
        r = rec[(OP[k[1]], f"{LAB[k[0]]} {BL[k[2]]}")]
        print(f"{OP[k[1]]:8s} {LAB[k[0]]:8s} {BL[k[2]]:5s} | K={K:2d} base={base:2d} gain={g} | {res[0][0]:.3f} ({r[0]:.3f}) | {res[1][0]:.3f} ({r[1]:.3f}) | q95_K={int(res[0][2][-1])} | above q95 at k={ab}")
    tot = sum(len(x[3]) for x in above)
    print("points where observed best-so-far gain > fitted-null 95% quantile, 12 runs:", tot, [x for x in above if x[3]])
    print("total (run,k) comparisons in the 12 runs:", sum(len(scored(runs[k])) if k[1] == 'gepa' else len(cands(runs[k])) for k in runs if k[1] in ('gepa', 'skillopt') and k[3] != 'fromscratch'))

    # ---- near-empty seed: three nulls
    print("\n== near-empty (Luna AutoBench), baseline 13/30, K = scored candidates")
    ne = SI["gpt-6-luna-nearempty"]["automationbench"]
    dev = ne["gepa"]["dev"]
    alld = ne["all_dev"]
    res_ne = {}
    for opt in ("gepa", "skillopt"):
        d = runs[("gpt-6-luna", opt, "automationbench", "fromscratch")]
        sc = scored(d) if opt == "gepa" else cands(d)
        K = len(sc); base = d["seed_dev"]["n_success"]; seq = [s["dev"]["n_success"] for s in sc]
        g = max(0, max(seq) - base)
        out = {}
        for name, m in (("moderate m=0.300", ne["pooled_mean_success_dev_and_test"]), ("moderate m=dev 0.433", ne["dev_mean_success"])):
            p = post_p(dev, (4 * m, 4 * (1 - m)), DRAWS)
            G = replay(p, K, base, rng)
            out[name] = (float((G[:, -1] >= g).mean()), float(G[:, -1].mean()))
        ks = sorted(alld)
        rate = np.array([sum(alld[t]) / len(alld[t]) for t in ks])
        p = np.broadcast_to(rate, (DRAWS, len(ks))).copy()
        G = replay(p, K, base, rng)
        out["plug-in"] = (float((G[:, -1] >= g).mean()), float(G[:, -1].mean()))
        print(opt, "K", K, "baseline", base, "obs gain", g, {k_: (round(v[0], 3), round(v[1], 2)) for k_, v in out.items()})
        res_ne[opt] = out
    print("record: GEPA 0.006 / 0.233 / 0.242, E 1.19/4.28/5.11 ; SkillOpt 0.002 / 0.074 / 0.067, E 0.44/2.59/4.28")
    nd = len(alld); print("plug-in model: tasks", nd, "executions per task", sorted({len(v) for v in alld.values()}), "overall rate %.3f" % (sum(sum(v) for v in alld.values()) / sum(len(v) for v in alld.values())))
    # seed baseline vs expectation
    print("seed dev draw 13/30 = %.3f; dev vs test: test 32/120 = %.3f" % (13 / 30, 32 / 120))
    from math import comb
    # two-sided Fisher exact on 13/30 vs 32/120
    def hyp(a, n1, n2, K):
        return comb(n1, a) * comb(n2, K - a) / comb(n1 + n2, K)
    n1, n2 = 30, 120; Kk = 13 + 32
    pobs = hyp(13, n1, n2, Kk)
    pv = sum(hyp(a, n1, n2, Kk) for a in range(0, min(n1, Kk) + 1) if hyp(a, n1, n2, Kk) <= pobs + 1e-12)
    print("Fisher exact two-sided p, seed dev 13/30 vs seed test 32/120: %.3f" % pv)
