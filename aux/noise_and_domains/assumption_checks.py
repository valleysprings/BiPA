"""Checks of the assumptions behind the selection replay and the MDE (read-only; no agent runs).

A1 (runs of a task are conditionally independent given the task, no run-level shock):
   per arm and benchmark, over tasks with three verified runs, the sample variance of the three run totals against its permutation null
   (run labels shuffled independently within each task, which is exactly A1's exchangeability).  Pooled per benchmark by summing the ratio.
A2 (task probabilities exchangeable draws from one Beta, pooled over dev and test):
   the replay's E[G0] (SkillOpt best-of-K, GEPA argmax over K) from the observed and from a random baseline, and the single-run MDE80,
   under priors that do not pool tasks: Jeffreys Beta(.5,.5), uniform Beta(1,1), a dev-only fitted prior, and the plug-in p = s/n.
Run: python -m aux.noise_and_domains.assumption_checks
"""
import json
import os
import numpy as np
from aux.noise_and_domains.load import BENCHES, OUT, load
from src.analysis import selection as S

# ---------------- A1: run-level shock -----------------------------------------------------------------------------------------------
NPERM = 20000

# ---------------- A2: replay under non-pooling priors -------------------------------------------------------------------------------
K = {"terminal-bench-2-1": {"SkillOpt": 5, "GEPA": 8}, "tau3-bench": {"SkillOpt": 4, "GEPA": 7}, "automationbench": {"SkillOpt": 4, "GEPA": 7}}
BASE = {"terminal-bench-2-1": 4, "tau3-bench": 18, "automationbench": 17}
GAIN = {"terminal-bench-2-1": {"SkillOpt": 1, "GEPA": 1}, "tau3-bench": {"SkillOpt": 4, "GEPA": 5}, "automationbench": {"SkillOpt": 2, "GEPA": 1}}
DRAWS = 50_000


def main():
    d = load()
    rng = np.random.default_rng(20261020)
    out = {"A1": {}, "A2": {}}
    for bench in BENCHES:
        rows = {}
        for arm, a in d[bench]["arms"].items():
            runs = a["runs"]
            reps = sorted(runs)
            tasks = sorted(set.intersection(*[set(runs[r]) for r in reps]))
            y = np.array([[float(runs[r][t]) for r in reps] for t in tasks])  # tasks x 3
            tot = y.sum(axis=0)
            v_obs = float(np.var(tot, ddof=1))
            within = float(np.sum(np.var(y, axis=1, ddof=1)))  # E[var(T)] under A1 equals the sum of per-task variances
            perm = np.empty(NPERM)
            idx = np.argsort(rng.random((NPERM, y.shape[0], 3)), axis=2)
            yp = np.take_along_axis(np.broadcast_to(y, (NPERM,) + y.shape), idx, axis=2)
            perm = np.var(yp.sum(axis=1), axis=1, ddof=1)
            rows[arm] = {"n": len(tasks), "totals": tot.tolist(), "var_totals": v_obs, "sum_within_var": within,
                         "ratio": v_obs / within if within else None, "p_perm": float((perm >= v_obs - 1e-12).mean()),
                         "null_mean": float(perm.mean())}
        out["A1"][bench] = rows
    for bi, bench in enumerate(BENCHES):
        dev, test = d[bench]["seed_dev"], d[bench]["seed_test_exec"]
        pooled = [(sum(v), len(v)) for v in list(dev.values()) + list(test.values())]
        fitted = S.fit_prior(pooled)
        devonly = S.fit_prior([(sum(v), len(v)) for v in dev.values()])
        testonly = S.fit_prior([(sum(v), len(v)) for v in test.values()])
        m = sum(k for k, _ in pooled) / sum(n for _, n in pooled)
        priors = {"fitted (paper)": fitted, "fitted dev only": devonly, "moderate (paper)": (4 * m, 4 * (1 - m)),
                  "Jeffreys .5,.5": (0.5, 0.5), "uniform 1,1": (1.0, 1.0), "plug-in s/n": None}
        tasks = sorted(dev)
        s = np.array([sum(dev[t]) for t in tasks]); n = np.array([len(dev[t]) for t in tasks])
        res = {"fitted_prior": fitted, "devonly_prior": devonly, "testonly_prior": testonly, "dev_exec_per_task": sorted(set(n.tolist())),
               "dev_sometimes_tasks": int(((s > 0) & (s < n)).sum()), "n_dev": len(tasks)}
        for j, (name, pr) in enumerate(priors.items()):
            S.reseed(bi, 50 + j)
            if pr is None:
                p = np.broadcast_to(s / n, (DRAWS, len(tasks))).copy()
            else:
                p = S.RNG.beta(pr[0] + s, pr[1] + n - s, size=(DRAWS, len(tasks)))
            row = {"prior": None if pr is None else [round(pr[0], 3), round(pr[1], 3)]}
            for rule in ("SkillOpt", "GEPA"):
                k = K[bench][rule]
                g_obs = S.best_of(p, k, BASE[bench]); g_rnd = S.best_of(p, k, S.outcomes(p))
                row[rule] = {"E_G0_obs_base": float(g_obs.mean()), "P_ge_g_obs_base": float((g_obs >= GAIN[bench][rule]).mean()),
                             "E_G0_rand_base": float(g_rnd.mean())}
            crit, mde = S.mde80(p[:20000])
            row["single_run_MDE80_tasks"] = mde; row["critical_net"] = crit
            row["seed_draw_quantile"] = S.mid_quantile(S.outcomes(p), BASE[bench])
            res[name] = row
        out["A2"][bench] = res

    json.dump(out, open(os.path.join(OUT, "assumption_checks.json"), "w"), indent=1, default=float)
    for bench in BENCHES:
        print("==", bench)
        for arm, r in out["A1"][bench].items():
            print(f"  A1 {arm:9s} n={r['n']:3d} totals={r['totals']} var={r['var_totals']:.2f} sum_within={r['sum_within_var']:.2f} ratio={r['ratio']:.2f} p={r['p_perm']:.3f}")
        a2 = out["A2"][bench]
        print("  priors fitted", a2["fitted_prior"], "dev-only", a2["devonly_prior"], "test-only", a2["testonly_prior"], "sometimes dev tasks", a2["dev_sometimes_tasks"])
        for name in ("fitted (paper)", "fitted dev only", "moderate (paper)", "Jeffreys .5,.5", "uniform 1,1", "plug-in s/n"):
            r = a2[name]
            print(f"  A2 {name:18s} SO E[G0|obs]={r['SkillOpt']['E_G0_obs_base']:.2f} P={r['SkillOpt']['P_ge_g_obs_base']:.2f} rand={r['SkillOpt']['E_G0_rand_base']:.2f} | "
                  f"GEPA E[G0|obs]={r['GEPA']['E_G0_obs_base']:.2f} P={r['GEPA']['P_ge_g_obs_base']:.2f} rand={r['GEPA']['E_G0_rand_base']:.2f} | MDE80={r['single_run_MDE80_tasks']} crit={r['critical_net']} q={r['seed_draw_quantile']:.2f}")


if __name__ == "__main__":
    main()
