"""A1: are runs of a task conditionally independent Bernoulli(p_v) given the task (no run-level shock)?
(i) per arm x bench: variance of the 3 run-level success rates vs the variance implied by independent Bernoulli per task
    (sum_t p_t(1-p_t)/n^2 with p_t the task's 3-run mean, x 3/2 small-sample correction); exact-null permutation test that shuffles the
    run labels within each task (Monte Carlo, 20000).
(ii) shared shock across arms: run-index main effect (sum_k (sum_arms dev_{arm,k})^2) with whole-run label permutations per arm (exact,
    6^5); and a task-level co-movement test: sum over arm pairs, tasks of the covariance of run-k outcomes, null = within-task shuffles.
Run: python -m aux.noise_and_domains.a1_runshock"""
import itertools, json, os
import numpy as np
from scipy import stats as sps
from aux.noise_and_domains import load
NPERM = 20000

def matrix(arm):
    runs = arm["runs"]
    tasks = sorted(set.intersection(*[set(r) for r in runs.values()]))
    return tasks, np.array([[float(runs[k][t]) for k in sorted(runs)] for t in tasks])


def main():
    rng = np.random.default_rng(20261020)
    d = load.load()
    out = {"per_cell": [], "pooled": {}, "shared": {}}

    def perm_within(Y, nperm):
        """(nperm, 3) run totals after shuffling the run labels within every task."""
        n = Y.shape[0]
        keys = rng.random((nperm, n, 3))
        idx = np.argsort(keys, axis=2)
        Yp = np.take_along_axis(np.broadcast_to(Y, (nperm, n, 3)), idx, axis=2)
        return Yp.sum(axis=1)

    for bench in load.BENCHES:
        pooled_obs, pooled_null = 0.0, np.zeros(NPERM)
        devs = {}
        mats = {}
        for name, arm in d[bench]["arms"].items():
            tasks, Y = matrix(arm)
            mats[name] = (tasks, Y)
            n = len(tasks)
            tot = Y.sum(axis=0)
            p = Y.mean(axis=1)
            var_obs = np.var(tot / n, ddof=1)
            var_bern = (p * (1 - p)).sum() * 1.5 / n ** 2
            null_tot = perm_within(Y, NPERM)
            null_var = np.var(null_tot / n, axis=1, ddof=1)
            p_perm = float((null_var >= var_obs - 1e-12).mean())
            ratio = var_obs / var_bern
            p_chi = float(sps.chi2.sf(2 * ratio, 2))
            out["per_cell"].append({"bench": bench, "arm": name, "n": n, "run_pct": [round(100 * x / n, 1) for x in tot],
                                    "sd_obs_pts": 100 * np.sqrt(var_obs), "sd_bern_pts": 100 * np.sqrt(var_bern),
                                    "sd_perm_null_pts": 100 * np.sqrt(null_var.mean()), "var_ratio": ratio, "p_perm": p_perm, "p_chi2_2df": p_chi,
                                    "sometimes_tasks": int(((p > 0) & (p < 1)).sum())})
            pooled_obs += var_obs / var_bern
            pooled_null += null_var / var_bern
            devs[name] = tot - tot.mean()
        out["pooled"][bench] = {"sum_ratio": pooled_obs, "p_perm": float((pooled_null >= pooled_obs - 1e-12).mean()),
                                "p_chi2_10df": float(sps.chi2.sf(2 * pooled_obs, 10))}
        # (ii-a) run-index main effect across arms, exact over whole-run permutations per arm
        names = list(devs)
        D = np.array([devs[nm] for nm in names])            # arms x 3
        stat = float((D.sum(axis=0) ** 2).sum())
        perms = list(itertools.permutations(range(3)))
        null = []
        for combo in itertools.product(perms, repeat=len(names)):
            null.append(float((np.array([D[i, list(c)] for i, c in enumerate(combo)]).sum(axis=0) ** 2).sum()))
        null = np.array(null)
        # Kendall-style concordance: correlation of deviations between arm pairs
        corr = {f"{a}~{b}": float(np.corrcoef(devs[a], devs[b])[0, 1]) for a, b in itertools.combinations(names, 2)}
        # (ii-b) task-level co-movement by run index across arms: sum over arm pairs of sum_t sum_k (y_a - ybar_a)(y_b - ybar_b)
        common = sorted(set.intersection(*[set(mats[nm][0]) for nm in names]))
        Ys = np.array([[mats[nm][1][mats[nm][0].index(t)] for t in common] for nm in names])   # arms x tasks x 3
        C = Ys - Ys.mean(axis=2, keepdims=True)
        def cstat(C):
            s = 0.0
            for a, b in itertools.combinations(range(len(names)), 2):
                s += (C[a] * C[b]).sum()
            return s
        obs_c = cstat(C)
        nullc = np.empty(5000)
        for i in range(5000):
            idx = np.argsort(rng.random(C.shape), axis=2)
            nullc[i] = cstat(np.take_along_axis(C, idx, axis=2))
        # seed vs no-skill only (same day for run 0) and learned arms only
        out["shared"][bench] = {"run_deviation_tasks": {nm: devs[nm].round(2).tolist() for nm in names},
                                "runindex_stat": stat, "runindex_p_exact": float((null >= stat - 1e-9).mean()),
                                "pair_corr_of_run_deviations": corr, "mean_pair_corr": float(np.mean(list(corr.values()))),
                                "task_comove_stat": obs_c, "task_comove_null_sd": float(nullc.std()),
                                "task_comove_z": float((obs_c - nullc.mean()) / nullc.std()),
                                "task_comove_p_two_sided": float((np.abs(nullc - nullc.mean()) >= abs(obs_c - nullc.mean())).mean())}

    # all 15 cells pooled
    tot_obs = sum(c["var_ratio"] for c in out["per_cell"])
    out["all15_sum_ratio"] = tot_obs
    out["all15_p_chi2_30df"] = float(sps.chi2.sf(2 * tot_obs, 30))
    json.dump(out, open(os.path.join(load.OUT, "a1.json"), "w"), indent=1, default=float)
    print("bench arm n run% sd_obs sd_bern(x1.5) sd_perm ratio p_perm p_chi2 sometimes")
    for c in out["per_cell"]:
        print(f"{c['bench'][:6]} {c['arm']:9s} {c['n']:3d} {c['run_pct']} {c['sd_obs_pts']:.2f} {c['sd_bern_pts']:.2f} {c['sd_perm_null_pts']:.2f} {c['var_ratio']:.2f} {c['p_perm']:.3f} {c['p_chi2_2df']:.3f} {c['sometimes_tasks']}")
    for b, v in out["pooled"].items(): print("pooled", b, v)
    print("all15", tot_obs, out["all15_p_chi2_30df"])
    for b, v in out["shared"].items(): print("shared", b, json.dumps(v))


if __name__ == "__main__":
    main()
