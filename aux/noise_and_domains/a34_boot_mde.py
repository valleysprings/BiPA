"""A3 / A4: what the test-difference intervals and the MDE assume, and how much they move.
A3, for the 9 DeepSeek learned-minus-seed test differences (avg@3, per-task means over verified runs, as mech_stats b):
  SE / 95% CI from (a) paired bootstrap over tasks (paper; A3 = tasks iid from the family), (b) the same stratified by domain (dev/test panels
  are domain-stratified samples), (c) two-way bootstrap (tasks, then each arm's runs within task with replacement; double-counts run noise,
  an upper bound), (d) fixed-task SE (only run noise, the tasks held fixed; needs A1 only), (e) (a) inflated by a run-level shock variance
  2*sigma2_shock/3 estimated from a1.json (point and 95% upper).
A4: MDE avg@3 = (z.975 + z.8) sqrt(v1 / 3n) with v1 the mean squared per-task difference between two seed runs (mech_stats d): its task-bootstrap
  CI; the 2-df alternative from the variance of the seed's three run means (chi-square CI); the 10-df version pooling the five arms; per-arm v1;
  and Z * SE of each cell's task bootstrap.
Run: python -m aux.noise_and_domains.a34_boot_mde"""
import json, math, os
import numpy as np
from scipy import stats as sps
from aux.noise_and_domains import load

HERE = load.OUT
B = 10000
LEARNED = ("GEPA", "SkillOpt", "SkillAdam")

def outcomes(arm):
    runs = arm["runs"]
    per = {}
    for k in sorted(runs):
        for t, y in runs[k].items():
            per.setdefault(t, []).append(float(y))
    return per


def main():
    d = load.load()
    a1 = json.load(open(os.path.join(HERE, "a1.json")))
    Z = sps.norm.ppf(0.975) + sps.norm.ppf(0.8)
    rng = np.random.default_rng(20261012)
    out = {"A3": [], "A4": [], "shock": {}}

    for bench in load.BENCHES:
        arms = d[bench]["arms"]
        # run-level shock variance (per run mean, fraction^2): per arm var_obs - var_bern; pooled over the 5 arms, 95% upper from chi2_10
        cells = [c for c in a1["per_cell"] if c["bench"] == bench]
        vb = np.array([(c["sd_bern_pts"] / 100) ** 2 for c in cells]); vo = np.array([(c["sd_obs_pts"] / 100) ** 2 for c in cells])
        lam = vo.sum() / vb.sum() if False else float(np.mean(vo / vb))
        lam_up = 2 * 5 * lam / sps.chi2.ppf(0.025, 10)
        lam_lo = 2 * 5 * lam / sps.chi2.ppf(0.975, 10)
        s2_point = max(0.0, (lam - 1)) * vb.mean(); s2_up = max(0.0, lam_up - 1) * vb.mean()
        out["shock"][bench] = {"lambda": lam, "lambda_ci": [lam_lo, lam_up], "shock_sd_pts_point": 100 * math.sqrt(s2_point), "shock_sd_pts_upper": 100 * math.sqrt(s2_up),
                               "bern_sd_pts_mean": 100 * math.sqrt(vb.mean())}
        seed = outcomes(arms["Seed"])
        for name in LEARNED:
            own = outcomes(arms[name])
            tasks = sorted(set(seed) & set(own))
            ys = [np.array(seed[t]) for t in tasks]; yo = [np.array(own[t]) for t in tasks]
            diff = np.array([o.mean() - s.mean() for o, s in zip(yo, ys)])
            n = len(tasks)
            dq = 100 * diff.mean()
            # (a) task bootstrap
            idx = rng.integers(0, n, size=(B, n))
            ba = 100 * diff[idx].mean(axis=1)
            # (b) stratified by domain
            dom = np.array([load.domain(bench, t) for t in tasks])
            bb = np.zeros(B)
            for G in np.unique(dom):
                ii = np.where(dom == G)[0]
                pick = ii[rng.integers(0, len(ii), size=(B, len(ii)))]
                bb += diff[pick].sum(axis=1)
            bb = 100 * bb / n
            # (c) two-way: tasks, then runs within task with replacement (per arm)
            bc = np.empty(B)
            lo = np.array([len(o) for o in yo]); ls = np.array([len(s) for s in ys])
            Yo = np.full((n, 3), np.nan); Ys = np.full((n, 3), np.nan)
            for i in range(n):
                Yo[i, :lo[i]] = yo[i]; Ys[i, :ls[i]] = ys[i]
            for b in range(B):
                ti = idx[b]
                ro = (rng.random((n, 3)) * lo[ti][:, None]).astype(int)
                rs = (rng.random((n, 3)) * ls[ti][:, None]).astype(int)
                mo = np.array([Yo[t, r[:lo[t]]].mean() for t, r in zip(ti, ro)])
                ms = np.array([Ys[t, r[:ls[t]]].mean() for t, r in zip(ti, rs)])
                bc[b] = 100 * (mo - ms).mean()
            # (d) fixed-task SE: within-task run variance only (unbiased, ddof=1)
            vw = sum((o.var(ddof=1) / len(o) if len(o) > 1 else 0) + (s.var(ddof=1) / len(s) if len(s) > 1 else 0) for o, s in zip(yo, ys))
            se_fixed = 100 * math.sqrt(vw) / n
            se_a = ba.std(ddof=1)
            # (e) add the run-level shock of each arm's 3-run mean: 2 * s2 / 3
            se_e_pt = math.sqrt(se_a ** 2 + 1e4 * 2 * s2_point / 3); se_e_up = math.sqrt(se_a ** 2 + 1e4 * 2 * s2_up / 3)
            ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
            out["A3"].append({"bench": bench, "method": name, "n": n, "dQ": dq,
                              "task_boot": {"se": se_a, "ci": ci(ba)}, "strat_boot": {"se": float(bb.std(ddof=1)), "ci": ci(bb)},
                              "twoway_boot": {"se": float(bc.std(ddof=1)), "ci": ci(bc)}, "fixed_task_se": se_fixed,
                              "fixed_task_ci": [dq - 1.96 * se_fixed, dq + 1.96 * se_fixed],
                              "shock_se_point": se_e_pt, "shock_ci_point": [dq - 1.96 * se_e_pt, dq + 1.96 * se_e_pt],
                              "shock_se_upper": se_e_up, "shock_ci_upper": [dq - 1.96 * se_e_up, dq + 1.96 * se_e_up],
                              "detectable_Z_se_task_boot": Z * se_a})
        # A4
        def v1_of(arm):
            per = outcomes(arm)
            Y = np.array([v for v in per.values() if len(v) == 3])
            return Y, float(np.mean([np.mean((Y[:, i] - Y[:, j]) ** 2) for i in range(3) for j in range(i + 1, 3)]))
        Y, v1 = v1_of(arms["Seed"])
        n = len(Y)
        mde = 100 * Z * math.sqrt(v1 / (3 * n))
        bt = []
        for _ in range(B):
            Yb = Y[rng.integers(0, n, n)]
            bt.append(np.mean([np.mean((Yb[:, i] - Yb[:, j]) ** 2) for i in range(3) for j in range(i + 1, 3)]))
        bt = np.array(bt)
        mde_boot = [100 * Z * math.sqrt(np.percentile(bt, q) / (3 * n)) for q in (2.5, 97.5)]
        # 2-df: variance of the seed's three run means (includes any run-level shock)
        rm = Y.mean(axis=0)
        s2 = float(np.var(rm, ddof=1))
        mde2 = 100 * Z * math.sqrt(2 * s2 / 3)
        ci2 = [100 * Z * math.sqrt(2 * (2 * s2 / sps.chi2.ppf(q, 2)) / 3) for q in (0.975, 0.025)]
        # 10 df: pool the five arms' run-mean variances
        s2s = []
        for nm, arm in arms.items():
            Ya, _ = v1_of(arm)
            s2s.append(np.var(Ya.mean(axis=0), ddof=1))
        s2p = float(np.mean(s2s))
        mde10 = 100 * Z * math.sqrt(2 * s2p / 3)
        ci10 = [100 * Z * math.sqrt(2 * (10 * s2p / sps.chi2.ppf(q, 10)) / 3) for q in (0.975, 0.025)]
        per_arm = {nm: 100 * Z * math.sqrt(v1_of(arm)[1] / (3 * len(v1_of(arm)[0]))) for nm, arm in arms.items()}
        out["A4"].append({"bench": bench, "n": n, "v1": v1, "mde_paper": mde, "mde_v1_taskboot_ci": mde_boot,
                          "seed_run_means": (100 * rm).round(1).tolist(), "mde_2df": mde2, "mde_2df_chi2_ci": ci2,
                          "mde_10df_5arms": mde10, "mde_10df_chi2_ci": ci10, "mde_v1_per_arm": per_arm})

    json.dump(out, open(os.path.join(HERE, "a34.json"), "w"), indent=1, default=float)
    f = lambda c: f"[{c[0]:+.1f}, {c[1]:+.1f}]"
    print("shock", json.dumps(out["shock"], default=float))
    print("bench method n dQ | task-boot SE CI | strat SE CI | two-way SE CI | fixed-task SE CI | +shock(pt) SE CI | +shock(up) SE CI | Z*SE")
    for r in out["A3"]:
        print(f"{r['bench'][:6]} {r['method']:9s} {r['n']} {r['dQ']:+.1f} | {r['task_boot']['se']:.2f} {f(r['task_boot']['ci'])} | {r['strat_boot']['se']:.2f} {f(r['strat_boot']['ci'])} | {r['twoway_boot']['se']:.2f} {f(r['twoway_boot']['ci'])} | {r['fixed_task_se']:.2f} {f(r['fixed_task_ci'])} | {r['shock_se_point']:.2f} {f(r['shock_ci_point'])} | {r['shock_se_upper']:.2f} {f(r['shock_ci_upper'])} | {r['detectable_Z_se_task_boot']:.1f}")
    for r in out["A4"]:
        print(json.dumps(r, default=lambda x: round(float(x), 2)))


if __name__ == "__main__":
    main()
