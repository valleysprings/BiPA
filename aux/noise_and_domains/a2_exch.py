"""A2: are task success probabilities exchangeable draws from one Beta?
(i) LRT of per-group beta-binomial fits vs one pooled fit on the seed's executions (dev + test, the replay's input), groups = domain,
    panel (dev / test), domain x panel; p from chi-square and from a parametric bootstrap under the pooled fit (300 draws).
(ii) goodness of fit of the pooled beta-binomial to the (n, s) counts (Pearson, parametric bootstrap) and Tarone's binomial-homogeneity Z.
(iii) the replay (SkillOpt best-of-K, GEPA argmax, SkillAdam with reuse, single-dev-run MDE80) under alternative priors:
    pooled fitted (paper), per-domain fitted, per-panel (dev-only), Jeffreys Beta(.5,.5) and uniform Beta(1,1) per task (no pooling).
Run: python -m aux.noise_and_domains.a2_exch"""
import json, math, os
import numpy as np
from scipy import optimize, special, stats as sps
from aux.noise_and_domains import load
from aux.selection_replay import selection as S
DRAWS = 40000
NBOOT = 300

def bb_ll(a, b, k, n):
    return float((special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)
                  + special.betaln(k + a, n - k + b) - special.betaln(a, b)).sum())


def main():
    d = load.load()
    rng = np.random.default_rng(20261020)

    def fit(k, n):
        """ML beta-binomial (a, b, loglik); grid start then Nelder-Mead on (logit m, log c)."""
        best = None
        for m in np.linspace(0.02, 0.98, 25):
            for lc in np.linspace(-3, 6, 19):
                v = bb_ll(m * math.exp(lc), (1 - m) * math.exp(lc), k, n)
                if best is None or v > best[0]:
                    best = (v, m, lc)
        def nll(x):
            m = special.expit(x[0]); c = math.exp(min(x[1], 12))
            return -bb_ll(m * c, (1 - m) * c, k, n)
        r = optimize.minimize(nll, [special.logit(best[1]), best[2]], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
        m = special.expit(r.x[0]); c = math.exp(min(r.x[1], 12))
        return m * c, (1 - m) * c, -r.fun

    def lrt(k, n, g):
        a, b, l0 = fit(k, n)
        l1 = sum(fit(k[g == G], n[g == G])[2] for G in np.unique(g))
        return 2 * (l1 - l0), (a, b)

    out = {"lrt": [], "gof": [], "replay": []}
    for bench in load.BENCHES:
        dev, test = d[bench]["seed_dev"], d[bench]["seed_test_exec"]
        tasks = [(t, "dev") for t in sorted(dev)] + [(t, "test") for t in sorted(test)]
        k = np.array([sum((dev if p == "dev" else test)[t]) for t, p in tasks])
        n = np.array([len((dev if p == "dev" else test)[t]) for t, p in tasks])
        dom = np.array([load.domain(bench, t) for t, _ in tasks])
        pan = np.array([p for _, p in tasks])
        a0, b0, l0 = fit(k, n)
        groupings = {"panel": pan}
        if bench != "terminal-bench-2-1":
            groupings["domain"] = dom
            groupings["domain x panel"] = np.array([x + "|" + y for x, y in zip(dom, pan)])
        for gname, g in groupings.items():
            stat, _ = lrt(k, n, g)
            G = len(np.unique(g))
            null = []
            for _ in range(NBOOT):
                p = rng.beta(a0, b0, size=len(k))
                kk = rng.binomial(n, p)
                null.append(lrt(kk, n, g)[0])
            null = np.array(null)
            per = {G_: [round(x, 3) for x in fit(k[g == G_], n[g == G_])[:2]] + [round(float(k[g == G_].sum() / n[g == G_].sum()), 3), int((g == G_).sum())] for G_ in np.unique(g)}
            out["lrt"].append({"bench": bench, "grouping": gname, "groups": G, "LR": stat, "df": 2 * (G - 1),
                               "p_chi2": float(sps.chi2.sf(stat, 2 * (G - 1))), "p_boot": float((null >= stat).mean()), "per_group_ab_mean_n": per})
        # (ii) GOF over (n, s) cells
        cells = sorted({(int(nn), int(s)) for nn in np.unique(n) for s in range(nn + 1)})
        def counts(kk):
            return np.array([((n == nn) & (kk == s)).sum() for nn, s in cells], float)
        def expected(a, b):
            return np.array([(n == nn).sum() * math.exp(bb_ll(a, b, np.array([s]), np.array([nn]))) for nn, s in cells])
        obs = counts(k); exp = expected(a0, b0)
        pear = float(((obs - exp) ** 2 / exp).sum())
        nullp = []
        for _ in range(1000):
            kk = rng.binomial(n, rng.beta(a0, b0, size=len(k)))
            aa, bb, _ = fit(kk, n)
            e = expected(aa, bb)
            nullp.append(((counts(kk) - e) ** 2 / e).sum())
        pbar = k.sum() / n.sum()
        # Tarone (1979) Z for extra-binomial variation
        S_ = (((k - n * pbar) ** 2) / (pbar * (1 - pbar))).sum()
        Z = (S_ - n.sum()) / math.sqrt(2 * (n * (n - 1)).sum())
        # dispersion of the per-task 3-run mean beyond the beta-binomial: var(s|n=3) observed vs predicted
        sel = n == 3
        m_ = a0 / (a0 + b0); rho = 1 / (a0 + b0 + 1)
        pred_var = 3 * m_ * (1 - m_) * (1 + 2 * rho)
        out["gof"].append({"bench": bench, "prior": [a0, b0], "cells": cells, "obs": obs.tolist(), "exp": exp.round(2).tolist(),
                           "pearson": pear, "p_boot": float((np.array(nullp) >= pear).mean()), "tarone_Z": Z,
                           "var_s_n3_obs": float(k[sel].var(ddof=1)), "var_s_n3_pred": pred_var})
        # (iii) replay under alternative priors
        obsr = S.observed(bench, len(dev))
        dtasks = sorted(dev)
        sdev = np.array([sum(dev[t]) for t in dtasks]); ndev = np.array([len(dev[t]) for t in dtasks])
        ddom = np.array([load.domain(bench, t) for t in dtasks])
        variants = {"pooled (paper)": (np.full(len(dtasks), a0), np.full(len(dtasks), b0))}
        if bench != "terminal-bench-2-1":
            dfit = {G_: fit(k[dom == G_], n[dom == G_])[:2] for G_ in np.unique(dom)}
            variants["per domain"] = (np.array([dfit[x][0] for x in ddom]), np.array([dfit[x][1] for x in ddom]))
        ad, bd, _ = fit(k[pan == "dev"], n[pan == "dev"])
        variants["dev-only prior"] = (np.full(len(dtasks), ad), np.full(len(dtasks), bd))
        variants["Jeffreys per task"] = (np.full(len(dtasks), .5), np.full(len(dtasks), .5))
        variants["uniform per task"] = (np.full(len(dtasks), 1.), np.full(len(dtasks), 1.))
        bi = load.BENCHES.index(bench)
        for vi, (vname, (pa, pb)) in enumerate(variants.items()):
            S.reseed(bi, 50 + vi)
            p = S.RNG.beta(pa + sdev, pb + ndev - sdev, size=(DRAWS, len(dtasks)))
            row = {"bench": bench, "variant": vname, "prior_ab_mean": [float(pa.mean()), float(pb.mean())]}
            for rule in ("SkillOpt", "GEPA"):
                K = obsr[rule]["K"]
                gr = S.best_of(p, K, S.outcomes(p)); go = S.best_of(p, K, obsr[rule]["baseline"])
                row[rule] = {"E_random": float(gr.mean()), "E_observed": float(go.mean()),
                             "P_ge_obs_random": float((gr >= obsr[rule]["gain"]).mean()), "P_ge_obs_observed": float((go >= obsr[rule]["gain"]).mean()),
                             "obs_gain": obsr[rule]["gain"]}
            sa = obsr["SkillAdam"]
            row["SkillAdam"] = S.skilladam_null(p[:20000], sa, np.random.default_rng([20261020, bi, vi]))
            row["SkillAdam"]["obs_points"] = sa["reported_gain_points"]
            crit, mde = S.mde80(p[:20000])
            row["MDE80_pts"] = 100 * mde / len(dtasks)
            row["P_accept_null"] = float((S.paired(p, p) > 0).mean())
            out["replay"].append(row)

    json.dump(out, open(os.path.join(load.OUT, "a2.json"), "w"), indent=1, default=float)
    for r in out["lrt"]:
        print(f"LRT {r['bench'][:6]} {r['grouping']:15s} G={r['groups']} LR={r['LR']:.2f} df={r['df']} p_chi2={r['p_chi2']:.4f} p_boot={r['p_boot']:.3f} {r['per_group_ab_mean_n']}")
    for r in out["gof"]:
        print(f"GOF {r['bench'][:6]} prior={np.round(r['prior'],3)} cells={r['cells']} obs={r['obs']} exp={r['exp']} X2={r['pearson']:.2f} p_boot={r['p_boot']:.3f} tarone={r['tarone_Z']:.1f} var_s|n3 obs={r['var_s_n3_obs']:.3f} pred={r['var_s_n3_pred']:.3f}")
    for r in out["replay"]:
        so, ge, sa = r["SkillOpt"], r["GEPA"], r["SkillAdam"]
        print(f"REPLAY {r['bench'][:6]} {r['variant']:18s} SO E rand/obs {so['E_random']:.2f}/{so['E_observed']:.2f} P {so['P_ge_obs_random']:.3f}/{so['P_ge_obs_observed']:.3f} | GEPA {ge['E_random']:.2f}/{ge['E_observed']:.2f} P {ge['P_ge_obs_random']:.3f}/{ge['P_ge_obs_observed']:.3f} | SA E {sa['E_reported_gain_points']:.1f} P {sa['P_reported_gain_ge_observed']:.2f} | MDE80 {r['MDE80_pts']:.1f} Pacc {r['P_accept_null']:.3f}")


if __name__ == "__main__":
    main()
