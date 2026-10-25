"""Power of the observed-baseline replay as a test: one of K candidates truly gains Delta tasks (raised on least-solved tasks), the rest
identical; reject when G >= c, c the 0.05 critical value under the null (M0, fitted pooled Beta). Absence-of-evidence check."""
import numpy as np
from aux.noise_and_domains import load
from src.analysis import selection as S


def main():
    d = load.load()
    obs = {'terminal-bench-2-1': {'SkillOpt': (5, 4, 1), 'GEPA': (8, 4, 1)},
           'tau3-bench': {'SkillOpt': (4, 18, 4), 'GEPA': (7, 18, 5)},
           'automationbench': {'SkillOpt': (4, 17, 2), 'GEPA': (7, 17, 1)}}
    rng = np.random.default_rng(2); D = 40000
    for bench, rules in obs.items():
        dev = d[bench]['seed_dev']; test = d[bench]['seed_test_exec']
        a, b = S.fit_prior([(sum(v), len(v)) for v in list(dev.values()) + list(test.values())])
        tasks = sorted(dev); s = np.array([sum(dev[t]) for t in tasks]); n = np.array([len(dev[t]) for t in tasks])
        p = rng.beta(a + s, b + n - s, size=(D, len(tasks)))
        for rule, (K, base, g) in rules.items():
            G0 = np.maximum((rng.random((D, K, len(tasks))) < p[:, None, :]).sum(2).max(1) - base, 0)
            c = next(t for t in range(0, len(tasks) + 1) if (G0 >= t).mean() <= 0.05)
            row = []
            for delta in (1, 2, 3, 5):
                q = S.raised(p.copy(), float(delta))
                cand = (rng.random((D, K, len(tasks))) < p[:, None, :]).sum(2)
                cand[:, 0] = (rng.random((D, len(tasks))) < q).sum(1)
                G = np.maximum(cand.max(1) - base, 0)
                row.append(f"d{delta}: pow {(G>=c).mean():.2f} P(G>=g_obs) {(G>=g).mean():.2f}")
            print(bench[:6], rule, f"K{K} g{g} c{c} null P(G0>=g) {(G0>=g).mean():.2f} |", ' | '.join(row))


if __name__ == "__main__":
    main()
