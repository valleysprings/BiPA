"""Independent check of the observed-baseline replay (SkillOpt/GEPA) under pooled Beta (M0) and plug-in (M1),
plus the power of the replay test: P(G >= g_obs) and p-value distribution when candidates truly gain."""
import numpy as np
from aux.noise_and_domains import load
from src.analysis import selection as S
D = 100000


def main():
    d = load.load()
    obs = {'terminal-bench-2-1': {'SkillOpt': (5, 4, 1), 'GEPA': (8, 4, 1)},
           'tau3-bench': {'SkillOpt': (4, 18, 4), 'GEPA': (7, 18, 5)},
           'automationbench': {'SkillOpt': (4, 17, 2), 'GEPA': (7, 17, 1)}}
    rng = np.random.default_rng(1)
    for bench, rules in obs.items():
        dev = d[bench]['seed_dev']
        test = d[bench]['seed_test_exec']
        pooled = [(sum(v), len(v)) for v in list(dev.values()) + list(test.values())]
        a, b = S.fit_prior(pooled)
        tasks = sorted(dev)
        s = np.array([sum(dev[t]) for t in tasks]); n = np.array([len(dev[t]) for t in tasks])
        for model in ('M0', 'M1'):
            if model == 'M0':
                p = rng.beta(a + s, b + n - s, size=(D, len(tasks)))
            else:
                p = np.broadcast_to(s / n, (D, len(tasks)))
            out = []
            for rule, (K, base, g) in rules.items():
                best = (rng.random((D, K, len(tasks))) < p[:, None, :]).sum(2).max(1)
                G = np.maximum(best - base, 0)
                # power-type check: candidates truly raised by +2 tasks (least-solved), same baseline
                q = S.raised(np.array(p, dtype=float, copy=True)[:20000], 2.0) if model == 'M0' else None
                out.append(f"{rule} K{K} E{G.mean():.2f} P{(G>=g).mean():.3f}")
            print(bench[:6], model, (round(a,3), round(b,3)), ' | '.join(out), 'mean s/n*N', round((s/n).sum(),2))


if __name__ == "__main__":
    main()
