"""Robustness check: tau3 replay (M0 pooled Beta) with a run-level shock on candidate dev totals.
shock SD in tasks; 'shared' = one shock per optimizer run (all K candidates), 'indep' = one per candidate."""
import numpy as np
from aux.noise_and_domains import load


def main():
    d = load.load(); dev = d['tau3-bench']['seed_dev']
    a, b = 0.548, 0.193
    tasks = sorted(dev); s = np.array([sum(dev[t]) for t in tasks]); n = np.array([len(dev[t]) for t in tasks])
    rng = np.random.default_rng(1); N = 100000
    p = rng.beta(a + s, b + n - s, size=(N, len(tasks)))
    for K, g in ((4, 4), (7, 5)):
        tot = np.stack([(rng.random((N, len(tasks))) < p).sum(1) for _ in range(K)], 1).astype(float)
        for sd in (0, 1.0, 1.8, 2.5):
            for mode in ('shared', 'indep'):
                sh = rng.normal(0, sd, (N, 1)) if mode == 'shared' else rng.normal(0, sd, (N, K))
                best = np.round(tot + sh).max(1); G = np.maximum(best - 18, 0)
                print(K, g, sd, mode, round(G.mean(), 2), round((G >= g).mean(), 3))


if __name__ == "__main__":
    main()
