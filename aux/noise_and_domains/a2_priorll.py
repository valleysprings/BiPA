"""Marginal log-likelihood of the seed's executions under the replay's prior variants (how strongly the data reject the no-pooling priors).
Run: python -m aux.noise_and_domains.a2_priorll"""
import os, json
import numpy as np
from aux.noise_and_domains import load
from aux.noise_and_domains.a2_exch_fns import bb_ll, fit


def main():
    d = load.load()
    res = {}
    for bench in load.BENCHES:
        dev, test = d[bench]["seed_dev"], d[bench]["seed_test_exec"]
        k = np.array([sum(v) for v in list(dev.values()) + list(test.values())]); n = np.array([len(v) for v in list(dev.values()) + list(test.values())])
        dom = np.array([load.domain(bench, t) for t in list(dev) + list(test)])
        a, b, l = fit(k, n)
        row = {"pooled": l, "Jeffreys": bb_ll(.5, .5, k, n), "uniform": bb_ll(1., 1., k, n), "moderate": bb_ll(4 * k.sum() / n.sum(), 4 * (1 - k.sum() / n.sum()), k, n)}
        if bench != "terminal-bench-2-1":
            row["per domain"] = sum(fit(k[dom == G], n[dom == G])[2] for G in np.unique(dom))
        res[bench] = {kk: round(v - l, 2) for kk, v in row.items()}
        print(bench, "loglik relative to pooled fit:", res[bench])
    json.dump(res, open(os.path.join(load.OUT, "a2_priorll.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
