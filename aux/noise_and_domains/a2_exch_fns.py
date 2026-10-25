import math
import numpy as np
from scipy import optimize, special
def bb_ll(a, b, k, n):
    return float((special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)
                  + special.betaln(k + a, n - k + b) - special.betaln(a, b)).sum())

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

