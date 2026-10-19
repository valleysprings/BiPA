"""Price slope vs cache price (Sec. 3 claim: the within-task price elasticity in the number of turns reaches 1 when cached input costs more than a cross-over
fraction c* of uncached input; cached tokens cost c, output tokens w per uncached input token). Also the price shares, the uncached/output/cached slopes and the
marginal price of the t-th turn. Output: assets/cache/analysis/pricing/pricing.txt.
Run: <bach-harness venv python> -m src.analysis.aux.pricing   (from bipa/)"""
import numpy as np
from src.utils.dataloader import BN
from src.utils.logger import Report

lR = lambda r: np.log(r["R"])


def price(r, c, w, cached="res"):
    """uncached input + c * cached input + w * output; cached = res (cache hits from request 2 on) or res_all (incl. the request-1 prefix, as facts-v9)."""
    return (r["X"] - r[cached]) + c * r[cached] + w * r["out_tot"]


def main():
    from src.analysis.aux.slopes import ep_rows, fe, samp  # local: slopes.py imports price() from this module
    rows = ep_rows(); say = Report()
    say("== Q2 price slope vs cache price c (output weight w=4 | w=1)")
    for kind in ("main", "all"):
        for w in (4, 1):
            for b in BN:
                s = samp(rows, b, kind); out = []
                for c in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
                    out.append(f"c={c}:{fe(s, lambda r: np.log(price(r, c, w)), lR)[0]:.2f}")
                cs = np.linspace(0, 1, 201); sl = [fe(s, lambda r: np.log(price(r, c, w)), lR)[0] for c in cs]
                cross = next((cs[i] for i in range(len(cs)) if sl[i] >= 1), None)
                say(kind, "w=", w, BN[b], "  ".join(out), "  cross(slope>=1) at c=", cross)
    say("== shares of price at c=0.1,w=4 (miss/cached/output) and cache share of input")
    for b in BN:
        s = samp(rows, b, "main"); X = sum(r["X"] for r in s); res = sum(r["res"] for r in s); o = sum(r["out_tot"] for r in s)
        miss = X - res; P = miss + .1 * res + 4 * o
        say(BN[b], f"miss {miss/P:.2f} cached {0.1*res/P:.2f} out {4*o/P:.2f}; cache share input {res/X:.3f}")
    say("== uncached-only slopes: miss tokens, output tokens, res (cached) tokens")
    for b in BN:
        s = samp(rows, b, "main")
        say(BN[b], "miss", round(fe(s, lambda r: np.log(r["X"] - r["res"]), lR)[0], 2), "out", round(fe(s, lambda r: np.log(r["out_tot"]), lR)[0], 2), "cached", round(fe(s, lambda r: np.log(r["res"]), lR)[0], 2))
    say("== marginal price of the t-th turn (mean over episodes with R>=t), c=0.1,w=4, relative to turn 2; and input tokens")
    for b in BN:
        s = samp(rows, b, "main")
        def pt(r, t, c):
            x = r["req_in"][t]; res = r["cache"][t] if t > 0 else 0
            return (x - res) + c * res + 4 * r["out"][t], x
        res = []
        for t in (1, 3, 7, 11, 15):
            v = [pt(r, t, .1) for r in s if r["R"] > t + 1]
            if len(v) > 10: res.append(f"t={t+1}: price {np.mean([a for a, _ in v]):.0f} (n={len(v)}) in {np.mean([x for _, x in v]):.0f}")
        say(BN[b], " | ".join(res))
    say("== within-task slope of price at c=0.1: structural 'e' for price: share of price in quadratic cached term")
    say("res_all (cache incl. request 1 prefix, as facts-v9)")
    for b in BN:
        s = samp(rows, b, "main"); o = []
        for c in (0.0, 0.1, 0.25, 0.5, 1.0): o.append(f"c={c}:{fe(s, lambda r: np.log(price(r, c, 4, 'res_all')), lR)[0]:.2f}")
        cs = np.linspace(0, 1, 201); sl = [fe(s, lambda r: np.log(price(r, c, 4, 'res_all')), lR)[0] for c in cs]
        say(BN[b], " ".join(o), " cross:", next((cs[i] for i in range(201) if sl[i] >= 1), None))
        say("   cache share of req-1 input:", sum(r["cache"][0] for r in s) / sum(r["req_in"][0] for r in s))
    say.save("pricing", "pricing.txt")


if __name__ == "__main__":
    main()
