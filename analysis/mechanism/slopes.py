"""Within-task slopes of input tokens on turns (Sec. 3 elasticity claims, appendix slope tables): slope of log X on log R by sample and skill, the gap
decomposition slope = e_R + s_H b_h + s_F b_F with a counterfactual at task-mean content, the share of input in the last quarter of turns, turns vs success,
where optimized skills' extra turns come from (calls per turn), and the per-turn mechanism of the seed.
Output: assets/cache/analysis/slopes/slopes.txt. Run: <bach-harness venv python> -m analysis.mechanism.slopes   (from bipa/)"""
import collections
import numpy as np
from analysis.common import BN, Report, gmean
from analysis.mechanism.episodes import episode_rows, fe, samp
from analysis.mechanism.pricing import price

lx = lambda r: np.log(r["X"]); lR = lambda r: np.log(r["R"])


def main():
    rows = episode_rows(); say = Report()
    say("== Q1 slope of log X on log R (within-task FE)")
    for kind in ("main", "seed", "opt", "all"):
        for b in BN:
            s = samp(rows, b, kind); bb, se = fe(s, lx, lR)
            eR = np.array([r["eR"] for r in s]); sH = np.array([r["sH"] for r in s])
            say(f"{kind:5s} {BN[b]:5s} n={len(s):4d} slope={bb:.2f}({se:.2f}) meanER={eR.mean():.2f} range=[{eR.min():.2f},{eR.max():.2f}] 1+meanSH={1+sH.mean():.2f}")
    say("-- per skill (within task FE, all reps for that skill)")
    for b in BN:
        for sk in ("seed", "gepa", "skillopt", "skilladam"):
            s = [r for r in rows if r["bench"] == b and r["skill"] == sk]
            if len(s) >= 17:
                bb, se = fe(s, lx, lR); say(f"  {BN[b]} {sk} n={len(s)} slope={bb:.2f}({se:.2f}) meanER={np.mean([r['eR'] for r in s]):.2f}")
    say("-- task x skill FE (within a skill-task pair across reruns)")
    for b in BN:
        s2 = [dict(r, task=r["task"] + "|" + r["skill"]) for r in rows if r["bench"] == b]
        c = collections.Counter(r["task"] for r in s2); s2 = [r for r in s2 if c[r["task"]] > 1]
        bb, se = fe(s2, lx, lR); say(f"  {BN[b]} n={len(s2)} slope={bb:.2f}({se:.2f})")
    say("-- gap decomposition: slope = E[e_R] + s_H*b_h + s_F*b_F  (within task, per-episode weights approximated by means)")
    for b in BN:
        s = samp(rows, b, "main")
        sl, _ = fe(s, lx, lR)
        bh, _ = fe(s, lambda r: np.log(r["h"]), lR); bF, _ = fe(s, lambda r: np.log(r["F"]), lR)
        eR = np.mean([r["eR"] for r in s]); sH = np.mean([r["sH"] for r in s]); sF = 1 - sH
        say(f"{BN[b]}: slope={sl:.2f}  mean eR={eR:.2f}  b_h(dlogh/dlogR)={bh:.2f} b_F={bF:.2f}  predicted={eR+sH*bh+sF*bF:.2f}  gap(slope-eR)={sl-eR:.2f}  sH*b_h={sH*bh:.2f} sF*bF={sF*bF:.2f}")
        g = collections.defaultdict(list)
        for r in s: g[r["task"]].append(r)
        for r in s:
            gg = g[r["task"]]; r["h_bar"] = gmean([q["h"] for q in gg])
            r["F_bar"] = gmean([q["F"] for q in gg])
            r["Xcf"] = r["R"] * r["F_bar"] + r["R"] * (r["R"] - 1) / 2 * r["h_bar"]
        scf, _ = fe(s, lambda r: np.log(r["Xcf"]), lR)
        say(f"    counterfactual slope with h,F fixed at task means (pure structural, geometric means): {scf:.2f}")
    say("-- spread of R within task (sd log R) and e_R std")
    for b in BN:
        s = samp(rows, b, "main"); g = collections.defaultdict(list)
        for r in s: g[r["task"]].append(np.log(r["R"]))
        say(BN[b], "median sd logR within task", np.median([np.std(v) for v in g.values()]), "median R", np.median([r["R"] for r in s]), "R range", min(r["R"] for r in s), max(r["R"] for r in s))
    say("== A: share of input tokens in last k turns / last 25% turns; marginal X_t / X_1")
    for b in BN:
        s = samp(rows, b, "main"); l4 = []; l3 = []
        for r in s:
            x = r["req_in"]; R = r["R"]; l4.append(x[-max(1, R // 4):].sum() / x.sum()); l3.append(x[-3:].sum() / x.sum())
        say(BN[b], "last quarter of turns carries", round(np.mean(l4), 2), "of input; last 3 turns", round(np.mean(l3), 2), "; median R", np.median([r['R'] for r in s]))
    say("== B: turns vs success within task (logit-free: success rate by R relative to task median)")
    for b in BN:
        s = samp(rows, b, "all"); g = collections.defaultdict(list)
        for r in s: g[r["task"]].append(r)
        d = []
        for t, v in g.items():
            if len(v) < 3: continue
            m = np.median([q["R"] for q in v])
            for q in v:
                if q["succ"] is not None: d.append((q["R"] / m, int(q["succ"])))
        d = np.array(d); lo = d[d[:, 0] < 1]; hi = d[d[:, 0] > 1]
        say(BN[b], "success rate: shorter-than-task-median", round(lo[:, 1].mean(), 2), f"(n={len(lo)})", "longer", round(hi[:, 1].mean(), 2), f"(n={len(hi)})")
        sd = [r for r in s if r["succ"] is not None]
        say("   mean R success", round(np.mean([r['R'] for r in sd if r['succ']]), 1), "fail", round(np.mean([r['R'] for r in sd if not r['succ']]), 1), "; mean X success", round(np.mean([r['X'] for r in sd if r['succ']]) / 1e3), "fail", round(np.mean([r['X'] for r in sd if not r['succ']]) / 1e3), "k")
    say("== C: where extra turns come from: optimized vs seed paired by task (main), dlogR = dlogN - dlog m ; zero-call turn share")
    for b in BN:
        sd = {r["task"]: r for r in rows if r["bench"] == b and r["skill"] == "seed" and r["rep"] == "main"}
        for sk in ("gepa", "skillopt", "skilladam"):
            so = [r for r in rows if r["bench"] == b and r["skill"] == sk and r["rep"] == "main"]
            if not so: continue
            p = [(r, sd[r["task"]]) for r in so if r["task"] in sd]
            dR = np.mean([np.log(a["R"] / c["R"]) for a, c in p]); dN = np.mean([np.log(a["N"] / c["N"]) for a, c in p])
            z = lambda r: np.mean(r["ncall"] == 0)
            dX = np.mean([np.log(a["X"] / c["X"]) for a, c in p])
            dP = np.mean([np.log(price(a, .1, 4, "res_all") / price(c, .1, 4, "res_all")) for a, c in p])
            say(BN[b], sk, f"dlogX={dX:+.3f} dlogR={dR:+.3f} dlogN={dN:+.3f} dlog m={dN-dR:+.3f} dlogPrice(.1,4)={dP:+.3f}  zero-call turn share seed {np.mean([z(c) for a, c in p]):.2f} -> {np.mean([z(a) for a, c in p]):.2f}")
    say("== D: per-turn mechanism: calls/turn m, share of zero-call turns, mean new content per turn h (seed main)")
    for b in BN:
        s = [r for r in samp(rows, b, "main") if r["skill"] == "seed"]
        say(BN[b], "m", round(np.mean([r['N'] / r['R'] for r in s]), 2), "zero-call share", round(np.mean([np.mean(r['ncall'] == 0) for r in s]), 2), "h tokens", round(np.median([r['h'] for r in s])), "F tokens", round(np.median([r['F'] for r in s])), "sH", round(np.mean([r['sH'] for r in s]), 2))
    say.save("slopes", "slopes.txt")


if __name__ == "__main__":
    main()
