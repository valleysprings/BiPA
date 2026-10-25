import json, math
from pathlib import Path
import numpy as np
from src.layout import OUT


if __name__ == "__main__":
    SC = OUT / "skill_paths"
    runs = {}
    for p in sorted((SC / "paths").glob("*.json")):
        d = json.load(open(p)); runs[(d["model"], d["optimizer"], d["benchmark"], d["run_id"])] = d
    def cands(d): return [s for s in d["steps"] if s["event"] == "candidate"]
    def scored(d): return [s for s in cands(d) if s.get("dev")]
    HW = [k for k in runs if k[3] != "fromscratch"]
    def avg_rank(a):
        a = np.asarray(a, float); order = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]: j += 1
            r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
        return r
    def corr(x, y):
        x = np.asarray(x); y = np.asarray(y)
        if x.std() == 0 or y.std() == 0: return float("nan")
        return float(np.corrcoef(x, y)[0, 1])
    def stat(groups):
        xs, ys = [], []
        for x, y in groups:
            if len(x) >= 2:
                rx, ry = avg_rank(x), avg_rank(y); xs.append(rx - rx.mean()); ys.append(ry - ry.mean())
        return corr(np.concatenate(xs), np.concatenate(ys))
    rng = np.random.default_rng(5)
    def rows_for(k):
        d = runs[k]; out = []
        for s in scored(d):
            if k[1] == "skilladam":
                det = s["gate"]["detail"]
                rho = det["candidate"]["dev"]["mean_input_tokens_per_episode"] / det["baseline"]["dev"]["mean_input_tokens_per_episode"]
                y = det["candidate"]["n_success"] - det["baseline"]["n_success"]
            else:
                rho = s["dev"]["mean_input_tokens_per_episode"] / d["seed_dev"]["mean_input_tokens_per_episode"]
                y = s["dev"]["n_success"] - d["seed_dev"]["n_success"]
            out.append((math.log(rho), y, bool(s["kept"])))
        return out
    REC = {"gepa": (-0.08, 0.63, 50, 2.9), "skillopt": (0.06, 0.83, 26, 0.1), "skilladam": (0.07, 0.51, 96, -0.0)}
    for opt in ("gepa", "skillopt", "skilladam"):
        groups = []; slopes = []; n = 0; kept_l = []; rej_l = []
        for k in sorted(k for k in HW if k[1] == opt):
            r = rows_for(k)
            x = np.array([a for a, _, _ in r]); y = np.array([b for _, b, _ in r]); groups.append((x, y)); n += len(x)
            if len(x) >= 2:
                kk = np.arange(1, len(x) + 1); sl = np.polyfit(kk, x, 1)[0]; slopes.append(sl)
            for a, _, kp in r: (kept_l if kp else rej_l).append(a)
        obs = stat(groups); cnt = 0; NP = 5000
        for _ in range(NP):
            if abs(stat([(x, rng.permutation(y)) for x, y in groups])) >= abs(obs) - 1e-12: cnt += 1
        p = (cnt + 1) / (NP + 1)
        rec = REC[opt]
        print(f"{opt:9s} within-run rank corr {obs:+.2f} p {p:.2f} n {n} (rec {rec[0]:+.2f}, p {rec[1]}, n {rec[2]}) | mean slope of log rho: {100*np.mean(slopes):+.1f}% (exp-1: {100*np.mean(np.expm1(slopes)):+.1f}%) rising {sum(s>0 for s in slopes)}/{len(slopes)} (rec {rec[3]:+.1f}) | n kept/rej {len(kept_l)}/{len(rej_l)} gm kept {math.exp(np.mean(kept_l)):.2f} rej {math.exp(np.mean(rej_l)):.2f}")
