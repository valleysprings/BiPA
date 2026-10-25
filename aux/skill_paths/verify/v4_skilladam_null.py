"""Independent own SkillAdam 'one live candidate' null replay (fitted prior), to check E and P of mech_paths.md, SkillAdam table."""
import json
from pathlib import Path
import numpy as np
from src.layout import OUT


if __name__ == "__main__":
    SC = OUT / "skill_paths"
    runs = {}
    for p in sorted((SC / "paths").glob("*skilladam*.json")):
        d = json.load(open(p)); runs[(d["model"], d["optimizer"], d["benchmark"], d["run_id"])] = d
    SI = json.load(open(SC / "inputs/seed_inputs.json"))
    def norm(t): return [l.strip() for l in t.splitlines() if l.strip()]
    rng = np.random.default_rng(4242)
    DRAWS = 100_000
    REC = {  # record: E one live, P fitted one live, P moderate one live
        ("deepseek-flash", "tau3-bench"): (34.1, 0.438, 0.613), ("deepseek-flash", "automationbench"): (33.2, 0.795, 0.950),
        ("deepseek-flash", "terminal-bench-2-1"): (9.6, 0.182, 0.637), ("gpt-6-luna", "tau3-bench"): (30.3, 0.214, 0.535),
        ("gpt-6-luna", "automationbench"): (34.8, 0.291, 0.629), ("gpt-6-luna", "terminal-bench-2-1"): (14.5, 0.787, 0.857)}

    def null_run(p_dev, batch, live_flags, fill_cache_on_nonlive):
        draws, n = p_dev.shape
        rows = np.arange(draws)[:, None]
        cur = np.full((draws, n), -1, dtype=int)
        total = np.zeros(draws)
        for live in live_flags:
            pick = np.argsort(rng.random((draws, n)), axis=1)[:, :batch]
            p = p_dev[rows, pick]
            seen = cur[rows, pick]
            fresh = (rng.random(p.shape) < p).astype(int)
            seen = np.where(seen < 0, fresh, seen)
            if live or fill_cache_on_nonlive:
                cur[rows, pick] = seen
            if not live:
                continue
            base = seen.sum(axis=1)
            cand = (rng.random(p.shape) < p).astype(int)
            acc = cand.sum(axis=1) > base
            total += np.where(acc, 100 * (cand.sum(axis=1) - base) / batch, 0)
            cur[acc] = -1
            cur[rows[acc], pick[acc]] = cand[acc]
        return total

    for k in sorted(runs):
        d = runs[k]; si = SI[k[0]][k[2]]
        steps = [s for s in d["steps"] if s["event"] == "candidate"]
        live = []
        for s in steps:
            par = next((x for x in d["steps"] if x["candidate_id"] == s["parent_id"]), None)
            live.append(bool(par is not None and norm(par["skill_text"]) != norm(s["skill_text"])))
        # observed gain in points: sum of accepted gate differences / draw size
        obs = 0.0
        for s in steps:
            if s["kept"]:
                det = s["gate"]["detail"]; obs += 100 * (det["candidate"]["n_success"] - det["baseline"]["n_success"]) / det["candidate"]["n"]
        batch = steps[0]["gate"]["detail"]["candidate"]["n"]
        ks = sorted(si["dev"])
        out = []
        for nm, prior in (("fitted", tuple(si["prior_fitted"])), ("moderate", (4 * si["mean_success"], 4 * (1 - si["mean_success"])))):
            pa = np.array([prior[0] + sum(si["dev"][t]) for t in ks]); pb = np.array([prior[1] + len(si["dev"][t]) - sum(si["dev"][t]) for t in ks])
            p = rng.beta(pa, pb, size=(DRAWS, len(ks)))
            row = []
            for fill in (True, False):
                g = null_run(p, batch, live, fill)
                P = float((g >= obs - 1e-9).mean()) if obs > 0 else float((g > 0).mean())
                row.append((round(float(g.mean()), 1), round(P, 3)))
            out.append((nm, row))
        r = REC[(k[0], k[2])]
        print(f"{k[0][:6]} {k[2][:6]:7s} obs {obs:5.1f} live {sum(live)}/{len(live)} | fitted: cache-filled-on-nonlive E,P {out[0][1][0]}, skip-nonlive {out[0][1][1]} | moderate {out[1][1][0]} | record E {r[0]} Pf {r[1]} Pm {r[2]}")
