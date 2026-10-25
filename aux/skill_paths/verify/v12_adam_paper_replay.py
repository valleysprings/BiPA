"""Independent replay of SkillAdam's *paper-replay* null (budget = number of candidate patches per iteration), fitted prior, own code."""
import json
from pathlib import Path
import numpy as np
from src.layout import OUT


if __name__ == "__main__":
    SC = OUT / "skill_paths"
    runs = {}
    for p in sorted((SC / "paths").glob("*skilladam*.json")):
        d = json.load(open(p)); runs[(d["model"], d["optimizer"], d["benchmark"], d["run_id"])] = d
    SI = json.load(open(SC / "inputs/seed_inputs.json")); rng = np.random.default_rng(31); D = 100_000
    REC = {("deepseek-flash", "tau3-bench"): (72.0, 0.871), ("deepseek-flash", "automationbench"): (43.2, 0.884), ("deepseek-flash", "terminal-bench-2-1"): (11.4, 0.248),
           ("gpt-6-luna", "tau3-bench"): (54.4, 0.598), ("gpt-6-luna", "automationbench"): (67.9, 0.752), ("gpt-6-luna", "terminal-bench-2-1"): (18.0, 0.886)}
    for k in sorted(runs):
        d = runs[k]; si = SI[k[0]][k[2]]; steps = [s for s in d["steps"] if s["event"] == "candidate"]
        budgets = [s["gate"]["detail"]["edit_budget"] for s in steps]; batch = steps[0]["gate"]["detail"]["candidate"]["n"]
        obs = sum(100 * (s["gate"]["detail"]["candidate"]["n_success"] - s["gate"]["detail"]["baseline"]["n_success"]) / batch for s in steps if s["kept"])
        ks = sorted(si["dev"]); a, b = si["prior_fitted"]
        pa = np.array([a + sum(si["dev"][t]) for t in ks]); pb = np.array([b + len(si["dev"][t]) - sum(si["dev"][t]) for t in ks])
        p = rng.beta(pa, pb, size=(D, len(ks))); n = len(ks); rows = np.arange(D)[:, None]
        cur = np.full((D, n), -1, dtype=int); total = np.zeros(D)
        for bud in budgets:
            pick = np.argsort(rng.random((D, n)), axis=1)[:, :batch]; pp = p[rows, pick]
            seen = cur[rows, pick]; fresh = (rng.random(pp.shape) < pp).astype(int); seen = np.where(seen < 0, fresh, seen); cur[rows, pick] = seen
            base = seen.sum(axis=1)
            patches = (rng.random((D, bud, batch)) < pp[:, None, :]).astype(int)
            better = patches.sum(axis=2) > base[:, None]; acc = better.any(axis=1); first = better.argmax(axis=1)
            winner = patches[np.arange(D), first]
            total += np.where(acc, 100 * (winner.sum(axis=1) - base) / batch, 0)
            cur[acc] = -1; cur[rows[acc], pick[acc]] = winner[acc]
        P = float((total >= obs - 1e-9).mean()) if obs > 0 else float((total > 0).mean())
        r = REC[(k[0], k[2])]
        print(f"{k[0][:6]} {k[2][:6]:7s} obs {obs:5.1f} budgets {budgets} | E {total.mean():5.1f} (rec {r[0]}) P {P:.3f} (rec {r[1]})")
