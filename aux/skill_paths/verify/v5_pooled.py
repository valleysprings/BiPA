"""Independent pooled null: mean final gain (points) over runs vs replay distribution of the same mean (independent replays per run), fitted prior."""
import json, importlib.util, sys
from pathlib import Path
import numpy as np
from src.layout import OUT


if __name__ == "__main__":
    SC = OUT / "skill_paths"
    runs = {}
    for p in sorted((SC / "paths").glob("*.json")):
        d = json.load(open(p)); runs[(d["model"], d["optimizer"], d["benchmark"], d["run_id"])] = d
    SI = json.load(open(SC / "inputs/seed_inputs.json"))
    rng = np.random.default_rng(99); D = 100_000
    def cands(d): return [s for s in d["steps"] if s["event"] == "candidate"]
    def scored(d): return [s for s in cands(d) if s.get("dev")]
    def norm(t): return [l.strip() for l in t.splitlines() if l.strip()]
    def pdraw(si, prior):
        ks = sorted(si["dev"]); pa = np.array([prior[0] + sum(si["dev"][t]) for t in ks]); pb = np.array([prior[1] + len(si["dev"][t]) - sum(si["dev"][t]) for t in ks])
        return rng.beta(pa, pb, size=(D, len(ks)))
    def prior_of(si, name): return tuple(si["prior_fitted"]) if name == "fitted" else (4 * si["mean_success"], 4 * (1 - si["mean_success"]))
    def gain_null(k, name):
        d = runs[k]; si = SI[k[0]][k[2]]; p = pdraw(si, prior_of(si, name)); n = p.shape[1]
        if k[1] in ("gepa", "skillopt"):
            K = len(scored(d)) if k[1] == "gepa" else len(cands(d)); base = d["seed_dev"]["n_success"]
            sc = np.stack([(rng.random(p.shape) < p).sum(axis=1) for _ in range(K)], axis=1)
            return 100 * np.maximum(sc.max(axis=1) - base, 0) / n
        steps = cands(d); batch = steps[0]["gate"]["detail"]["candidate"]["n"]; rows = np.arange(D)[:, None]
        live = []
        for s in steps:
            par = next((x for x in d["steps"] if x["candidate_id"] == s["parent_id"]), None)
            live.append(bool(par is not None and norm(par["skill_text"]) != norm(s["skill_text"])))
        cur = np.full((D, n), -1, dtype=int); total = np.zeros(D)
        for lv in live:
            pick = np.argsort(rng.random((D, n)), axis=1)[:, :batch]; pp = p[rows, pick]
            seen = cur[rows, pick]; fresh = (rng.random(pp.shape) < pp).astype(int); seen = np.where(seen < 0, fresh, seen); cur[rows, pick] = seen
            if not lv: continue
            base = seen.sum(axis=1); cand = (rng.random(pp.shape) < pp).astype(int); acc = cand.sum(axis=1) > base
            total += np.where(acc, 100 * (cand.sum(axis=1) - base) / batch, 0); cur[acc] = -1; cur[rows[acc], pick[acc]] = cand[acc]
        return total
    def obs_gain(k):
        d = runs[k]
        if k[1] == "skilladam":
            return sum(100 * (s["gate"]["detail"]["candidate"]["n_success"] - s["gate"]["detail"]["baseline"]["n_success"]) / s["gate"]["detail"]["candidate"]["n"] for s in cands(d) if s["kept"])
        seq = [s["dev"]["n_success"] for s in (scored(d) if k[1] == "gepa" else cands(d))]
        return 100 * max(0, max(seq) - d["seed_dev"]["n_success"]) / d["dev_panel_size"]
    HW = [k for k in runs if k[3] != "fromscratch"]
    REC = {("skillopt", "DeepSeek"): (10.4, 11.5, 0.693), ("skillopt", "Luna"): (4.4, 7.4, 0.813), ("skillopt", "both"): (7.4, 9.4, 0.820),
           ("gepa", "DeepSeek"): (10.4, 13.0, 0.858), ("gepa", "Luna"): (6.7, 8.6, 0.720), ("gepa", "both"): (8.5, 10.8, 0.849),
           ("skilladam", "DeepSeek"): (27.4, 25.6, 0.383), ("skilladam", "Luna"): (33.3, 26.6, 0.278), ("skilladam", "both"): (30.4, 26.1, 0.289)}
    for opt in ("skillopt", "gepa", "skilladam"):
        ks = sorted(k for k in HW if k[1] == opt)
        G = {k: gain_null(k, "fitted") for k in ks}
        for grp in ("DeepSeek", "Luna", "both"):
            sel = [k for k in ks if grp == "both" or (k[0] == "deepseek-flash") == (grp == "DeepSeek")]
            mean_null = sum(G[k] for k in sel) / len(sel); o = sum(obs_gain(k) for k in sel) / len(sel)
            P = float((mean_null >= o - 1e-9).mean())
            r = REC[(opt, grp)]
            print(f"{opt:9s} {grp:8s} obs mean {o:5.1f} (rec {r[0]}) | null mean {mean_null.mean():5.1f} [{np.quantile(mean_null,.05):.1f}, {np.quantile(mean_null,.95):.1f}] (rec {r[1]}) | P {P:.3f} (rec {r[2]})")
