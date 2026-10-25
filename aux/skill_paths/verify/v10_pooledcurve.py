import json
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
    for mdl in ("deepseek-flash", "gpt-6-luna"):
        ks = [k for k in runs if k[0] == mdl and k[1] == "gepa" and k[3] != "fromscratch"]
        curves = []
        for k in ks:
            d = runs[k]; seq = [s["dev"]["n_success"] for s in scored(d)]; b = d["seed_dev"]["n_success"]
            curves.append([100 * max(0, max(seq[:i + 1]) - b) / d["dev_panel_size"] for i in range(len(seq))])
        K = max(len(c) for c in curves)
        out = []
        for i in range(K):
            v = [c[i] for c in curves if len(c) > i]
            out.append((i + 1, len(v), round(float(np.mean(v)), 1)))
        print(mdl, "GEPA pooled best-so-far mean (k, n runs, mean pts):", [o for o in out if o[1] >= 2])
        print("   final per-run gains:", [round(c[-1], 1) for c in curves], "mean of the three final gains %.1f" % np.mean([c[-1] for c in curves]))
