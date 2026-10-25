"""Do the learned skills' convergent gains look like shortcuts (fewer turns / calls than the seed) or like
the full procedure? For every task where a learned skill's 3-run mean success exceeds the seed's, split by converged
(divergence below the seed's, as in explore.py) vs not; report turns R and calls N (3-run means) as ratios to the seed,
and the success pattern (seed wins out of 3 -> skill wins out of 3). Run:  python -m aux.transfer_and_trajectories.convergent_gains
"""
import collections
import json
import math
import numpy as np

from src.analysis import dynamics as D
from src import layout

OUT = layout.OUT / "transfer_and_trajectories/convergent_gains.json"
CELLS = [(m, b) for m in ("deepseek-flash", "gpt-6-luna") for b in ("tau3-bench", "automationbench")]


def per_task(reps):
    ks = sorted(reps)
    tasks = [t for t in set.intersection(*(set(reps[k]) for k in ks))
             if all(reps[k][t]["R"] > 0 and reps[k][t]["success"] is not None for k in ks)]
    out = {}
    for t in tasks:
        seqs = [reps[k][t]["seq"] for k in ks]
        sims = [D.editsim(seqs[i], seqs[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        out[t] = {"D": 1 - float(np.mean(sims)), "wins": sum(bool(reps[k][t]["success"]) for k in ks),
                  "R": float(np.mean([reps[k][t]["R"] for k in ks])), "N": float(np.mean([reps[k][t]["N"] for k in ks])),
                  "X": float(np.mean([reps[k][t]["X"] for k in ks]))}
    return out


def main():
    groups = collections.defaultdict(list)
    listing = []
    for model, bench in CELLS:
        seed = per_task(D.load_cell(model, "native-seed", bench))
        for arm in ("gepa", "skillopt", "skilladam"):
            own = per_task(D.load_cell(model, arm, bench))
            for t in own:
                if t not in seed or own[t]["wins"] <= seed[t]["wins"]:
                    continue
                key = "converged" if own[t]["D"] < seed[t]["D"] else "not converged"
                row = {"model": model, "bench": bench, "arm": arm, "task": t, "group": key,
                       "pattern": f"{seed[t]['wins']}->{own[t]['wins']}",
                       **{f"r{k}": own[t][k] / seed[t][k] for k in ("R", "N", "X") if seed[t][k] > 0}}
                groups[key].append(row)
                listing.append(row)
    for key, rows in groups.items():
        gm = {k: math.exp(np.mean([math.log(r[f"r{k}"]) for r in rows if r.get(f"r{k}")])) for k in ("R", "N", "X")}
        fewer = np.mean([r["rR"] < 1 for r in rows if "rR" in r])
        pats = collections.Counter(r["pattern"] for r in rows)
        print(f"{key:14s} n={len(rows)} turns x{gm['R']:.2f} calls x{gm['N']:.2f} tokens x{gm['X']:.2f} "
              f"fewer turns than seed {100*fewer:.0f}% patterns {dict(pats.most_common())}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(listing, indent=1))


if __name__ == "__main__":
    main()
