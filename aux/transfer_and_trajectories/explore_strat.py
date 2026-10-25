"""Do a task's three runs agree more under a learned skill, stratified by how its success moved.

Runs: Table 1's avg@3 test runs (dynamics.load_cell). D = 1 - mean pairwise edit similarity of the executed tool-name sequences;
converged = D(own) < D(ref); tasks with 3 verified runs, R > 0, on both sides. Each task-contrast gets k_ref, k_own = successes out of 3.
Question: is the excess convergence of learned-skill gains over the seed (65% vs 48% of unchanged) a ceiling effect of
2->3 gains? (1) Converged share per transition k_ref->k_own, (2) gains against unchanged tasks at the same
starting level (k_own = k_ref) and at the same end level, with Mantel-Haenszel odds ratios (Robins-Breslow-Greenland CI),
for learned-over-seed (3 optimizers, 4 cells pooled) and seed-over-no-skill.
Run:  python -m aux.transfer_and_trajectories.explore_strat
Output: OUT/transfer_and_trajectories/explore_strat.json; summary on stdout.
"""
import collections
import json
import math
import numpy as np

from src import layout
from aux.headline_numbers import dynamics as D

THETA = 0.5
OUT = layout.OUT / "transfer_and_trajectories/explore_strat.json"
MODELS = ("deepseek-flash", "gpt-6-luna")
BENCHES = ("tau3-bench", "automationbench")
ARMS = ("native-noseed", "native-seed", "gepa", "skillopt", "skilladam")


def task_rows(reps):
    if len(reps) != 3:
        return {}
    ks = sorted(reps)
    tasks = [t for t in set.intersection(*(set(reps[k]) for k in ks))
             if all(reps[k][t]["R"] > 0 and reps[k][t]["success"] is not None for k in ks)]
    out = {}
    for t in tasks:
        seqs = [reps[k][t]["seq"] for k in ks]
        sims = [D.editsim(seqs[i], seqs[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        out[t] = {"D": 1 - float(np.mean(sims)), "k": sum(reps[k][t]["success"] is True for k in ks)}
    return out


def pairs(own, ref):
    return [(ref[t]["k"], own[t]["k"], own[t]["D"] < ref[t]["D"]) for t in own if t in ref]


def wilson(x, n, z=1.96):
    if n == 0:
        return [None, None]
    p = x / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [100 * (c - h), 100 * (c + h)]


def mh(strata):
    """strata: list of (a, b, c, d) = (gain&conv, gain&not, unch&conv, unch&not). MH OR with RGB 95% CI."""
    R = S = PR = PS_QR = QS = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n == 0 or (a + b) == 0 or (c + d) == 0:
            continue
        r, s = a * d / n, b * c / n
        p, q = (a + d) / n, (b + c) / n
        R += r; S += s; PR += p * r; PS_QR += p * s + q * r; QS += q * s
    if R == 0 or S == 0:
        return None
    orr = R / S
    var = PR / (2 * R * R) + PS_QR / (2 * R * S) + QS / (2 * S * S)
    se = math.sqrt(var)
    return {"OR": orr, "ci": [orr * math.exp(-1.96 * se), orr * math.exp(1.96 * se)]}


def crude_or(strata):
    a, b, c, d = (sum(s[i] for s in strata) for i in range(4))
    return a * d / (b * c) if b * c else None


def analyse(P, label):
    trans = collections.defaultdict(lambda: [0, 0])
    for kr, ko, cv in P:
        trans[(kr, ko)][0] += cv
        trans[(kr, ko)][1] += 1
    gains = [(kr, ko, cv) for kr, ko, cv in P if ko > kr]
    unch = [(kr, ko, cv) for kr, ko, cv in P if ko == kr]
    out = {"label": label, "n": len(P),
           "transitions": {f"{kr}->{ko}": {"conv": v[0], "n": v[1], "share": 100 * v[0] / v[1]}
                           for (kr, ko), v in sorted(trans.items())},
           "gain_conv": [sum(c for *_, c in gains), len(gains)], "unch_conv": [sum(c for *_, c in unch), len(unch)]}
    for key, idx in (("by_start", 0), ("by_end", 1)):
        strata, rows = [], {}
        for lvl in range(4):
            g = [c for kr, ko, c in gains if (kr, ko)[idx] == lvl]
            u = [c for kr, ko, c in unch if (kr, ko)[idx] == lvl]
            a, b, c, d = sum(g), len(g) - sum(g), sum(u), len(u) - sum(u)
            strata.append((a, b, c, d))
            rows[lvl] = {"gain": [a, a + b], "unch": [c, c + d],
                         "gain_share": 100 * a / (a + b) if a + b else None,
                         "unch_share": 100 * c / (c + d) if c + d else None}
        out[key] = {"strata": rows, "mh": mh(strata), "crude_OR": crude_or(strata)}
    return out


def main():
    learned, seed = [], []
    per_cell = {}
    for model in MODELS:
        for bench in BENCHES:
            rows = {arm: task_rows(D.load_cell(model, arm, bench)) for arm in ARMS}
            s = pairs(rows["native-seed"], rows["native-noseed"])
            l = [p for arm in ARMS[2:] for p in pairs(rows[arm], rows["native-seed"])]
            seed += s
            learned += l
            per_cell[f"{model}|{bench}"] = {"learned": l, "seed": s}
    res = {"theta": THETA, "learned_over_seed": analyse(learned, "learned skills over seed"),
           "seed_over_noskill": analyse(seed, "seed over no skill"),
           "per_cell_pairs": per_cell}
    # transition-matched comparison: same k_ref->k_own, learned gains vs seed gains
    lt, st = res["learned_over_seed"]["transitions"], res["seed_over_noskill"]["transitions"]
    res["matched_gain_transitions"] = {k: {"learned": [lt[k]["conv"], lt[k]["n"]], "seed": [st[k]["conv"], st[k]["n"]]}
                                       for k in lt if k in st and int(k[-1]) > int(k[0])}
    # expected seed-contrast gain convergence if it had the learned contrast's transition mix
    num = den = 0
    for k, v in lt.items():
        if int(k[-1]) > int(k[0]) and k in st:
            num += v["n"] * st[k]["share"] / 100
            den += v["n"]
    res["seed_rate_at_learned_mix"] = 100 * num / den if den else None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    for key in ("learned_over_seed", "seed_over_noskill"):
        r = res[key]
        print(f"== {r['label']} (n={r['n']}): gains conv {r['gain_conv']}, unchanged conv {r['unch_conv']}")
        for k, v in r["transitions"].items():
            lo, hi = wilson(v["conv"], v["n"])
            print(f"   {k}: {v['conv']}/{v['n']} = {v['share']:.1f}%  [{lo:.0f}, {hi:.0f}]")
        for s in ("by_start", "by_end"):
            print(f"   {s}: crude OR {r[s]['crude_OR']:.2f}  MH {r[s]['mh']}")
            for lvl, v in r[s]["strata"].items():
                print(f"      level {lvl}: gain {v['gain']} unch {v['unch']}")
    print("matched gain transitions:", res["matched_gain_transitions"])
    print("seed-contrast gain conv rate reweighted to learned mix:", res["seed_rate_at_learned_mix"])


if __name__ == "__main__":
    main()
