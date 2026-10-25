"""Success difference to the seed, in-domain vs off-distribution, with paired per-task bootstrap intervals (transfer figure).

In-domain DeepSeek: the nine Table-1 skills vs the seed (per-task 3-run means; dynamics.load_cell).
In-domain Luna tau3: Luna's own three skills vs Luna's seed. Cross-model: the DeepSeek tau3 skills under Luna (avg@3;
transfer.cell_data). Cross-benchmark: the six single-run transfers vs the target seed's Table-1 3-run means
(single run per task minus the seed's per-task mean). rho: input-token ratios as reported in the appendix text.
Output: OUT/transfer_and_trajectories/transfer_dq.json.
Run:  python -m aux.transfer_and_trajectories.transfer_dq
"""
import json, random

import numpy as np

from src import layout
from src.analysis import dynamics as D
from src.analysis import transfer as L
from src.utils.dataloader import DS

HERE = layout.OUT / "transfer_and_trajectories"


def boot(d, n=5000):
    d = list(d)
    b = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(n))
    return 100 * float(np.mean(d)), 100 * b[int(0.025 * n)], 100 * b[int(0.975 * n) - 1], len(d)


def means3(reps):
    ks = sorted(reps)
    tasks = [t for t in set.intersection(*(set(reps[k]) for k in ks)) if all(reps[k][t]["success"] is not None for k in ks)]
    return {t: float(np.mean([float(reps[k][t]["success"]) for k in ks])) for t in tasks}


def single(dirs):
    """{task: success} of a single-run transfer, base evaluation merged with retry supplements."""
    out = {}
    for d in dirs:
        for p in sorted(d.glob("results/evaluations/test/**/evaluation.json"), key=lambda p: len(p.parts)):
            for ep in json.loads(p.read_text())["episodes"]:
                if type(ep.get("success")) is bool and ep.get("outcome_valid", True) is True and ep["task_id"] not in out:
                    out[ep["task_id"]] = float(ep["success"])
    return out


def main():
    random.seed(0)
    rows = []
    OPT = {"gepa": "GEPA", "skillopt": "SkillOpt", "skilladam": "SkillAdam"}
    BN = {"tau3-bench": "tau3", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}
    for bench in ("terminal-bench-2-1", "tau3-bench", "automationbench"):
        seed = means3(D.load_cell("deepseek-flash", "native-seed", bench))
        for arm, name in OPT.items():
            own = means3(D.load_cell("deepseek-flash", arm, bench))
            q, lo, hi, n = boot(own[t] - seed[t] for t in own if t in seed)
            rows.append({"group": "in-domain", "model": "DeepSeek", "bench": BN[bench], "method": name, "dQ": q, "lo": lo, "hi": hi, "n": n})
    for src, tgt in (("terminal-bench-2-1", "automationbench"), ("automationbench", "terminal-bench-2-1")):
        seed = means3(D.load_cell("deepseek-flash", "native-seed", tgt))
        for arm, name in OPT.items():
            tag = f"ood-from-bench-{src}"
            own = single([DS / f"{arm}/{tgt}/{tag}", DS / f"test-reps/{tgt}__{arm}__{tag}"])
            q, lo, hi, n = boot(own[t] - seed[t] for t in own if t in seed)
            rows.append({"group": "cross-benchmark", "model": "DeepSeek", "bench": f"{BN[src]}->{BN[tgt]}", "method": name, "dQ": q, "lo": lo, "hi": hi, "n": n})
    cells = {(a, t): L.cell_data(a, t, "tau3-bench") for a, t in L.ARMS}
    lseed = L.task_means(cells[("native-seed", "native-20261011")]["reps"])
    for tag, group in (("native-20261012", "in-domain"), ("ood-from-deepseek-flash", "cross-model")):
        for arm, name in OPT.items():
            m = L.task_means(cells[(arm, tag)]["reps"])
            q, lo, hi, n = boot(m[t]["S"] - lseed[t]["S"] for t in m if t in lseed)
            rows.append({"group": group, "model": "Luna", "bench": "tau3", "method": name, "dQ": q, "lo": lo, "hi": hi, "n": n})
    for r in rows:
        print(f"{r['group']:16s} {r['model']:8s} {r['bench']:12s} {r['method']:9s} dQ={r['dQ']:+5.1f} [{r['lo']:+5.1f}, {r['hi']:+5.1f}] n={r['n']}")
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "transfer_dq.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
