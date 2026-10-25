"""Collect the episode graphs the bipartite figures need into one compact pickle (OUT/token_attribution/bipdata.pkl).

DeepSeek-V4.1-Flash: the paper's own test files (src.analysis.attribution.run_files: skills native-20261008-https + test-reps, seed native-20261003-trace-final
+ rep-1/rep-2), graphs from src.graph.episode.load.  GPT-6-Luna: attribution_luna.luna_side picks the files exactly as Table 1 does, graphs
from attribution_luna's graph_cache.pkl (provider-failed requests already removed).

Per episode: task, rep, outcome, R, N, X and the R x 4 matrix of tokens read per turn by component group (group_of).
For the four-term / content split, per cell (model, bench, optimizer) and usable task (the tasks of attribution.json / attribution_luna.json),
the 3-run mean tokens of each group for the seed side and the skill side.

Run:  python -m aux.token_attribution.bipdata   (after attribution_luna)
"""
import collections
import json
import pickle

import numpy as np

from aux.token_attribution import attribution_luna
from aux.token_attribution.bipcommon import OUT, group_of, load_cached
from src.analysis import attribution as A

GROUPS = ["prompt", "skill", "tools", "messages"]
LUNA_JSON = OUT / "attribution_luna.json"
LUNA_GCACHE = OUT / "graph_cache.pkl"
BENCH = [("tau3-bench", "tau3"), ("automationbench", "AutoB"), ("terminal-bench-2-1", "TB2")]


def compact(e):
    """task, outcome, R, N, X and the R x 4 tokens-per-turn matrix by group."""
    M = np.zeros((e.R, len(GROUPS)))
    for j, c in enumerate(e.comps):
        M[:, GROUPS.index(group_of(c))] += e.Xrg[:, j]
    return dict(task=e.task, success=e.success, R=int(e.R), N=int(e.N), X=float(e.X), M=M)


def main():
    out = dict(episodes={}, cells={}, files={})
    # DeepSeek
    dsj = json.loads(LUNA_JSON.read_text())["deepseek"]
    seen_seed = {}
    for disp, m in A.OPT:
        for bench, short in BENCH:
            own_files, seed_files = A.run_files(m, bench)
            own = {rep: {e.task: e for e in load_cached(p)} for rep, p in own_files.items()}
            seed = {rep: {e.task: e for e in load_cached(p)} for rep, p in seed_files.items()}
            out["files"][("deepseek-flash", bench, m)] = {str(k): str(v) for k, v in own_files.items()}
            out["files"][("deepseek-flash", bench, "seed")] = {str(k): str(v) for k, v in seed_files.items()}
            out["episodes"][("deepseek-flash", bench, m)] = [dict(compact(e), rep=rep) for rep, d in own.items() for e in d.values()]
            if ("deepseek-flash", bench) not in seen_seed:
                seen_seed[("deepseek-flash", bench)] = 1
                out["episodes"][("deepseek-flash", bench, "seed")] = [dict(compact(e), rep=rep) for rep, d in seed.items() for e in d.values()]
            # usable tasks of the paper's four-term split: the task lists stored with the Luna script's DeepSeek recomputation (same rule as attribution.compute)
            cell = next(c for c in dsj if c['method'] == disp and c['bench'] == bench)
            usable = cell['tasks']
            side = lambda d, t: np.mean([compact(d[r][t])["M"].sum(0) for r in (0, 1, 2)], axis=0)
            out["cells"][("deepseek-flash", bench, m)] = dict(
                tasks_rebuilt=usable, n_json=cell["n"], seed={t: side(seed, t) for t in usable}, own={t: side(own, t) for t in usable})
            print("DS", disp, short, "usable rebuilt", len(usable), "json n", cell["n"])
    # Luna
    L = attribution_luna
    G = pickle.loads(LUNA_GCACHE.read_bytes())
    lj = json.loads(LUNA_JSON.read_text())["luna"]
    for bench, short in (("tau3-bench", "tau3"), ("automationbench", "AutoB"), ("terminal-bench-2-1", "TB2")):
        arms = ["native-seed"] + ([m for _, m in A.OPT] if bench != "terminal-bench-2-1" else [])
        sides = {}
        for arm in arms:
            s = L.luna_side(arm, bench)
            per = collections.defaultdict(list)   # task -> [(rep, compact)]
            eps = []
            for k, rep in s["reps"].items():
                for t, (e, p) in rep.items():
                    row = L.luna_row(e)
                    if row["X"] is None:
                        continue
                    key = (str(p), t, p.stat().st_mtime_ns)
                    g = G.get(key)
                    if g is None:
                        continue
                    c = dict(compact(g), rep=k)
                    per[t].append(c)
                    eps.append(c)
            sides[arm] = per
            name = "seed" if arm == "native-seed" else arm
            out["episodes"][("gpt-6-luna", bench, name)] = eps
            out["files"][("gpt-6-luna", bench, name)] = s["run"]
        for disp, m in A.OPT:
            if bench == "terminal-bench-2-1":
                continue
            cj = next(c for c in lj if c["method"] == disp and c["bench"] == bench)
            tasks = cj["tasks"]
            mean = lambda per, t: np.mean([c["M"].sum(0) for c in per[t]], axis=0)
            out["cells"][("gpt-6-luna", bench, m)] = dict(
                tasks_rebuilt=tasks, n_json=cj["n"], seed={t: mean(sides["native-seed"], t) for t in tasks}, own={t: mean(sides[m], t) for t in tasks})
            print("Luna", disp, short, "n", len(tasks))
    (OUT / "bipdata.pkl").write_bytes(pickle.dumps(out))
    print({k: len(v) for k, v in out["episodes"].items()})


if __name__ == "__main__":
    main()
