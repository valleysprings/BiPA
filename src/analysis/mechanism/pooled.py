"""Pooled dual-currency table (paragraph 'Success gains do not survive reruns', facts-v9 Sec. 6): per optimizer x benchmark, success pooled over the three runs
(main + two reruns) and the pooled geometric-mean ratio to the seed of input tokens X, price P (cached 0.1, output 4), turns R, executed calls N and output tokens O,
with the seed's own run-to-run floor (pairwise and leave-one-out ratios). Loaders are shared with analysis/mechanism/reproducibility.py.
Output: assets/cache/analysis/pooled/pooled.json and pooled.txt. Run: <bach-harness venv python> -m src.analysis.mechanism.pooled   (from bipa/)"""
import json, math
import numpy as np
from src.utils.dataloader import BN, gmean, out_dir
from src.utils.logger import Report
from src.analysis.mechanism.reproducibility import load, run_files

KEYS = ["X", "P", "R", "N", "O"]


def gm_ratio(A, Bs, tasks, k):
    """geometric-mean over tasks of (per-task geometric mean over runs in A) / (same over Bs)."""
    d = []
    for t in tasks:
        a = [r[t][k] for r in A if t in r and r[t][k] > 0]; b = [r[t][k] for r in Bs if t in r and r[t][k] > 0]
        if a and b: d.append(sum(map(math.log, a)) / len(a) - sum(map(math.log, b)) / len(b))
    return gmean(np.exp(d)), len(d)


def main():
    out, rawf = {}, []
    for short, bench in ((v, k) for k, v in BN.items()):
        F = run_files(bench, noskill=True); rawf += [f for v in F.values() for n, f in v]; R = {k: [load(f) for n, f in v] for k, v in F.items()}
        seed = R["seed"]; st = set.intersection(*[set(r) for r in seed])
        floor = {}
        for k in KEYS:
            rs = [gm_ratio([seed[j]], [seed[i]], st, k)[0] for i in range(len(seed)) for j in range(i + 1, len(seed))]
            loo = [gm_ratio([seed[i]], [s for j, s in enumerate(seed) if j != i], st, k)[0] for i in range(len(seed))] if len(seed) > 2 else []
            floor[k] = dict(pairs=[round(x, 3) for x in rs], max_dev=round(max(abs(math.log(x)) for x in rs), 3) if rs else None, loo=[round(x, 3) for x in loo])
        rows = {}
        for name, runs in R.items():
            tasks = set.intersection(*[set(r) for r in runs])
            s = [sum(r[t]["succ"] for t in r) / len(r) for r in runs]
            row = dict(runs=len(runs), run_names=[n for n, f in F[name]], n_tasks=len(runs[0]), succ_runs=[round(100 * x, 1) for x in s], succ_pooled=round(100 * sum(s) / len(s), 1))
            if name != "seed":
                for k in KEYS:
                    v, n = gm_ratio(runs, seed, tasks & st, k); row["ratio_" + k] = round(v, 3); row["n_paired"] = n
            rows[name] = row
        out[short] = dict(rows=rows, seed_floor=floor, seed_run_means={k: round(sum(sum(r[t][k] for t in r) / len(r) for r in seed) / len(seed)) for k in KEYS})
    json.dump(out, open(out_dir("pooled")/"pooled.json", "w"), indent=1)
    say = Report()
    for b, v in out.items():
        say("##", b, "floor", {k: (f["pairs"], f["max_dev"]) for k, f in v["seed_floor"].items() if k in "XPRN"})
        for n, r in v["rows"].items(): say(n, r)
    say.save("pooled", "pooled.txt")


if __name__ == "__main__":
    main()
