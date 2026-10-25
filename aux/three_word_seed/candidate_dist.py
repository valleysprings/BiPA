"""(d, extra) Whole candidate distribution vs copies of the 3-word seed under the paper's fitted model: P(one copy scores >= x dev tasks) and how many of the run's K candidates reached x.
Run: python -m aux.three_word_seed.candidate_dist"""
import json, math
import numpy as np
from src.analysis import selection as S
from src import layout
OUT = layout.OUT / "three_word_seed"
# the file lists of selection_replay.py, re-implemented (importing it runs the full replay)
import glob
EXE = layout.EXECUTION / "gpt-6-luna"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.load(open(OUT / "selection_replay.json"))
    out = {}
    for bi, bench in enumerate(("automationbench", "tau3-bench")):
        dev_files = sorted(glob.glob(str(EXE / f"*/{bench}/fromscratch/results/evaluations/dev/cf7b932190ed-*/evaluation.json")))
        if bench == "automationbench":
            rr = json.load(open(EXE / "test-reps/automationbench__native-seed__fromscratch/results.json")); test_files = [p for r in rr["runs"] for p in r["evaluation_paths"]]
        else:
            test_files = sorted(glob.glob(str(EXE / f"*/{bench}/fromscratch/results/evaluations/test/cf7b932190ed-*/evaluation.json")))
        dev, test = S.executions(dev_files), S.executions(test_files)
        a, b = S.fit_prior([(sum(v), len(v)) for v in list(dev.values()) + list(test.values())])
        S.RNG = np.random.default_rng([20261024, bi, 123])
        p = S.task_probabilities(dev, (a, b), len(dev), 400_000)
        sc = S.outcomes(p)
        cands = {opt: sel[bench]["rules"][opt]["candidate_dev_tasks"] for opt in ("skillopt", "gepa")}
        out[bench] = {"copy_score_mean_tasks": float(sc.mean()), "copy_score_sd_tasks": float(sc.std()), "per_x": {}}
        for opt, cs in cands.items():
            for x in sorted(set(cs)):
                pr = float((sc >= x).mean()); n_ge = sum(c >= x for c in cs)
                tail = sum(math.comb(len(cs), j) * pr ** j * (1 - pr) ** (len(cs) - j) for j in range(n_ge, len(cs) + 1))
                out[bench]["per_x"].setdefault(opt, {})[str(x)] = dict(P_copy_ge_x=pr, candidates_ge_x=n_ge, K=len(cs), binom_tail_if_independent=tail)
        print(bench, "copy mean %.2f sd %.2f tasks" % (sc.mean(), sc.std()))
        for opt, d in out[bench]["per_x"].items():
            for x, v in d.items():
                print("  ", opt, "x>=%s" % x, "P(copy>=x)=%.4f" % v["P_copy_ge_x"], "cands>=x: %d/%d" % (v["candidates_ge_x"], v["K"]), "binom tail %.2e" % v["binom_tail_if_independent"])
    json.dump(out, open(OUT / "candidate_dist.json", "w"), indent=1)


if __name__ == "__main__":
    main()
