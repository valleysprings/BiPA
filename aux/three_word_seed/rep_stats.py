"""(a) Test numbers: hand-written seed vs 3-word seed vs from-scratch skills; paired task-level differences with bootstrap intervals (AutoBench avg@3), and tau3 single-run comparisons.
Per-task success = mean over the three reps of the verified outcome (all outcomes verified: known 120/120 in every rep).
Run: python -m aux.three_word_seed.rep_stats"""
import glob, json
from pathlib import Path
from src import layout
import numpy as np
from src.analysis import table1 as T

EXE = layout.EXECUTION / "gpt-6-luna"
REPS = EXE / "test-reps"
OUT = layout.OUT / "three_word_seed"


def reps(tag):
    rr = json.load(open(REPS / tag / "results.json"))
    out = {}
    for r in rr["runs"]:
        eps = T.merged_episodes([Path(p) for p in r["evaluation_paths"]])
        out[r["repetition"]] = {t: int(bool(e["success"])) for (t, _), e in eps.items() if type(e.get("success")) is bool and e.get("outcome_valid", True) is True}
    return out


def per_task(r):
    tasks = sorted(set.intersection(*[set(v) for v in r.values()]))
    return tasks, np.array([[r[k][t] for t in tasks] for k in sorted(r)])  # reps x tasks


AB = {"hand_seed": "automationbench__native-seed__native-20261011", "seed3": "automationbench__native-seed__fromscratch", "skillopt_fs": "automationbench__skillopt__fromscratch", "gepa_fs": "automationbench__gepa__fromscratch",
      "skillopt_native": "automationbench__skillopt__native-20261012", "gepa_native": "automationbench__gepa__native-20261012"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {k: per_task(reps(v)) for k, v in AB.items()}
    base_tasks = data["hand_seed"][0]
    assert all(data[k][0] == base_tasks for k in data)
    res = {"automationbench": {"avg3_pct": {}, "rep_pct": {}, "paired": {}}}
    for k, (tasks, M) in data.items():
        res["automationbench"]["avg3_pct"][k] = float(100 * M.mean())
        res["automationbench"]["rep_pct"][k] = [float(100 * x) for x in M.mean(1)]
    rng = np.random.default_rng(20261024)
    for a, b in (("skillopt_fs", "hand_seed"), ("gepa_fs", "hand_seed"), ("skillopt_fs", "seed3"), ("gepa_fs", "seed3"), ("hand_seed", "seed3"), ("skillopt_fs", "skillopt_native"), ("gepa_fs", "gepa_native"), ("gepa_fs", "skillopt_fs")):
        da = data[a][1].mean(0) - data[b][1].mean(0)
        bs = [100 * da[rng.integers(0, len(da), len(da))].mean() for _ in range(10000)]
        res["automationbench"]["paired"][f"{a} - {b}"] = dict(diff_pts=float(100 * da.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                                              tasks_up=int((da > 0).sum()), tasks_down=int((da < 0).sum()), tasks_equal=int((da == 0).sum()))
    # seed rep-to-rep spread (hand-written, 3-word)
    for k in ("hand_seed", "seed3"):
        r = np.array(res["automationbench"]["rep_pct"][k]); res["automationbench"][f"rep_sd_{k}"] = float(r.std(ddof=1))

    # tau3: single evaluation each
    tau = {}
    for opt, ret in (("skillopt", "dfd18dc47eb0"), ("gepa", "d1ee2d3a774f")):
        run = EXE / opt / "tau3-bench/fromscratch"
        def one(v):
            f = glob.glob(str(run / f"results/evaluations/test/{v}-*/evaluation.json"))[0]
            return {e["task_id"]: int(bool(e["success"])) for e in json.load(open(f))["episodes"] if type(e.get("success")) is bool}
        s, r = one("cf7b932190ed"), one(ret)
        tasks = sorted(set(s) & set(r))
        tau[opt] = dict(n=len(tasks), seed_pct=100 * np.mean([s[t] for t in tasks]), returned_pct=100 * np.mean([r[t] for t in tasks]),
                        gained=sum(s[t] == 0 and r[t] == 1 for t in tasks), lost=sum(s[t] == 1 and r[t] == 0 for t in tasks))
        if opt == "skillopt":
            l = one("d0f2b28fe27d"); tau[opt]["last_iterate_d0f2b28fe27d_pct"] = 100 * np.mean([l[t] for t in tasks]); tau[opt]["last_iterate_gained"] = sum(s[t] == 0 and l[t] == 1 for t in tasks); tau[opt]["last_iterate_lost"] = sum(s[t] == 1 and l[t] == 0 for t in tasks)
    hand = reps("tau3-bench__native-seed__native-20261011")
    tau["hand_seed_rep_pct"] = [float(100 * np.mean(list(v.values()))) for k, v in sorted(hand.items())]
    tau["hand_seed_avg3_pct"] = float(np.mean(tau["hand_seed_rep_pct"]))
    res["tau3-bench"] = tau
    json.dump(res, open(OUT / "test_stats.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
