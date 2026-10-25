"""(c) input-token ratio rho (and turns / tool-call ratios) of the from-scratch skills against the 3-word seed, with the paper's own Table-1 machinery.

Uses src/analysis/table1.py unchanged: luna_row (distinct requests with usage; X = sum input_tokens, R = number of requests, N = executed tool calls),
merged_mean_rows (per-task mean over the runs, only slots with a verified outcome and X,N known), geo (geometric mean over tasks of skill/seed, on tasks with
verified outcome on both sides and N, X > 0).

AutoBench: skills' three reps (test-reps/<bench>__<method>__fromscratch/results.json runs[].evaluation_paths), seed = test-reps/automationbench__native-seed__fromscratch.
tau3: one test evaluation (rep 0) per skill inside each run dir; ratio vs the same run's own seed evaluation, and vs the two seed evaluations pooled (task mean of 2 runs).
Run: python -m aux.three_word_seed.rho
"""
import json
from pathlib import Path
from src import layout
import numpy as np
from src.analysis import table1 as T

EXE = layout.EXECUTION / "gpt-6-luna"
REPS = EXE / "test-reps"
OUT = layout.OUT / "three_word_seed"
ROW = T.luna_row


def rep_files(results_json):
    d = json.load(open(results_json))
    return {r["repetition"]: [Path(p) for p in r["evaluation_paths"]] for r in d["runs"]}


def ratio(skill_reps, seed_reps_list):
    """skill_reps: {rep: [files]}; seed_reps_list: list of {rep: [files]} groups pooled into one reference (task mean over all their runs)."""
    _, mean = T.merged_mean_rows(skill_reps, ROW)
    allrows = []
    for g in seed_reps_list:
        allrows += [T.merged_rows_of(fs, ROW) for _, fs in sorted(g.items())]
    ref = T.accumulate(allrows)
    n, rr = T.geo(mean, ref, turns=True)
    return dict(n_tasks=n, **{k: float(v) for k, v in rr.items()})


def mean_tokens(reps):
    """per-task mean over the reps, then mean over tasks of X, R, N (descriptive)."""
    _, mean = T.merged_mean_rows(reps, ROW)
    xs = [r["X"] for r in mean.values()]; rs = [r["R"] for r in mean.values()]; ns = [r["N"] for r in mean.values()]
    return dict(tasks=len(xs), mean_X=float(np.mean(xs)), mean_R=float(np.mean(rs)), mean_N=float(np.mean(ns)))


def ev_in_run(run, split, vid12):
    fs = sorted((run / f"results/evaluations/{split}").glob(f"{vid12}-*/evaluation.json"))
    return fs


def main():
    OUT.mkdir(parents=True, exist_ok=True)


    out = {}
    # ---------------- AutoBench: three reps each
    ab_seed3 = rep_files(REPS / "automationbench__native-seed__fromscratch/results.json")
    ab_hand = rep_files(REPS / "automationbench__native-seed__native-20261011/results.json")
    ab = {"skillopt": rep_files(REPS / "automationbench__skillopt__fromscratch/results.json"), "gepa": rep_files(REPS / "automationbench__gepa__fromscratch/results.json")}
    out["automationbench"] = {"vs_3word_seed": {}, "vs_hand_seed": {}, "descriptive_mean_per_episode": {}}
    for k, v in ab.items():
        out["automationbench"]["vs_3word_seed"][k] = ratio(v, [ab_seed3])
        out["automationbench"]["vs_hand_seed"][k] = ratio(v, [ab_hand])
    out["automationbench"]["vs_hand_seed"]["3word_seed"] = ratio(ab_seed3, [ab_hand])
    out["automationbench"]["vs_3word_seed"]["hand_seed"] = ratio(ab_hand, [ab_seed3])
    for name, reps in [("3word_seed", ab_seed3), ("hand_seed", ab_hand), *ab.items()]:
        out["automationbench"]["descriptive_mean_per_episode"][name] = mean_tokens(reps)

    # ---------------- tau3: one test evaluation each, rep 0
    tau = {}
    for opt, ret in (("skillopt", "dfd18dc47eb0"), ("gepa", "d1ee2d3a774f")):
        run = EXE / opt / "tau3-bench/fromscratch"
        seed = {0: ev_in_run(run, "test", "cf7b932190ed")}
        skill = {0: ev_in_run(run, "test", ret)}
        assert len(seed[0]) == 1 and len(skill[0]) == 1, (seed, skill)
        tau[opt] = dict(seed=seed, skill=skill, run=run)
        if opt == "skillopt":   # the last-iterate (post slow-update) skill that was NOT returned
            tau[opt]["last_iterate"] = {0: ev_in_run(run, "test", "d0f2b28fe27d")}
    out["tau3-bench"] = {"vs_own_seed_eval": {}, "vs_pooled_two_seed_evals": {}, "seed_vs_seed": {}, "descriptive_mean_per_episode": {}}
    pool = [tau["skillopt"]["seed"], tau["gepa"]["seed"]]
    for opt in ("skillopt", "gepa"):
        out["tau3-bench"]["vs_own_seed_eval"][opt] = ratio(tau[opt]["skill"], [tau[opt]["seed"]])
        out["tau3-bench"]["vs_pooled_two_seed_evals"][opt] = ratio(tau[opt]["skill"], pool)
    out["tau3-bench"]["vs_own_seed_eval"]["skillopt_last_iterate_d0f2b28fe27d_NOT_returned"] = ratio(tau["skillopt"]["last_iterate"], [tau["skillopt"]["seed"]])
    out["tau3-bench"]["vs_pooled_two_seed_evals"]["skillopt_last_iterate_d0f2b28fe27d_NOT_returned"] = ratio(tau["skillopt"]["last_iterate"], pool)
    # seed evaluated in run A vs seed evaluated in run B (how much rho moves between two evaluations of the same text)
    out["tau3-bench"]["seed_vs_seed"]["skillopt_run_seed_over_gepa_run_seed"] = ratio(tau["skillopt"]["seed"], [tau["gepa"]["seed"]])
    for name, reps in (("seed_in_skillopt_run", tau["skillopt"]["seed"]), ("seed_in_gepa_run", tau["gepa"]["seed"]), ("skillopt_returned", tau["skillopt"]["skill"]), ("gepa_returned", tau["gepa"]["skill"]),
                       ("skillopt_last_iterate", tau["skillopt"]["last_iterate"])):
        out["tau3-bench"]["descriptive_mean_per_episode"][name] = mean_tokens(reps)
    # hand-written seed vs 3-word seed on tau3 (hand-written: 3 reps from test-reps)
    tau_hand = rep_files(REPS / "tau3-bench__native-seed__native-20261011/results.json")
    out["tau3-bench"]["hand_seed_rep0_to_2_vs_pooled_3word"] = ratio(tau_hand, pool)
    out["tau3-bench"]["descriptive_mean_per_episode"]["hand_seed_3reps"] = mean_tokens(tau_hand)
    out["tau3-bench"]["hand_seed_rep0_only_vs_pooled_3word"] = ratio({0: tau_hand[0]}, pool)

    json.dump(out, open(OUT / "rho.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
