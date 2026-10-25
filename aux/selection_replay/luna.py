"""mech_selection_luna: the selection replay of aux.selection_replay.selection for the GPT-6 Luna runs of Table 1.

Run: python -m aux.selection_replay.luna   -> CACHE/analysis/main/selection_luna.json, data/mech/mech_selection_luna.md
"""
import glob
import json
from collections import defaultdict

import numpy as np

from aux.selection_replay.selection import (BENCHES, DRAWS, MODERATE, POWER_GAINS, SEED_VERSION, batch_gain_tail, best_of, check_inputs,
                                            executions, fit_prior, mid_quantile, outcomes, posterior_mean, replay_power, reseed, seed_files,
                                            signature, skilladam_accepts, skilladam_null, skilladam_observed, skilladam_reported_gain,
                                            task_probabilities, verified)
from src import layout
from src.analysis.table1 import rep_groups
from src.utils import dataloader, logger

# The same replay for the Luna runs of Table 1 (optimizers native-20261012, seed native-20261011), with its own inputs, checks,
# streams and record.
#
# Inputs.  The seed's dev executions are its dev evaluations under the native protocol (run tags native-20261011/12; every optimizer run and
# the seed run reuse one cached draw, so each dev task has one execution).  The aborted runs (aborted_dev_files; an earlier task config and
# cache namespace, stopped at the seed's dev evaluation on unknown outcomes) are not seed draws of this protocol; they are compared in the
# record only.  The seed's test executions are the files Table 1 reads (native-seed run + test-reps, repetitions 0-2).
# Task models.  fitted / moderate as for DeepSeek.  With one execution per dev task a prior-free model built on the seed's executions alone
# is degenerate (p_v in {0, 1}, so G0 = 0); the prior-free models are therefore estimated under the null: if no candidate differs from the
# seed, every dev execution of every version in the benchmark's three optimizer runs is a seed draw, and each task's rate is its share of
# successes among them (plug-in) or Beta(s, n - s) (Bayesian bootstrap; point masses for unanimous tasks).  A candidate that truly differs
# widens these rates, which makes P(G0 >= g) larger.
# Model-free.  Under the null the seed draw and the K candidate scores of a SkillOpt or GEPA run are exchangeable; the rank test gives
# P(max of the other K - the one >= g) averaged over which of the K + 1 scores is the baseline.
LUNA = "gpt-6-luna"
LUNA_EXECUTION = layout.EXECUTION / LUNA
LUNA_RUN = "native-20261012"
LUNA_SEED_RUN = "native-20261011"
LUNA_TEST_AVG3 = {"tau3-bench": 69.3, "automationbench": 37.2, "terminal-bench-2-1": 64.8}   # Table 1, Luna Seed skill row (mech_table1.md)
# (dev tasks, distinct dev executions, distinct test executions) of the Luna inputs; a new run that matches the globs fails loudly
LUNA_PAPER_INPUTS = {"terminal-bench-2-1": (9, 9, 51), "tau3-bench": (30, 30, 225), "automationbench": (30, 30, 360)}
LUNA_MODELS = ("fitted", "moderate", "null plug-in", "null Bayesian bootstrap")


def luna_seed_files(bench) -> tuple[list, list]:
    """(dev evaluation files, test evaluation files) of the seed skill under Luna's native protocol; test = the files Table 1 reads."""
    dev_files = sorted(glob.glob(str(LUNA_EXECUTION / f"*/{bench}/native-2026101[12]/results/evaluations/dev/{SEED_VERSION}-*/evaluation.json")))
    seed_run = LUNA_EXECUTION / "native-seed" / bench / LUNA_SEED_RUN
    groups = rep_groups([seed_run, layout.repetition_dir(seed_run), seed_run.parent / "rep-1", seed_run.parent / "rep-2"],
                               dataloader.vid(dataloader.SEED_TEXT))
    return dev_files, [str(f) for _, fs in sorted(groups.items()) for f in fs]


def luna_check_inputs(bench, dev, test):
    """Fail if the Luna seed executions differ from LUNA_PAPER_INPUTS."""
    found = (len(dev), sum(map(len, dev.values())), sum(map(len, test.values())))
    assert found == LUNA_PAPER_INPUTS[bench], f"{bench}: Luna seed executions {found} differ from {LUNA_PAPER_INPUTS[bench]}"


def luna_dev_draws(dev_files, n_dev) -> list[dict]:
    """The Luna seed's distinct full-panel dev draws (dev_draws with the Luna path in the run names)."""
    draws = defaultdict(lambda: {"runs": []})
    for path in dev_files:
        episodes = json.load(open(path))["episodes"]
        if len(episodes) != n_dev:
            continue
        ok = [e for e in episodes if verified(e)]
        draw = draws[frozenset((e["task_id"], signature(e)) for e in ok)]
        draw.update(solved=sum(bool(e["success"]) for e in ok), verified=len(ok))
        draw["runs"].append(path.split(f"{LUNA}/")[1].split("/results")[0])
    return sorted(draws.values(), key=lambda d: d["solved"] / d["verified"])


def luna_null_executions(bench) -> dict:
    """{task: [0/1 per distinct execution]} over every dev evaluation (any version) of the benchmark's Luna optimizer runs and seed run."""
    files = glob.glob(str(LUNA_EXECUTION / f"*/{bench}/native-2026101[12]/results/evaluations/dev/*/evaluation.json"))
    return executions(files)


def aborted_dev_files(bench) -> list:
    """Seed dev evaluations of the aborted SkillOpt run: the one non-native run of the cell that has a seed dev evaluation but no summary.json."""
    runs = [d for d in (LUNA_EXECUTION / "skillopt" / bench).iterdir()
            if d.is_dir() and not d.name.startswith("native-") and not (d / "summary.json").exists()
            and any((d / "results/evaluations/dev").glob(f"{SEED_VERSION}-*/evaluation.json"))]
    assert len(runs) == 1, f"{bench}: {len(runs)} aborted Luna SkillOpt runs"
    return sorted(str(f) for f in (runs[0] / "results/evaluations/dev").glob(f"{SEED_VERSION}-*/evaluation.json"))


def luna_aborted_comparison(bench, dev) -> dict:
    """The aborted run's seed dev draw against the native draw on the tasks verified in both."""
    other = executions(aborted_dev_files(bench))
    common = sorted(set(dev) & set(other))
    return {"verified": len(other), "solved": sum(v[0] for v in other.values()), "common": len(common),
            "native_solved_common": sum(dev[t][0] for t in common), "aborted_solved_common": sum(other[t][0] for t in common),
            "native_1_aborted_0": sum(dev[t][0] == 1 and other[t][0] == 0 for t in common),
            "native_0_aborted_1": sum(dev[t][0] == 0 and other[t][0] == 1 for t in common)}


def luna_observed(bench, n_dev, seed_solved) -> dict:
    """What the Luna Table 1 runs recorded.  SkillOpt's summary leaves baseline_selection_hard empty on Luna: its baseline is the seed draw
    the run started at (history.json's starting current_score), its candidates the history's selection scores."""
    so_run = LUNA_EXECUTION / f"skillopt/{bench}/{LUNA_RUN}"
    so = json.load(open(so_run / "summary.json"))["extra"]
    history = json.load(open(so_run / "skillopt/history.json"))
    ge = json.load(open(LUNA_EXECUTION / f"gepa/{bench}/{LUNA_RUN}/summary.json"))["extra"]
    sa = json.load(open(LUNA_EXECUTION / f"skilladam/{bench}/{LUNA_RUN}/summary.json"))["extra"]
    scored = [i for i in sa["iterations"] if isinstance(i.get("baseline"), (int, float)) and isinstance(i.get("candidate"), (int, float))]
    baseline = round(so["baseline_selection_hard"] * n_dev) if so["baseline_selection_hard"] is not None else seed_solved
    candidates = [round(h["selection_hard"] * n_dev) for h in history]
    assert so["total_accepts"] + so["total_rejects"] == len(candidates)
    vals = ge["val_scores"]
    return {
        "SkillOpt": {"K": len(candidates), "baseline": baseline, "candidates": candidates,
                     "gain": max(round(so["best_selection_hard"] * n_dev) - baseline, 0), "returned": so["best_origin"]},
        "GEPA": {"K": len(vals) - 1, "baseline": round(vals[0] * n_dev), "candidates": [round(v * n_dev) for v in vals[1:]],
                 "gain": round((max(vals) - vals[0]) * n_dev), "returned": f"index {ge['best_idx']}"},
        "SkillAdam": skilladam_observed(scored),
    }


def null_probabilities(model, null_exec, tasks, draws, rng):
    """(draws, tasks) rates under the null-estimated prior-free models: plug-in s/n, or Beta(s, n - s) with point masses at 0 / 1."""
    s = np.array([sum(null_exec[t]) for t in tasks], dtype=float)
    n = np.array([len(null_exec[t]) for t in tasks], dtype=float)
    if model == "null plug-in":
        return np.broadcast_to(s / n, (draws, len(tasks))).copy()
    p = rng.beta(np.maximum(s, 1e-12), np.maximum(n - s, 1e-12), size=(draws, len(tasks)))
    p[:, s == 0], p[:, s == n] = 0.0, 1.0
    return p


def rank_test(baseline, candidates, gain) -> float:
    """Model-free P(G0 >= gain) under exchangeable scores: over each of the K + 1 scores as the baseline, max(others) - it >= gain."""
    scores = [baseline] + list(candidates)
    return float(np.mean([max(scores[:i] + scores[i + 1:]) - s >= gain for i, s in enumerate(scores)]))


def analyse_luna(bench) -> dict:
    """Every Luna number of one benchmark."""
    dev_files, test_files = luna_seed_files(bench)
    dev, test = executions(dev_files), executions(test_files)
    luna_check_inputs(bench, dev, test)
    n_dev = len(dev)
    draws = luna_dev_draws(dev_files, n_dev)
    assert len(draws) == 1, f"{bench}: {len(draws)} distinct Luna seed dev draws"
    seed_solved = draws[0]["solved"]
    pooled = [(sum(v), len(v)) for v in list(dev.values()) + list(test.values())]
    a, b = fit_prior(pooled)
    mean = sum(k for k, _ in pooled) / sum(n for _, n in pooled)
    obs = luna_observed(bench, n_dev, seed_solved)
    null_exec = luna_null_executions(bench)
    tasks = sorted(dev)
    test_runs = [sum(v) for v in test.values()]
    out = {"dev_tasks": n_dev, "dev_executions_per_task": sorted({len(v) for v in dev.values()}), "test_tasks": len(test),
           "prior_fitted": [a, b], "mean_success": mean, "observed": obs, "seed_dev_draws": draws, "test_avg3": LUNA_TEST_AVG3[bench],
           "test_always_sometimes_never": [sum(k == 3 for k in test_runs), sum(0 < k < 3 for k in test_runs), sum(k == 0 for k in test_runs)],
           "expected_seed_success": {"dev": posterior_mean(dev, a, b), "test": posterior_mean(test, a, b), "prior_mean": 100 * a / (a + b)},
           "null_executions": {"per_task_min": min(len(null_exec[t]) for t in tasks), "per_task_max": max(len(null_exec[t]) for t in tasks),
                               "unanimous": sum(sum(null_exec[t]) in (0, len(null_exec[t])) for t in tasks),
                               "s_over_n": {t: f"{sum(null_exec[t])}/{len(null_exec[t])}" for t in tasks}},
           "aborted": luna_aborted_comparison(bench, dev),
           "rank_test": {rule: rank_test(obs[rule]["baseline"], obs[rule]["candidates"], max(obs[rule]["gain"], 1))
                         for rule in ("SkillOpt", "GEPA")},
           "models": {}}
    index = [x for x, _ in BENCHES].index(bench)
    for j, name in enumerate(LUNA_MODELS):
        reseed(index, j, 500)   # Luna streams: [20261012, benchmark, model, 500], disjoint from DeepSeek's
        if name in ("fitted", "moderate"):
            prior = (a, b) if name == "fitted" else (MODERATE * mean, MODERATE * (1 - mean))
            p = task_probabilities(dev, prior, n_dev, DRAWS)
        else:
            p = null_probabilities(name, null_exec, tasks, DRAWS, np.random.default_rng([20261012, index, j, 501]))
        rules = {}
        for rule in ("SkillOpt", "GEPA"):
            k, g = obs[rule]["K"], obs[rule]["gain"]
            gain_random = best_of(p, k, outcomes(p))
            gain_fixed = best_of(p, k, obs[rule]["baseline"])
            rules[rule] = {"K": k, "E_gain_random_baseline": float(gain_random.mean()), "E_gain_observed_baseline": float(gain_fixed.mean()),
                           # TB2 gains are 0: P(G0 >= 1) is then the chance that selection alone would have kept a candidate
                           "P_ge_observed_random_baseline": float((gain_random >= max(g, 1)).mean()),
                           "P_ge_observed_observed_baseline": float((gain_fixed >= max(g, 1)).mean())}
            if name == "fitted":
                rules[rule]["power_observed_baseline"] = replay_power(p, k, obs[rule]["baseline"], gain_fixed,
                                                                      np.random.default_rng([20261012, index, j, 511, k]))
        sa = obs["SkillAdam"]
        accepts = skilladam_accepts(p, sa["batch"], sa["budgets"])
        rules["SkillAdam"] = {"E_accepts": float(accepts.mean()), "P_accepts_ge_observed": float((accepts >= max(sa["accepted"], 1)).mean()),
                              "P_better_one_comparison": batch_gain_tail(p, sa["batch"], 1),
                              "P_gain_ge_accepted": {str(g): batch_gain_tail(p, sa["batch"], g) for g in sorted(set(sa["accepted_gains"]))},
                              **skilladam_null(p, sa, np.random.default_rng([20261012, index, j, 507]))}
        if sa["reported_gain_points"] == 0:   # nothing accepted: report P(reported gain > 0) instead of the trivial P(>= 0)
            gain, _ = skilladam_reported_gain(p, sa["batch"], sa["budgets"], np.random.default_rng([20261012, index, j, 507]))
            rules["SkillAdam"]["P_reported_gain_ge_observed"] = float((gain > 0).mean())
        out["models"][name] = {"rules": rules, "seed_draw_mid_quantile": mid_quantile(outcomes(p), seed_solved)}
    return out


def record_luna(result):
    """The audit record data/mech/mech_selection_luna.md."""
    names = {"terminal-bench-2-1": "TB2", "tau3-bench": "tau3", "automationbench": "AutoBench"}
    inputs, runs, rules, sa_rows, power, checks = [], [], [], [], [], []
    for bench, _ in BENCHES:
        r, lab = result[bench], names[bench]
        d, e, nx = r["seed_dev_draws"][0], r["expected_seed_success"], r["null_executions"]
        inputs.append([lab, r["dev_tasks"], r["dev_executions_per_task"], f"{d['solved']}/{d['verified']}", len(d["runs"]), r["test_tasks"],
                       " / ".join(map(str, r["test_always_sometimes_never"])), f"({r['prior_fitted'][0]:.2f}, {r['prior_fitted'][1]:.2f})",
                       f"{r['mean_success']:.3f}", f"{r['models']['fitted']['seed_draw_mid_quantile']:.2f}",
                       f"dev {e['dev']:.1f}, test {e['test']:.1f}; test avg@3 {r['test_avg3']}",
                       f"{nx['per_task_min']}-{nx['per_task_max']} per task, {nx['unanimous']}/{r['dev_tasks']} unanimous"])
        o = r["observed"]
        for rule in ("SkillOpt", "GEPA"):
            runs.append([lab, rule, o[rule]["K"], f"{o[rule]['baseline']}/{r['dev_tasks']}", " ".join(map(str, o[rule]["candidates"])),
                         f"+{o[rule]['gain']}", o[rule]["returned"]])
        s = o["SkillAdam"]
        runs.append([lab, "SkillAdam", f"{s['iterations']} iterations, {s['batch']}-task subsets, {s['attempts']} patches",
                     f"better/worse/tied {s['better']}/{s['worse']}/{s['tied']}", "accepted gains " + (" ".join(f"+{g}" for g in s["accepted_gains"]) or "none"),
                     f"{s['reported_gain_points']:.1f} pts summed", f"{s['accepted']} accepted"])
        for name in LUNA_MODELS:
            m = r["models"][name]
            for rule in ("SkillOpt", "GEPA"):
                x = m["rules"][rule]
                rules.append([lab, name, rule, x["K"], f"{o[rule]['baseline']}/{r['dev_tasks']}", f"+{o[rule]['gain']}",
                              f"{x['E_gain_observed_baseline']:.2f}", f"{x['P_ge_observed_observed_baseline']:.3f}",
                              f"{x['E_gain_random_baseline']:.2f}", f"{x['P_ge_observed_random_baseline']:.3f}"])
                if "power_observed_baseline" in x:
                    pw = x["power_observed_baseline"]
                    power.append([lab, rule, f"{o[rule]['baseline']}/{r['dev_tasks']}", f"+{o[rule]['gain']}", pw["critical_gain_tasks"]]
                                 + [f"{pw[str(g)]:.2f}" for g in POWER_GAINS])
            x = m["rules"]["SkillAdam"]
            tails = ", ".join(f"+{g}: {p:.3f}" for g, p in x["P_gain_ge_accepted"].items()) or "-"
            sa_rows.append([lab, name, f"{s['accepted']} ({s['reported_gain_points']:.1f} pts)", f"{x['E_accepts_cached']:.2f}",
                            f"{x['E_reported_gain_points']:.1f}", f"{x['P_reported_gain_ge_observed']:.3f}",
                            f"{x['P_better_one_comparison']:.3f}", tails])
        bo = r["aborted"]
        checks.append([lab, f"{o['SkillOpt']['baseline']}/{r['dev_tasks']} vs " + " ".join(map(str, o["SkillOpt"]["candidates"])),
                       f"{r['rank_test']['SkillOpt']:.3f}", f"{o['GEPA']['baseline']}/{r['dev_tasks']} vs " + " ".join(map(str, o["GEPA"]["candidates"])),
                       f"{r['rank_test']['GEPA']:.3f}",
                       f"{bo['solved']}/{bo['verified']}; on {bo['common']} common tasks native {bo['native_solved_common']} vs aborted "
                       f"{bo['aborted_solved_common']} ({bo['native_1_aborted_0']} solved only natively, {bo['native_0_aborted_1']} only in the aborted run)"])
    t_in = logger.md_table(["bench", "dev tasks", "seed executions per dev task", "seed dev draw", "runs using it", "test tasks",
                            "seed test always / sometimes / never (3 runs)", "fitted prior (a, b)", "mean success",
                            "mid-quantile of the seed dev draw (fitted)",
                            "expected seed success, fitted posteriors", "dev executions of all versions (null models)"], inputs)
    t_runs = logger.md_table(["bench", "optimizer", "K / iterations", "baseline (starting seed draw)", "candidate dev scores (tasks)",
                              "reported dev gain g", "returned"], runs)
    t_rules = logger.md_table(["bench", "task model", "rule", "K", "starting draw", "g", "E[G0], starting draw", "P(G0 >= max(g,1)), starting draw",
                               "E[G0], random seed draw", "P(G0 >= max(g,1)), random seed draw"], rules)
    t_sa = logger.md_table(["bench", "task model", "accepted (summed gain)", "E[accepts], identical patches", "E[summed gain] (pts)",
                            "P(summed gain >= observed) (TB2: > 0)", "P(better) per comparison", "P(gain >= accepted gain) per comparison"], sa_rows)
    t_pw = logger.md_table(["bench", "rule", "starting draw", "g", "critical gain (one-sided 0.05, tasks)"]
                           + [f"power, one candidate +{g} task{'s' if g > 1 else ''}" for g in POWER_GAINS], power)
    t_chk = logger.md_table(["bench", "SkillOpt scores (seed vs candidates)", "rank test P", "GEPA scores (seed vs children)", "rank test P",
                             "aborted run's seed dev draw (not a replicate)"], checks)
    tb2 = result["terminal-bench-2-1"]["null_executions"]["s_over_n"]
    logger.write_record(
        "selection_luna",
        "The selection replay of mech_selection for GPT-6 Luna: the dev gain each optimizer's own rule reports when every candidate is "
        "identical to the seed, from the seed draw each Luna run started at, against the gain the run reported; on TB2, where all three "
        "optimizers returned the seed, what the dev gate saw and how often selection alone would have kept a candidate.",
        f"Luna Table 1 runs: skillopt/gepa/skilladam/<bench>/{LUNA_RUN} (summary.json `extra`; SkillOpt also skillopt/history.json, since its "
        f"summary leaves baseline_selection_hard empty on Luna), seed native-seed/<bench>/{LUNA_SEED_RUN}. Seed dev executions: every dev "
        f"evaluation of version {SEED_VERSION} under run tags native-20261011/12; all of them (the seed run, every optimizer run, SkillAdam's "
        "subset evaluations) reuse one cached draw per benchmark, so each dev task has one seed execution. Seed test executions: the three "
        "runs Table 1 reads (TB2: 17 tasks; the verifier-timeout task is dropped as in Table 1). The aborted SkillOpt runs (no summary.json; earlier "
        "task config and cache namespace, stopped at the seed's dev evaluation on unknown outcomes) are not pooled.",
        "fitted / moderate: as mech_selection (beta-binomial ML prior over the seed's dev and test executions; dev posteriors from the single "
        "dev execution). A prior-free model on the seed's own executions is degenerate here (one execution per task gives p_v in {0, 1} and "
        "G0 = 0), so the prior-free models are estimated under the null: every dev execution of every version in the benchmark's three "
        "optimizer runs is a seed draw if no candidate differs, and a task's rate is its success share among them (null plug-in) or "
        "Beta(s, n - s) with point masses for unanimous tasks (null Bayesian bootstrap); a candidate that truly differs widens these rates and "
        "raises P(G0 >= g). Model-free: no second seed dev draw exists under this protocol; under the null the seed draw and a run's K "
        "candidate scores are exchangeable, and the rank test averages 1[max(others) - one >= g] over the K + 1 choices of baseline. "
        f"{DRAWS} draws per setting, random seed [20261012, benchmark index, model index, 500]. Where g = 0 (TB2, all three returned "
        "the seed) the P columns give P(G0 >= 1), the chance that selection alone would have kept a candidate.",
        "### Seed inputs\n\n" + t_in + "\n\n### What each run recorded\n\n" + t_runs +
        "\n\n### SkillOpt and GEPA rules under no effect\n\n" + t_rules +
        "\n\n### SkillAdam's gate under no effect (identical patches, outcomes reused as in the runs)\n\n" + t_sa +
        "\n\n### Model-free checks\n\n" + t_chk +
        "\n\n### Power of the replay from the starting draw (fitted model; one candidate truly better, the others identical)\n\n" + t_pw +
        "\n\nTB2 dev tasks over every evaluation of every version (successes / executions): "
        + ", ".join(f"{t} {v}" for t, v in tb2.items()) + ".",
        "python -m aux.selection_replay.luna", extra=luna_notes(result))


def luna_notes(result) -> str:
    """Two facts the paper cites, written from the numbers: SkillOpt on AutoBench (Luna's one test interval above zero) and TB2's gate."""
    r = result["automationbench"]
    o = r["observed"]["SkillOpt"]
    ps = ", ".join(f"{r['models'][m]['rules']['SkillOpt']['P_ge_observed_observed_baseline']:.2f} ({m})" for m in LUNA_MODELS)
    t = result["terminal-bench-2-1"]
    s_over_n = t["null_executions"]["s_over_n"]
    never = [k for k, v in s_over_n.items() if v.startswith("0/")]
    return ("## Notes\n\n"
            f"- SkillOpt on AutoBench: the kept candidate scored {max(o['candidates'])}/30 against the starting draw {o['baseline']}/30 (+{o['gain']} "
            f"task), its candidates {' '.join(map(str, o['candidates']))}; identical candidates report at least +{o['gain']} from that draw with "
            f"probability {ps}, and {r['rank_test']['SkillOpt']:.2f} by the rank test. The replay would flag a candidate that truly solved two more "
            f"dev tasks with probability {r['models']['fitted']['rules']['SkillOpt']['power_observed_baseline']['2']:.2f}.\n"
            f"- TB2: the seed draw {t['observed']['SkillOpt']['baseline']}/9 failed {', '.join(never)}, which every version failed in every "
            "dev execution, and solved the other six; no version scored above it, and under the null models P(G0 >= 1) from that draw is 0, against "
            f"{t['models']['fitted']['rules']['SkillOpt']['P_ge_observed_observed_baseline']:.2f} (SkillOpt) under the fitted model, which "
            "gives these tasks a nonzero rate.\n")


def luna():
    """Check the DeepSeek inputs (unchanged), then compute the Luna replay and write its cache and record."""
    for bench, _ in BENCHES:
        dev_files, test_files = seed_files(bench)
        check_inputs(bench, executions(dev_files), executions(test_files))
    result = {bench: analyse_luna(bench) for bench, _ in BENCHES}
    dataloader.write_json("selection_luna", result)
    record_luna(result)


if __name__ == "__main__":
    luna()
