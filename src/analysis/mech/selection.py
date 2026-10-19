"""mech_selection and mech_devsize: the dev-set gain that success-only selection reports when no candidate helps (paper 1; replaces
appB_nullsim, whose observed gains came from superseded runs, with the Table 1 baseline runs native-20261008-https), and how that gain and the
detectable effect change with the size of the dev panel.

Task model.  Per benchmark, a prior Beta(a, b) over per-task success probability is fitted by maximum likelihood (beta-binomial) to the seed's
distinct executions of the dev and test tasks (a task's executions are its episodes with a verified outcome, deduplicated by their request ids).
Each dev task's probability is the posterior Beta(a + s, b + n - s) from its own distinct dev executions.  Sensitivity: the same with a
moderate-spread prior Beta(4m, 4(1 - m)), m the mean success.  A candidate identical to the seed draws fresh outcomes from these probabilities.

Each optimizer's own rule, with the number of scored candidates of its Table 1 run (summary.json `extra`):
  SkillOpt   best of K candidates on the dev panel, kept only if strictly above the baseline score      K = total_accepts + total_rejects
  GEPA       argmax over the seed and K candidates on the dev panel, ties to the seed                    K = len(val_scores) - 1
  SkillAdam  per iteration, the current skill once and up to `budget` candidate patches on one random subset of `batch` dev tasks (its
             validation minibatch), the first candidate that scores strictly higher is accepted (its gate: primary gain > 0); batch size,
             iterations and budgets from `iterations`.  Its recorded comparisons (the last attempt of each iteration) are counted as better /
             worse / tied, and each accepted gain gets P(gain >= it) for one null comparison.
The baseline (seed) score is either drawn like the candidates (random) or fixed at the run's observed value.  Reported per rule: E[announced gain]
and P(gain >= observed gain).  For an accept-if-better comparison of two single runs on the full dev panel: P(accept | no effect), the SD of the net
difference in tasks, P(accept | true gain of g tasks), and the minimum detectable effect at 80% power of a one-sided 0.05 test (MDE80).
The seed's own dev draws (every full-panel dev evaluation, one row per distinct set of episodes) are listed next to its test avg@3.

Dev size (mech_devsize).  For dev sizes 9, 15, 30, 60, 120 under the same task model: the dev tasks of a draw are a random subset of the
benchmark's dev tasks (size <= its dev panel) or all of them plus tasks drawn from the prior (larger sizes).  For SkillOpt's best of 4 and GEPA's
argmax over 7: E[announced gain] in points with a random baseline and with the baseline at the same mid-quantile of the baseline distribution as
the run's own seed draw; and the MDE80 of a single-run comparison in points.

  .venv/bin/python -m src.analysis.mech.selection compute      -> assets/cache/analysis/mech/selection.json, data/mech/mech_selection.md, data/mech/mech_devsize.md
  python3 -m src.analysis.mech.selection draw   -> fig/mech/mech_selection.{pdf,svg} (full width), fig/mech/mech_devsize.{pdf,svg} (one column)
"""
import glob
import json
import math
import sys
from collections import defaultdict

import numpy as np

from src.utils import fig, logger, dataloader
EXECUTION = dataloader.H / "assets/artifacts/execution/deepseek-flash"
RUN = "native-20261008-https"
BENCHES = [("terminal-bench-2-1", "TB2"), ("tau3-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench")]
SEED_VERSION = "4fd5ca441c55"
SEED_TEST_RUNS = ("native-20261003", "rep-1", "rep-2")
TEST_AVG3 = {"tau3-bench": 76.4, "automationbench": 45.0, "terminal-bench-2-1": 75.9}   # Table 1, Seed skill row (data/mech/mech_table1.md)
DRAWS = 100_000
DEVSIZE_DRAWS = 40_000
DEV_SIZES = (9, 15, 30, 60, 120)
DEVSIZE_RULES = (("SkillOpt", 4), ("GEPA", 7))
MODERATE = 4.0           # concentration of the sensitivity prior
GAINS = (1, 2, 3)        # true gains (tasks) for P(accept | gain)
PRIORS = ("fitted", "moderate")
RNG = np.random.default_rng(20261012)   # re-seeded per benchmark, prior and dev size, so one cell's numbers never depend on another's


def reseed(*key):
    """Re-seed the module's generator from a fixed key (benchmark index, prior index, ...)."""
    global RNG
    RNG = np.random.default_rng([20261012, *key])


# ---- the seed's executions ---------------------------------------------------------------------------------------------------------

def signature(e) -> tuple:
    """The request ids of an episode: the same cached episode in several evaluation files has the same signature."""
    return tuple(sorted(r["request_id"] for r in e["requests"]))


def verified(e) -> bool:
    """An episode with requests and a verified outcome."""
    return bool(e["requests"]) and bool(e.get("outcome_valid")) and e.get("success") is not None


def executions(files) -> dict:
    """{task: [0/1 per distinct execution]} over the evaluation files."""
    seen = defaultdict(dict)
    for path in files:
        for e in json.load(open(path))["episodes"]:
            if verified(e):
                seen[e["task_id"]][signature(e)] = int(bool(e["success"]))
    return {task: list(runs.values()) for task, runs in seen.items()}


def seed_files(bench) -> tuple[list, list]:
    """(dev evaluation files, test evaluation files) of the seed skill on one benchmark."""
    dev_files = glob.glob(str(EXECUTION / f"*/{bench}/*/results/evaluations/dev/{SEED_VERSION}-*/evaluation.json"))
    test_files = []
    for run in SEED_TEST_RUNS:
        test_files += glob.glob(str(EXECUTION / f"native-seed/{bench}/{run}/results/evaluations/test/*/evaluation.json"))
    return dev_files, test_files


def dev_draws(dev_files, n_dev) -> list[dict]:
    """The seed's distinct full-panel dev draws: one row per distinct set of verified episodes covering the dev panel, with its success and
    the runs that used it (an evaluation of a 10-task validation subset is not a full-panel draw)."""
    draws = defaultdict(lambda: {"runs": []})
    for path in dev_files:
        episodes = json.load(open(path))["episodes"]
        if len(episodes) != n_dev:
            continue
        ok = [e for e in episodes if verified(e)]
        key = frozenset((e["task_id"], signature(e)) for e in ok)
        draw = draws[key]
        draw.update(solved=sum(bool(e["success"]) for e in ok), verified=len(ok))
        draw["runs"].append(path.split("deepseek-flash/")[1].split("/results")[0])
    return sorted(draws.values(), key=lambda d: d["solved"] / d["verified"])


def fit_prior(tasks: list[tuple[int, int]]) -> tuple[float, float]:
    """(a, b) of the Beta prior maximizing the beta-binomial likelihood of [(successes, executions)] (grid over mean and concentration)."""
    def loglik(a, b):
        return sum(math.lgamma(k + a) + math.lgamma(n - k + b) - math.lgamma(n + a + b) - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b)
                   for k, n in tasks)
    best = None
    for m in np.linspace(0.01, 0.99, 99):
        for log_c in np.linspace(-2.0, 7.0, 91):
            c = math.exp(log_c)
            value = loglik(m * c, (1 - m) * c)
            if best is None or value > best[0]:
                best = (value, m * c, (1 - m) * c)
    return best[1], best[2]


def posterior_mean(runs: dict, a: float, b: float) -> float:
    """Expected seed success (%) over the tasks under their posteriors Beta(a + s, b + n - s)."""
    return 100 * float(np.mean([(a + sum(v)) / (a + b + len(v)) for v in runs.values()]))


# ---- the baseline runs' selection records ------------------------------------------------------------------------------------------

def batch_size(scores: list[float]) -> int:
    """The smallest batch size n (<= 20) for which every recorded score times n is a whole number of tasks."""
    return next(n for n in range(1, 21) if all(abs(s * n - round(s * n)) < 1e-6 for s in scores))


def skilladam_observed(scored: list[dict]) -> dict:
    """SkillAdam's recorded iterations: batch size, budgets, accepted count, and the last attempt's difference (candidate - current, tasks)."""
    n = batch_size([i["baseline"] for i in scored] + [i["candidate"] for i in scored])
    diffs = [round((i["candidate"] - i["baseline"]) * n) for i in scored]
    return {"iterations": len(scored), "batch": n, "budgets": [i["budget"] for i in scored], "attempts": sum(i["attempts"] for i in scored),
            "accepted": sum(1 for i in scored if i["accepted"]), "accepted_gains": [d for d, i in zip(diffs, scored) if i["accepted"]],
            "reported_gain_points": 100 * sum(d for d, i in zip(diffs, scored) if i["accepted"]) / n,
            "better": sum(d > 0 for d in diffs), "worse": sum(d < 0 for d in diffs), "tied": sum(d == 0 for d in diffs)}


def observed(bench, n_dev) -> dict:
    """What the Table 1 runs recorded: per optimizer the candidates scored, the baseline score and the announced gain in dev tasks."""
    so = json.load(open(EXECUTION / f"skillopt/{bench}/{RUN}/summary.json"))["extra"]
    ge = json.load(open(EXECUTION / f"gepa/{bench}/{RUN}/summary.json"))["extra"]
    sa = json.load(open(EXECUTION / f"skilladam/{bench}/{RUN}/summary.json"))["extra"]
    scored = [i for i in sa["iterations"] if isinstance(i.get("baseline"), (int, float)) and isinstance(i.get("candidate"), (int, float))]
    vals = ge["val_scores"]
    return {
        "SkillOpt": {"K": so["total_accepts"] + so["total_rejects"], "baseline": round(so["baseline_selection_hard"] * n_dev),
                     "gain": round((so["best_selection_hard"] - so["baseline_selection_hard"]) * n_dev)},
        "GEPA": {"K": len(vals) - 1, "baseline": round(vals[0] * n_dev), "gain": round((max(vals) - vals[0]) * n_dev)},
        "SkillAdam": skilladam_observed(scored),
    }


# ---- the simulation ----------------------------------------------------------------------------------------------------------------

def outcomes(p):
    """Solved-task counts of one run per draw: p is (draws, tasks)."""
    return (RNG.random(p.shape) < p).sum(axis=1)


def best_of(p, k, baseline):
    """The gain the best of k candidates announces over the baseline score (0 when none is strictly above it)."""
    best = (RNG.random((p.shape[0], k, p.shape[1])) < p[:, None, :]).sum(axis=2).max(axis=1)
    return np.maximum(best - baseline, 0)


def raised(p, gain):
    """Probabilities raised on the least-solved tasks so that the expected number of solved tasks grows by `gain`."""
    q = p.copy()
    order = np.argsort(p, axis=1)
    rows = np.arange(p.shape[0])
    left = np.full(p.shape[0], float(gain))
    for column in range(p.shape[1]):
        index = order[:, column]
        add = np.minimum(1.0 - q[rows, index], left)
        q[rows, index] += add
        left -= add
    return q


def paired(p, q):
    """Net difference (candidate - parent, tasks) of two single runs on the same tasks: parent from p, candidate from q."""
    return outcomes(q) - outcomes(p)


def mde80(p, step=0.25) -> tuple[int, float]:
    """(critical net difference, MDE80 in tasks): the one-sided 0.05 critical value under no effect, and the smallest true gain (in `step`
    tasks) detected at it with probability 0.8 (nan when no gain up to the panel size is)."""
    null = paired(p, p)
    critical = next(t for t in range(0, p.shape[1] + 1) if (null >= t).mean() <= 0.05)
    for gain in np.arange(step, p.shape[1] + step, step):
        if (paired(p, raised(p, gain)) >= critical).mean() >= 0.8:
            return critical, float(gain)
    return critical, float("nan")


def skilladam_accepts(p_dev, batch, budgets):
    """(draws,) number of iterations that accept a null patch: per iteration a random subset of `batch` dev tasks, the current skill once and
    up to `budget` candidates on it, the first strictly better candidate accepted."""
    draws, n_dev = p_dev.shape
    accepted = np.zeros(draws, dtype=int)
    for budget in budgets:
        pick = np.argsort(RNG.random((draws, n_dev)), axis=1)[:, :batch]
        p = np.take_along_axis(p_dev, pick, axis=1)
        current = outcomes(p)
        candidates = (RNG.random((draws, budget, batch)) < p[:, None, :]).sum(axis=2)
        accepted += (candidates > current[:, None]).any(axis=1)
    return accepted


def skilladam_reported_gain(p_dev, batch, budgets, rng):
    """((draws,) dev gain in points, (draws,) accepted iterations) that SkillAdam's rule reports for patches identical to the seed, summed
    over its iterations.

    As in the runs, a skill-task pair is run once and reused: the current skill's outcome on a task is drawn the first time the task is
    drawn and kept, and an accepted patch becomes the current skill with its own outcomes on the tasks it was scored on (its other tasks
    unrun).  Per iteration: a random subset of `batch` dev tasks, the current skill's (cached) score on it, up to `budget` patches with fresh
    outcomes, and the gain of the first strictly better patch.  Uses its own generator so that the module's stream, and every other number,
    stays as it was."""
    draws, n_dev = p_dev.shape
    rows = np.arange(draws)[:, None]
    current = np.full((draws, n_dev), -1, dtype=int)   # -1 = the current skill has not run this task
    total = np.zeros(draws)
    accepts = np.zeros(draws, dtype=int)
    for budget in budgets:
        pick = np.argsort(rng.random((draws, n_dev)), axis=1)[:, :batch]
        p = p_dev[rows, pick]
        # the current skill runs the tasks of this subset it has not run yet
        seen = current[rows, pick]
        fresh = (rng.random(p.shape) < p).astype(int)
        seen = np.where(seen < 0, fresh, seen)
        current[rows, pick] = seen
        base = seen.sum(axis=1)
        patches = (rng.random((draws, budget, batch)) < p[:, None, :]).astype(int)
        better = patches.sum(axis=2) > base[:, None]
        accepted = better.any(axis=1)
        first = better.argmax(axis=1)
        winner = patches[np.arange(draws), first]
        total += np.where(accepted, 100 * (winner.sum(axis=1) - base) / batch, 0)
        accepts += accepted
        # an accepted patch becomes the current skill: its outcomes on this subset, every other task unrun
        current[accepted] = -1
        current[rows[accepted], pick[accepted]] = winner[accepted]
    return total, accepts


def skilladam_null(p_dev, sa, rng) -> dict:
    """SkillAdam's summed reported gain under no effect: its mean (points) and the probability that it reaches the run's own."""
    gain, accepts = skilladam_reported_gain(p_dev, sa["batch"], sa["budgets"], rng)
    return {"E_reported_gain_points": float(gain.mean()),
            "P_reported_gain_ge_observed": float((gain >= sa["reported_gain_points"] - 1e-9).mean()),
            "E_accepts_cached": float(accepts.mean())}


def batch_gain_tail(p_dev, batch, gain):
    """P(candidate - current >= gain) for one null comparison on a random subset of `batch` dev tasks."""
    pick = np.argsort(RNG.random(p_dev.shape), axis=1)[:, :batch]
    p = np.take_along_axis(p_dev, pick, axis=1)
    return float((paired(p, p) >= gain).mean())


def mid_quantile(sample, value) -> float:
    """P(X < value) + P(X = value) / 2 of an integer sample."""
    return float((sample < value).mean() + (sample == value).mean() / 2)


def task_probabilities(dev, prior, size, draws):
    """(draws, size) probabilities of a dev panel of `size` tasks: a random subset of the benchmark's dev tasks (posteriors) when size is at
    most the panel, else all of them plus size - panel tasks drawn from the prior."""
    pa, pb = prior
    tasks = sorted(dev)
    post_a = np.array([pa + sum(dev[t]) for t in tasks])
    post_b = np.array([pb + len(dev[t]) - sum(dev[t]) for t in tasks])
    p = RNG.beta(post_a, post_b, size=(draws, len(tasks)))
    if size <= len(tasks):
        pick = np.argsort(RNG.random((draws, len(tasks))), axis=1)[:, :size]
        return np.take_along_axis(p, pick, axis=1)
    return np.concatenate([p, RNG.beta(pa, pb, size=(draws, size - len(tasks)))], axis=1)


# ---- per benchmark -----------------------------------------------------------------------------------------------------------------

def analyse(bench) -> dict:
    """Every number of one benchmark under the fitted and the moderate prior: the seed's dev draws, the rules at the real dev panel, the
    accept-if-better comparison, and the dev-size sensitivity."""
    dev_files, test_files = seed_files(bench)
    dev, test = executions(dev_files), executions(test_files)
    n_dev = len(dev)
    pooled = [(sum(v), len(v)) for v in list(dev.values()) + list(test.values())]
    a, b = fit_prior(pooled)
    mean = sum(k for k, _ in pooled) / sum(n for _, n in pooled)
    obs = observed(bench, n_dev)
    out = {"dev_tasks": n_dev, "dev_executions_per_task": sorted({len(v) for v in dev.values()}), "test_tasks": len(test),
           "prior_fitted": [a, b], "mean_success": mean, "observed": obs, "priors": {}, "devsize": {},
           "seed_dev_draws": dev_draws(dev_files, n_dev), "test_avg3": TEST_AVG3[bench],
           "expected_seed_success": {"dev": posterior_mean(dev, a, b), "test": posterior_mean(test, a, b), "prior_mean": 100 * a / (a + b)}}
    index = [x for x, _ in BENCHES].index(bench)
    for j, name in enumerate(PRIORS):
        prior = (a, b) if name == "fitted" else (MODERATE * mean, MODERATE * (1 - mean))
        reseed(index, j)
        p = task_probabilities(dev, prior, n_dev, DRAWS)
        rules = {}
        for rule in ("SkillOpt", "GEPA"):
            k = obs[rule]["K"]
            gain_random = best_of(p, k, outcomes(p))
            gain_fixed = best_of(p, k, obs[rule]["baseline"])
            rules[rule] = {"K": k, "E_gain_random_baseline": float(gain_random.mean()), "E_gain_observed_baseline": float(gain_fixed.mean()),
                           "P_ge_observed_random_baseline": float((gain_random >= obs[rule]["gain"]).mean()),
                           "P_ge_observed_observed_baseline": float((gain_fixed >= obs[rule]["gain"]).mean()),
                           "hist_random_baseline": np.bincount(gain_random, minlength=n_dev + 1).tolist(),
                           "hist_observed_baseline": np.bincount(gain_fixed, minlength=n_dev + 1).tolist()}
        sa = obs["SkillAdam"]
        accepts = skilladam_accepts(p, sa["batch"], sa["budgets"])
        rules["SkillAdam"] = {"E_accepts": float(accepts.mean()), "P_accepts_ge_observed": float((accepts >= sa["accepted"]).mean()),
                              "P_better_one_comparison": batch_gain_tail(p, sa["batch"], 1),
                              "P_gain_ge_accepted": {str(g): batch_gain_tail(p, sa["batch"], g) for g in sa["accepted_gains"]},
                              **skilladam_null(p, sa, np.random.default_rng([20261012, index, j, 7]))}
        null = paired(p, p)
        critical, mde = mde80(p)
        baseline_quantile = mid_quantile(outcomes(p), obs["SkillOpt"]["baseline"])
        out["priors"][name] = {"prior": list(prior), "rules": rules, "seed_draw_mid_quantile": baseline_quantile, "accept_if_better": {
            "P_accept_no_effect": float((null > 0).mean()), "SD_net_tasks": float(null.std()),
            "P_accept_true_gain": {str(g): float((paired(p, raised(p, g)) > 0).mean()) for g in GAINS},
            "critical_net_tasks": critical, "MDE80_tasks": mde, "MDE80_points": 100 * mde / n_dev}}
        out["devsize"][name] = devsize(dev, prior, index, j, baseline_quantile)
    return out


def devsize(dev, prior, index, j, baseline_quantile) -> dict:
    """Per dev size: E[announced gain] (points) of SkillOpt's best of 4 and GEPA's argmax over 7 with a random baseline and with the baseline
    at the run's own seed-draw mid-quantile, and the MDE80 of a single-run comparison (points)."""
    rows = {}
    for s, size in enumerate(DEV_SIZES):
        reseed(index, j, 100 + s)
        p = task_probabilities(dev, prior, size, DEVSIZE_DRAWS)
        base = outcomes(p)
        at_quantile = float(np.quantile(base, baseline_quantile, method="nearest"))
        row = {"baseline_at_quantile": at_quantile}
        for rule, k in DEVSIZE_RULES:
            row[rule] = {"E_gain_random_points": 100 * float(best_of(p, k, base).mean()) / size,
                         "E_gain_quantile_points": 100 * float(best_of(p, k, at_quantile).mean()) / size}
        critical, mde = mde80(p, step=0.5 if size > 30 else 0.25)
        row["MDE80_points"] = 100 * mde / size
        row["critical_net_tasks"] = critical
        rows[str(size)] = row
    return rows


# ---- outputs -----------------------------------------------------------------------------------------------------------------------

def records_from_cache():
    """Rewrite mech_selection.md and mech_devsize.md from selection.json."""
    result = json.loads((dataloader.CACHE / "selection.json").read_text())
    record_selection(result)
    record_devsize(result)


def record_selection(result):
    """The audit record data/mech/mech_selection.md."""
    rows, rows2, draws = [], [], []
    for bench, _ in BENCHES:
        r = result[bench]
        for prior in PRIORS:
            g = r["priors"][prior]
            for rule in ("SkillOpt", "GEPA"):
                x, o = g["rules"][rule], r["observed"][rule]
                rows.append([bench, prior, rule, x["K"], f"{o['baseline']}/{r['dev_tasks']}", f"+{o['gain']}",
                             f"{x['E_gain_random_baseline']:.2f}", f"{x['E_gain_observed_baseline']:.2f}",
                             f"{x['P_ge_observed_random_baseline']:.3f}", f"{x['P_ge_observed_observed_baseline']:.3f}"])
            sa, o = g["rules"]["SkillAdam"], r["observed"]["SkillAdam"]
            gains = ", ".join(f"+{k} (P {sa['P_gain_ge_accepted'][str(k)]:.3f})" for k in o["accepted_gains"])
            rows.append([bench, prior, "SkillAdam", f"{o['iterations']} iterations on {o['batch']}-task dev subsets, {o['attempts']} patches",
                         f"better/worse/tied {o['better']}/{o['worse']}/{o['tied']}", f"{o['accepted']} accepted: {gains}",
                         f"E[accepts] {sa['E_accepts']:.2f} fresh outcomes, {sa['E_accepts_cached']:.2f} with reuse",
                         f"P(better) per comparison {sa['P_better_one_comparison']:.3f}",
                         f"P(accepts >= observed) {sa['P_accepts_ge_observed']:.3f}",
                         f"summed gain {o['reported_gain_points']:.1f} pts; with reuse E {sa['E_reported_gain_points']:.1f} pts, "
                         f"P(>= observed) {sa['P_reported_gain_ge_observed']:.2f}"])
            ab = g["accept_if_better"]
            rows2.append([bench, prior, r["dev_tasks"], f"{ab['P_accept_no_effect']:.3f}", f"{ab['SD_net_tasks']:.2f}",
                          " / ".join(f"{ab['P_accept_true_gain'][str(k)]:.3f}" for k in GAINS), ab["critical_net_tasks"],
                          f"{ab['MDE80_tasks']:.2f} ({ab['MDE80_points']:.1f} pts)"])
        e = r["expected_seed_success"]
        qs = [100 * d["solved"] / d["verified"] for d in r["seed_dev_draws"]]
        for d in r["seed_dev_draws"]:
            draws.append([bench, f"{d['solved']}/{d['verified']}", f"{100 * d['solved'] / d['verified']:.1f}", len(d["runs"]),
                          ", ".join(d["runs"][:3]) + (" ..." if len(d["runs"]) > 3 else "")])
        draws.append([bench, "summary", f"mean {np.mean(qs):.1f}, min {min(qs):.1f}, max {max(qs):.1f}",
                      f"run's start draw {r['observed']['SkillOpt']['baseline']}/{r['dev_tasks']} (mid-quantile {r['priors']['fitted']['seed_draw_mid_quantile']:.2f})",
                      f"expected seed success under the fitted posteriors: dev {e['dev']:.1f}, test {e['test']:.1f}; prior mean {e['prior_mean']:.1f}; test avg@3 {r['test_avg3']}"])
    t1 = logger.md_table(["bench", "prior", "rule", "K / iterations", "observed baseline", "observed gain", "E[gain], random baseline",
                     "E[gain], observed baseline", "P(gain >= obs), random", "P(gain >= obs), observed"], rows)
    t2 = logger.md_table(["bench", "prior", "dev tasks", "P(accept | no effect)", "SD of net difference (tasks)",
                     "P(accept | true gain 1 / 2 / 3 tasks)", "critical net difference (one-sided 0.05)", "MDE80"], rows2)
    t3 = logger.md_table(["bench", "seed dev draw (solved/verified)", "success %", "runs using it", "runs (first three)"], draws)
    priors = logger.md_table(["bench", "dev tasks", "seed executions per dev task", "test tasks", "fitted prior (a, b)", "mean success"],
                        [[b, result[b]["dev_tasks"], result[b]["dev_executions_per_task"], result[b]["test_tasks"],
                          f"({result[b]['prior_fitted'][0]:.2f}, {result[b]['prior_fitted'][1]:.2f})", f"{result[b]['mean_success']:.3f}"]
                         for b, _ in BENCHES])
    logger.write_record(
        "selection",
        "The gain each optimizer's own selection rule announces on the dev panel when every candidate is identical to the seed (no effect), "
        "against the gain its Table 1 run announced; how an accept-if-better comparison of two single runs behaves on the dev panel; and the "
        "seed's own dev draws next to its test success.",
        f"Seed: its distinct executions of the dev tasks (every dev evaluation of version {SEED_VERSION} under assets/artifacts/execution/"
        f"deepseek-flash, deduplicated by request ids) and of the test tasks (native-seed runs {', '.join(SEED_TEST_RUNS)}). Optimizers: "
        f"summary.json `extra` of skillopt/gepa/skilladam/<bench>/{RUN} (SkillOpt baseline_selection_hard, best_selection_hard, total_accepts, "
        "total_rejects; GEPA val_scores (index 0 = the seed); SkillAdam iterations with numeric baseline and candidate, scored on dev subsets).",
        f"Prior fitted by beta-binomial maximum likelihood to the seed's dev and test executions; dev-task posteriors from the dev executions; "
        f"moderate prior Beta(4m, 4(1-m)). {DRAWS} draws per setting; random seed [20261012, benchmark index, prior index]. A true gain of g tasks "
        "raises the least-solved tasks' probabilities until the expected number of solved tasks grows by g. Seed dev draws: full-panel dev "
        "evaluations of the seed, one row per distinct set of verified episodes (sorted by success).",
        "### Seed task model\n\n" + priors + "\n\n### The seed's dev draws and its test success\n\n" + t3 +
        "\n\n### Selection rules under no effect\n\n" + t1 + "\n\n### Accept-if-better on the dev panel (two single runs)\n\n" + t2,
        ".venv/bin/python -m src.analysis.mech.selection compute\npython3 -m src.analysis.mech.selection draw")


def record_devsize(result):
    """The audit record data/mech/mech_devsize.md."""
    rows = []
    for bench, _ in BENCHES:
        for prior in PRIORS:
            q = result[bench]["priors"][prior]["seed_draw_mid_quantile"]
            for size in DEV_SIZES:
                r = result[bench]["devsize"][prior][str(size)]
                rows.append([bench, prior, size, f"{q:.2f} -> {r['baseline_at_quantile']:.0f}/{size}",
                             f"{r['SkillOpt']['E_gain_random_points']:.1f}", f"{r['SkillOpt']['E_gain_quantile_points']:.1f}",
                             f"{r['GEPA']['E_gain_random_points']:.1f}", f"{r['GEPA']['E_gain_quantile_points']:.1f}",
                             f"{r['MDE80_points']:.1f}", r["critical_net_tasks"]])
    logger.write_record(
        "devsize",
        "How the gain a success-only selector announces under no effect, and the smallest effect a single-run dev comparison detects, change "
        "with the size of the dev panel (SkillOpt's best of 4, GEPA's argmax over 7), in percentage points of the dev panel.",
        "The task model of mech_selection (the same seed executions and fitted / moderate priors).",
        f"Per dev size, {DEVSIZE_DRAWS} draws: the dev tasks are a random subset of the benchmark's dev tasks (posteriors) when the size is at most "
        "its dev panel, else all of them plus tasks drawn from the prior. Random baseline: the seed's score drawn like the candidates. Quantile "
        "baseline: the baseline fixed at the same mid-quantile of the seed-score distribution as the run's own seed draw at the real dev size "
        "(the quantile and the resulting score are listed). MDE80: the smallest true gain detected with probability 0.8 by a one-sided 0.05 "
        "comparison of two single runs. Random seed [20261012, benchmark index, prior index, 100 + size index].",
        logger.md_table(["bench", "prior", "dev size", "seed-draw quantile -> baseline", "SkillOpt E[gain] random (pts)", "SkillOpt E[gain] at quantile (pts)",
                    "GEPA E[gain] random (pts)", "GEPA E[gain] at quantile (pts)", "MDE80 (pts)", "critical net difference (tasks)"], rows),
        ".venv/bin/python -m src.analysis.mech.selection compute\npython3 -m src.analysis.mech.selection draw")


def compute():
    """Simulate every benchmark and write the cache and both audit records."""
    result = {bench: analyse(bench) for bench, _ in BENCHES}
    dataloader.write_json("selection", result)
    record_selection(result)
    record_devsize(result)


# ---- figures -----------------------------------------------------------------------------------------------------------------------

def style():
    """matplotlib with TrueType fonts in the pdf (the ACL check rejects Type 3)."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    return plt


def draw_selection(result):
    """mech_selection, full width (6.3 in): two rows of three benchmark panels, the distribution of the announced dev gain under no effect for
    SkillOpt's rule (filled bars) and GEPA's rule (hatched outline bars), fitted prior; (a) baseline drawn like the candidates, (b) baseline at
    the run's own seed draw; the gain each Table 1 run announced as a thick dark line with a marker on top (solid, triangle: SkillOpt; dashed,
    diamond: GEPA).  Readable in grayscale; every text >= 7 pt."""
    plt = style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from src.analysis.mechanism import appb
    look = {"SkillOpt": {"offset": -0.2, "face": appb.OC["SkillOpt"], "hatch": None, "line": "-", "marker": "v"},
            "GEPA": {"offset": 0.2, "face": "white", "hatch": "////", "line": "--", "marker": "D"}}
    rows = (("hist_random_baseline", "(a) random seed draw as baseline"), ("hist_observed_baseline", "(b) baseline = the run's own seed draw"))
    fig, axs = plt.subplots(2, 3, figsize=(6.3, 4.3))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.1, top=0.83, wspace=0.3, hspace=0.85)
    for row, (key, title) in enumerate(rows):
        for ax, (bench, disp) in zip(axs[row], BENCHES):
            r = result[bench]
            g = r["priors"]["fitted"]["rules"]
            hists = {rule: np.array(g[rule][key], dtype=float) / sum(g[rule][key]) for rule in look}
            top = max(max(int(np.nonzero(h > 0.002)[0].max()) for h in hists.values()), max(r["observed"][k]["gain"] for k in look)) + 1
            ymax = max(h[:top + 1].max() for h in hists.values())
            for rule, lk in look.items():
                xs = np.arange(top + 1) + lk["offset"]
                ax.bar(xs, hists[rule][:top + 1], width=0.38, color=lk["face"], edgecolor=appb.OC[rule], linewidth=0.8, hatch=lk["hatch"])
                # the gain this optimizer's Table 1 run announced: a thick dark line with a marker above the bars
                x_obs = r["observed"][rule]["gain"] + lk["offset"]
                ax.plot([x_obs, x_obs], [0, ymax * 1.08], color="0.1", lw=1.8, ls=lk["line"], zorder=5)
                ax.plot([x_obs], [ymax * 1.12], marker=lk["marker"], color="0.1", markersize=4.5, zorder=6, clip_on=False)
            label = f"{disp}  (K = {g['SkillOpt']['K']} / {g['GEPA']['K']})"
            if row == 1:
                label = f"{disp}  (baseline {r['observed']['SkillOpt']['baseline']}/{r['dev_tasks']})"
            ax.set_title(label, fontsize=8, pad=8)
            ax.set_ylim(0, ymax * 1.18)
            ax.set_xlabel("announced dev gain (tasks)", fontsize=7.5)
            ax.set_ylabel("probability", fontsize=7.5)
            ax.tick_params(labelsize=7)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # gains are whole tasks
            ax.spines[["top", "right"]].set_visible(False)
        axs[row][1].annotate(title, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 21), textcoords="offset points",
                             ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    handles = [Patch(facecolor=look["SkillOpt"]["face"], edgecolor=appb.OC["SkillOpt"], label="SkillOpt rule, no effect"),
               Patch(facecolor="white", edgecolor=appb.OC["GEPA"], hatch="////", label="GEPA rule, no effect"),
               Line2D([], [], color="0.1", lw=1.8, ls="-", marker="v", markersize=4.5, label="SkillOpt run's gain"),
               Line2D([], [], color="0.1", lw=1.8, ls="--", marker="D", markersize=4, label="GEPA run's gain")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.53, 1.0))
    fig.save_figure(fig, "selection")


def draw_devsize(result):
    """mech_devsize, one column (3.25 in): two rows, (a) the announced dev gain under no effect (points, random baseline) and (b) the MDE80 of a
    single-run comparison (points), against the dev size (log axis), per benchmark (marker) for the fitted prior (solid) and the moderate prior
    (dashed); in (a) SkillOpt's best of 4 dark, GEPA's argmax over 7 light.  A missing MDE80 point: no effect up to the panel size is detected
    with probability 0.8.  The dev size of the Table 1 runs (30) is marked.  Every text >= 7 pt."""
    plt = style()
    from matplotlib.lines import Line2D
    benches = (("terminal-bench-2-1", "TB2", "s"), ("tau3-bench", r"$\tau^3$", "o"), ("automationbench", "AutoB", "^"))
    shades = {"SkillOpt": "0.15", "GEPA": "0.6"}
    fig, axs = plt.subplots(2, 1, figsize=(3.25, 3.7), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.12, top=0.98, hspace=0.12)
    sizes = list(DEV_SIZES)
    for bench, _, marker in benches:
        for prior, ls in (("fitted", "-"), ("moderate", "--")):
            rows = [result[bench]["devsize"][prior][str(s)] for s in sizes]
            for rule, _ in DEVSIZE_RULES:
                axs[0].plot(sizes, [r[rule]["E_gain_random_points"] for r in rows], color=shades[rule], ls=ls, lw=1.0, marker=marker, markersize=3.5)
            axs[1].plot(sizes, [r["MDE80_points"] for r in rows], color="0.15", ls=ls, lw=1.0, marker=marker, markersize=3.5)
    for ax in axs:
        ax.axvline(30, color="0.75", lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.minorticks_off()
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("(a) null-announced\ngain (points)", fontsize=7.5)
    axs[1].set_ylabel("(b) MDE80 (points)", fontsize=7.5)
    axs[1].set_xlabel("dev panel size (tasks); 30 = Table 1 runs", fontsize=7.5)
    rule_handles = [Line2D([], [], color=shades["SkillOpt"], lw=1.2, label="SkillOpt, best of 4"),
                    Line2D([], [], color=shades["GEPA"], lw=1.2, label="GEPA, argmax over 7"),
                    Line2D([], [], color="0.3", lw=1.0, ls="-", label="fitted prior"),
                    Line2D([], [], color="0.3", lw=1.0, ls="--", label="moderate prior")]
    axs[0].legend(handles=rule_handles, loc="upper right", fontsize=7, frameon=False, handlelength=1.8, borderaxespad=0.1)
    bench_handles = [Line2D([], [], color="0.3", lw=0, marker=m, markersize=4, label=d) for _, d, m in benches]
    axs[1].legend(handles=bench_handles, loc="upper right", fontsize=7, frameon=False, ncol=3, columnspacing=0.8, handletextpad=0.3,
                  borderaxespad=0.1)
    fig.save_figure(fig, "devsize")


def draw():
    """Both figures from the cache."""
    result = json.load(open(dataloader.CACHE / "selection.json"))
    draw_selection(result)
    draw_devsize(result)


if __name__ == "__main__":
    # records: rewrite the two audit records from the cache, without recomputing
    {"compute": compute, "draw": draw, "records": records_from_cache}[sys.argv[1]]()
