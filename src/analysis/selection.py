"""The dev gain each optimizer's selection rule reports when no candidate differs from the seed (DeepSeek, Table 1 runs), and Figure 4.

Task model: a Beta prior fitted by beta-binomial maximum likelihood to the seed's distinct dev and test executions (sensitivity: a
moderate prior Beta(4m, 4(1 - m))); each dev task draws from its posterior. SkillOpt keeps the best of K candidates if strictly above the
baseline, GEPA takes the argmax over the seed and K candidates, SkillAdam accepts the first of up to `budget` patches that beats the current
skill on a random dev subset.

  python -m src.analysis.selection compute   -> selection.json, dev2test.json (panel a), mech_selection.md
  python -m src.analysis.selection draw      -> mech_dev2test4.{pdf,svg} (Figure 4) and dev2test.json (panel b)
"""
import glob, json, math, re, sys
from collections import defaultdict

import numpy as np

from src.analysis.common import rep_files, rows_of
from src.utils import logger
from src.utils.dataloader import CACHE, DS, out_dir, write_json
from src.utils.fig import save_figure

EXECUTION = DS
RUN = "native-20261008-https"
BENCHES = [("terminal-bench-2-1", "TB2"), ("tau3-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench")]
SEED_VERSION = "4fd5ca441c55"
SEED_TEST_RUNS = ("native-20261003", "rep-1", "rep-2")
DRAWS = 100_000
MODERATE = 4.0
GAINS = (1, 2, 3)
POWER_GAINS = (1, 2, 3)
PRIORS = ("fitted", "moderate")
RNG = np.random.default_rng(20261012)
LAST_RUN_DATE = "20261015"  # later seed runs use another dev panel or the upgraded model
EXCLUDED_RUN_TAGS = ("lodo", "-1017")
# (dev tasks, distinct dev executions, distinct test executions) of the seed inputs
PAPER_INPUTS = {"terminal-bench-2-1": (9, 18, 54), "tau3-bench": (30, 90, 225), "automationbench": (30, 89, 360)}
RULE = {"skillopt": "SkillOpt", "gepa": "GEPA", "skilladam": "SkillAdam"}


def reseed(*key):
    global RNG
    RNG = np.random.default_rng([20261012, *key])


def signature(e):
    """Request ids of an episode: one cached episode has one signature in every file."""
    return tuple(sorted(r["request_id"] for r in e["requests"]))


def verified(e):
    return bool(e["requests"]) and bool(e.get("outcome_valid")) and e.get("success") is not None


def executions(files):
    """{task: [0/1 per distinct execution]}."""
    seen = defaultdict(dict)
    for path in files:
        for e in json.load(open(path))["episodes"]:
            if verified(e):
                seen[e["task_id"]][signature(e)] = int(bool(e["success"]))
    return {task: list(runs.values()) for task, runs in seen.items()}


def paper_input(path):
    run = path.split("deepseek-flash/")[1].split("/results")[0].split("/")[-1]
    dated = re.search(r"(2026\d{4})", run)
    return not any(tag in run for tag in EXCLUDED_RUN_TAGS) and (dated is None or dated.group(1) < LAST_RUN_DATE)


def seed_files(bench):
    """(dev evaluation files, test evaluation files) of the seed."""
    dev = sorted(f for f in glob.glob(str(EXECUTION / f"*/{bench}/*/results/evaluations/dev/{SEED_VERSION}-*/evaluation.json")) if paper_input(f))
    test = [f for run in SEED_TEST_RUNS for f in glob.glob(str(EXECUTION / f"native-seed/{bench}/{run}/results/evaluations/test/*/evaluation.json"))]
    return dev, test


def check_inputs(bench, dev, test):
    found = (len(dev), sum(map(len, dev.values())), sum(map(len, test.values())))
    assert found == PAPER_INPUTS[bench], f"{bench}: seed executions {found} differ from {PAPER_INPUTS[bench]}; check seed_files"


def seed_test_avg3(bench):
    """Seed test avg@3 (percent, one decimal) over its three test runs."""
    reps = rep_files([EXECUTION / "native-seed" / bench / run for run in SEED_TEST_RUNS], SEED_VERSION)
    rates = []
    for _, f in sorted(reps.items()):
        known = [r for r in rows_of(f).values() if r["success"] is not None]
        rates.append(sum(r["success"] is True for r in known) / len(known))
    return round(100 * sum(rates) / len(rates), 1)


def dev_draws(dev_files, n_dev):
    """The seed's distinct full-panel dev draws, sorted by success."""
    draws = defaultdict(lambda: {"runs": []})
    for path in dev_files:
        episodes = json.load(open(path))["episodes"]
        if len(episodes) != n_dev:
            continue
        ok = [e for e in episodes if verified(e)]
        draw = draws[frozenset((e["task_id"], signature(e)) for e in ok)]
        draw.update(solved=sum(bool(e["success"]) for e in ok), verified=len(ok))
        draw["runs"].append(path.split("deepseek-flash/")[1].split("/results")[0])
    return sorted(draws.values(), key=lambda d: d["solved"] / d["verified"])


def fit_prior(tasks):
    """(a, b) maximizing the beta-binomial likelihood of [(successes, executions)] on a grid."""
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


def posterior_mean(runs, a, b):
    return 100 * float(np.mean([(a + sum(v)) / (a + b + len(v)) for v in runs.values()]))


def batch_size(scores):
    """Smallest n <= 20 that makes every score a whole number of tasks."""
    return next(n for n in range(1, 21) if all(abs(s * n - round(s * n)) < 1e-6 for s in scores))


def skilladam_observed(scored):
    n = batch_size([i["baseline"] for i in scored] + [i["candidate"] for i in scored])
    diffs = [round((i["candidate"] - i["baseline"]) * n) for i in scored]
    gains = [d for d, i in zip(diffs, scored) if i["accepted"]]
    return {"iterations": len(scored), "batch": n, "budgets": [i["budget"] for i in scored], "attempts": sum(i["attempts"] for i in scored),
            "accepted": sum(1 for i in scored if i["accepted"]), "accepted_gains": gains, "reported_gain_points": 100 * sum(gains) / n,
            "better": sum(d > 0 for d in diffs), "worse": sum(d < 0 for d in diffs), "tied": sum(d == 0 for d in diffs)}


def observed(bench, n_dev):
    """Candidates scored, baseline score and reported gain (dev tasks) of each optimizer's Table 1 run."""
    so, ge, sa = (json.load(open(EXECUTION / f"{m}/{bench}/{RUN}/summary.json"))["extra"] for m in ("skillopt", "gepa", "skilladam"))
    vals = ge["val_scores"]
    return {"SkillOpt": {"K": so["total_accepts"] + so["total_rejects"], "baseline": round(so["baseline_selection_hard"] * n_dev),
                         "gain": round((so["best_selection_hard"] - so["baseline_selection_hard"]) * n_dev)},
            "GEPA": {"K": len(vals) - 1, "baseline": round(vals[0] * n_dev), "gain": round((max(vals) - vals[0]) * n_dev)},
            "SkillAdam": skilladam_observed([i for i in sa["iterations"] if isinstance(i.get("baseline"), (int, float))
                                             and isinstance(i.get("candidate"), (int, float))])}


def outcomes(p):
    """Solved tasks of one run per draw; p is (draws, tasks)."""
    return (RNG.random(p.shape) < p).sum(axis=1)


def best_of(p, k, baseline):
    best = (RNG.random((p.shape[0], k, p.shape[1])) < p[:, None, :]).sum(axis=2).max(axis=1)
    return np.maximum(best - baseline, 0)


def raised(p, gain):
    """p raised on the least-solved tasks until the expected solved count grows by `gain`."""
    q, order, rows = p.copy(), np.argsort(p, axis=1), np.arange(p.shape[0])
    left = np.full(p.shape[0], float(gain))
    for column in range(p.shape[1]):
        index = order[:, column]
        add = np.minimum(1.0 - q[rows, index], left)
        q[rows, index] += add
        left -= add
    return q


def paired(p, q):
    return outcomes(q) - outcomes(p)


def critical_value(p, level=0.05):
    null = paired(p, p)
    return next(t for t in range(0, p.shape[1] + 2) if (null >= t).mean() <= level)


def mde80(p, step=0.25):
    """(one-sided 0.05 critical net difference, smallest gain detected with probability 0.8)."""
    critical = critical_value(p)
    for gain in np.arange(step, p.shape[1] + step, step):
        if (paired(p, raised(p, gain)) >= critical).mean() >= 0.8:
            return critical, float(gain)
    return critical, float("nan")


def skilladam_accepts(p_dev, batch, budgets):
    """Iterations that accept a null patch, with fresh outcomes for the current skill."""
    draws, n_dev = p_dev.shape
    accepted = np.zeros(draws, dtype=int)
    for budget in budgets:
        pick = np.argsort(RNG.random((draws, n_dev)), axis=1)[:, :batch]
        p = np.take_along_axis(p_dev, pick, axis=1)
        current = outcomes(p)
        accepted += ((RNG.random((draws, budget, batch)) < p[:, None, :]).sum(axis=2) > current[:, None]).any(axis=1)
    return accepted


def skilladam_reported_gain(p_dev, batch, budgets, rng):
    """(summed dev gain in points, accepted iterations) for patches identical to the seed; a skill-task outcome is drawn once and reused."""
    draws, n_dev = p_dev.shape
    rows = np.arange(draws)[:, None]
    current = np.full((draws, n_dev), -1, dtype=int)  # -1: not run yet
    total, accepts = np.zeros(draws), np.zeros(draws, dtype=int)
    for budget in budgets:
        pick = np.argsort(rng.random((draws, n_dev)), axis=1)[:, :batch]
        p = p_dev[rows, pick]
        seen = current[rows, pick]
        seen = np.where(seen < 0, (rng.random(p.shape) < p).astype(int), seen)
        current[rows, pick] = seen
        base = seen.sum(axis=1)
        patches = (rng.random((draws, budget, batch)) < p[:, None, :]).astype(int)
        better = patches.sum(axis=2) > base[:, None]
        accepted = better.any(axis=1)
        winner = patches[np.arange(draws), better.argmax(axis=1)]
        total += np.where(accepted, 100 * (winner.sum(axis=1) - base) / batch, 0)
        accepts += accepted
        current[accepted] = -1
        current[rows[accepted], pick[accepted]] = winner[accepted]
    return total, accepts


def null_summary(gain, accepts, sa):
    return {"E_reported_gain_points": float(gain.mean()), "P_reported_gain_ge_observed": float((gain >= sa["reported_gain_points"] - 1e-9).mean()),
            "E_accepts_cached": float(accepts.mean())}


def skilladam_null(p_dev, sa, rng):
    return null_summary(*skilladam_reported_gain(p_dev, sa["batch"], sa["budgets"], rng), sa)


def replay_power(p, k, baseline, null_gain, rng, gains=POWER_GAINS):
    """Critical gain of the null replay and P(reaching it) when one of the k candidates truly solves g more tasks."""
    critical = next(t for t in range(0, p.shape[1] + 2) if (null_gain >= t).mean() <= 0.05)
    out = {"critical_gain_tasks": critical}
    for g in gains:
        q = raised(p, float(g))
        scores = (rng.random((p.shape[0], k, p.shape[1])) < p[:, None, :]).sum(axis=2)
        scores[:, 0] = (rng.random(p.shape) < q).sum(axis=1)
        out[str(g)] = float((np.maximum(scores.max(axis=1) - baseline, 0) >= critical).mean())
    return out


def batch_gain_tail(p_dev, batch, gain):
    """P(candidate - current >= gain) for one null comparison on a random dev subset."""
    pick = np.argsort(RNG.random(p_dev.shape), axis=1)[:, :batch]
    p = np.take_along_axis(p_dev, pick, axis=1)
    return float((paired(p, p) >= gain).mean())


def mid_quantile(sample, value):
    return float((sample < value).mean() + (sample == value).mean() / 2)


def task_probabilities(dev, prior, size, draws):
    """(draws, size) task probabilities: a random subset of the dev posteriors, padded with prior draws beyond the panel."""
    pa, pb = prior
    tasks = sorted(dev)
    p = RNG.beta(np.array([pa + sum(dev[t]) for t in tasks]), np.array([pb + len(dev[t]) - sum(dev[t]) for t in tasks]), size=(draws, len(tasks)))
    if size <= len(tasks):
        return np.take_along_axis(p, np.argsort(RNG.random((draws, len(tasks))), axis=1)[:, :size], axis=1)
    return np.concatenate([p, RNG.beta(pa, pb, size=(draws, size - len(tasks)))], axis=1)


def spread(g0, G, **extra):
    """One Figure 4 row: reported gain G and the null replay g0, in points."""
    return dict(**extra, G=G, P=float((g0 >= G - 1e-9).mean()), mean=float(g0.mean()), p5=float(np.percentile(g0, 5)), p95=float(np.percentile(g0, 95)))


def analyse(bench):
    """Every number of one benchmark under both priors, and the Figure 4 replay rows (fitted prior)."""
    dev_files, test_files = seed_files(bench)
    dev, test = executions(dev_files), executions(test_files)
    check_inputs(bench, dev, test)
    n_dev = len(dev)
    pooled = [(sum(v), len(v)) for v in list(dev.values()) + list(test.values())]
    a, b = fit_prior(pooled)
    mean = sum(k for k, _ in pooled) / sum(n for _, n in pooled)
    obs = observed(bench, n_dev)
    out = {"dev_tasks": n_dev, "dev_executions_per_task": sorted({len(v) for v in dev.values()}), "test_tasks": len(test),
           "prior_fitted": [a, b], "mean_success": mean, "observed": obs, "priors": {}, "seed_dev_draws": dev_draws(dev_files, n_dev),
           "test_avg3": seed_test_avg3(bench),
           "expected_seed_success": {"dev": posterior_mean(dev, a, b), "test": posterior_mean(test, a, b), "prior_mean": 100 * a / (a + b)}}
    index, replay = [x for x, _ in BENCHES].index(bench), []
    for j, name in enumerate(PRIORS):
        prior = (a, b) if name == "fitted" else (MODERATE * mean, MODERATE * (1 - mean))
        reseed(index, j)
        p = task_probabilities(dev, prior, n_dev, DRAWS)
        rules = {}
        for opt in ("skillopt", "gepa"):
            o = obs[RULE[opt]]
            gain_random = best_of(p, o["K"], outcomes(p))
            gain_fixed = best_of(p, o["K"], o["baseline"])
            rules[RULE[opt]] = {
                "K": o["K"], "E_gain_random_baseline": float(gain_random.mean()), "E_gain_observed_baseline": float(gain_fixed.mean()),
                "P_ge_observed_random_baseline": float((gain_random >= o["gain"]).mean()),
                "P_ge_observed_observed_baseline": float((gain_fixed >= o["gain"]).mean()),
                "hist_random_baseline": np.bincount(gain_random, minlength=n_dev + 1).tolist(),
                "hist_observed_baseline": np.bincount(gain_fixed, minlength=n_dev + 1).tolist(),
                "power_observed_baseline": replay_power(p, o["K"], o["baseline"], gain_fixed, np.random.default_rng([20261012, index, j, 11, o["K"]]))}
            if name == "fitted":
                replay.append(spread(100 * gain_fixed / n_dev, 100 * o["gain"] / n_dev, bench=bench, opt=opt, K=o["K"], start=f"{o['baseline']}/{n_dev}"))
        sa = obs["SkillAdam"]
        accepts = skilladam_accepts(p, sa["batch"], sa["budgets"])
        rules["SkillAdam"] = {"E_accepts": float(accepts.mean()), "P_accepts_ge_observed": float((accepts >= sa["accepted"]).mean()),
                              "P_better_one_comparison": batch_gain_tail(p, sa["batch"], 1),
                              "P_gain_ge_accepted": {str(g): batch_gain_tail(p, sa["batch"], g) for g in sa["accepted_gains"]}}
        gain, acc = skilladam_reported_gain(p, sa["batch"], sa["budgets"], np.random.default_rng([20261012, index, j, 7]))
        rules["SkillAdam"].update(null_summary(gain, acc, sa))
        if name == "fitted":
            replay.append(spread(gain, sa["reported_gain_points"], bench=bench, opt="skilladam", K=len(sa["budgets"]), batch=sa["batch"]))
        null = paired(p, p)
        critical, mde = mde80(p)
        out["priors"][name] = {"prior": list(prior), "rules": rules, "seed_draw_mid_quantile": mid_quantile(outcomes(p), obs["SkillOpt"]["baseline"]),
                               "accept_if_better": {"P_accept_no_effect": float((null > 0).mean()), "SD_net_tasks": float(null.std()),
                                                    "P_accept_true_gain": {str(g): float((paired(p, raised(p, g)) > 0).mean()) for g in GAINS},
                                                    "critical_net_tasks": critical, "MDE80_tasks": mde, "MDE80_points": 100 * mde / n_dev}}
    return out, replay


def record(result):
    """mech_selection.md."""
    md, rows, rows2, draws = logger.md_table, [], [], []
    for bench, _ in BENCHES:
        r = result[bench]
        for prior in PRIORS:
            g = r["priors"][prior]
            for rule in ("SkillOpt", "GEPA"):
                x, o = g["rules"][rule], r["observed"][rule]
                rows.append([bench, prior, rule, x["K"], f"{o['baseline']}/{r['dev_tasks']}", f"+{o['gain']}", f"{x['E_gain_random_baseline']:.2f}",
                             f"{x['E_gain_observed_baseline']:.2f}", f"{x['P_ge_observed_random_baseline']:.3f}", f"{x['P_ge_observed_observed_baseline']:.3f}"])
            sa, o = g["rules"]["SkillAdam"], r["observed"]["SkillAdam"]
            gains = ", ".join(f"+{k} (P {sa['P_gain_ge_accepted'][str(k)]:.3f})" for k in o["accepted_gains"])
            rows.append([bench, prior, "SkillAdam", f"{o['iterations']} iterations on {o['batch']}-task dev subsets, {o['attempts']} patches",
                         f"better/worse/tied {o['better']}/{o['worse']}/{o['tied']}", f"{o['accepted']} accepted: {gains}",
                         f"E[accepts] {sa['E_accepts']:.2f} fresh outcomes, {sa['E_accepts_cached']:.2f} with reuse",
                         f"P(better) per comparison {sa['P_better_one_comparison']:.3f}", f"P(accepts >= observed) {sa['P_accepts_ge_observed']:.3f}",
                         f"summed gain {o['reported_gain_points']:.1f} pts; with reuse E {sa['E_reported_gain_points']:.1f} pts, "
                         f"P(>= observed) {sa['P_reported_gain_ge_observed']:.2f}"])
            ab = g["accept_if_better"]
            rows2.append([bench, prior, r["dev_tasks"], f"{ab['P_accept_no_effect']:.3f}", f"{ab['SD_net_tasks']:.2f}",
                          " / ".join(f"{ab['P_accept_true_gain'][str(k)]:.3f}" for k in GAINS), ab["critical_net_tasks"],
                          f"{ab['MDE80_tasks']:.2f} ({ab['MDE80_points']:.1f} pts)"])
        e, qs = r["expected_seed_success"], [100 * d["solved"] / d["verified"] for d in r["seed_dev_draws"]]
        for d in r["seed_dev_draws"]:
            draws.append([bench, f"{d['solved']}/{d['verified']}", f"{100 * d['solved'] / d['verified']:.1f}", len(d["runs"]),
                          ", ".join(d["runs"][:3]) + (" ..." if len(d["runs"]) > 3 else "")])
        draws.append([bench, "summary", f"mean {np.mean(qs):.1f}, min {min(qs):.1f}, max {max(qs):.1f}",
                      f"run's start draw {r['observed']['SkillOpt']['baseline']}/{r['dev_tasks']} (mid-quantile {r['priors']['fitted']['seed_draw_mid_quantile']:.2f})",
                      f"expected seed success under the fitted posteriors: dev {e['dev']:.1f}, test {e['test']:.1f}; prior mean {e['prior_mean']:.1f}; test avg@3 {r['test_avg3']}"])
    power = [[b, prior, rule, f"{result[b]['observed'][rule]['baseline']}/{result[b]['dev_tasks']}", f"+{result[b]['observed'][rule]['gain']}",
              x["critical_gain_tasks"]] + [f"{x[str(g)]:.2f}" for g in POWER_GAINS]
             for b, _ in BENCHES for prior in PRIORS for rule in ("SkillOpt", "GEPA") for x in [result[b]["priors"][prior]["rules"][rule]["power_observed_baseline"]]]
    priors = [[b, result[b]["dev_tasks"], result[b]["dev_executions_per_task"], result[b]["test_tasks"],
               f"({result[b]['prior_fitted'][0]:.2f}, {result[b]['prior_fitted'][1]:.2f})", f"{result[b]['mean_success']:.3f}"] for b, _ in BENCHES]
    logger.write_record(
        "selection", "The dev gain each optimizer's selection rule reports when every candidate is identical to the seed, against the gain its "
        "Table 1 run reported; an accept-if-better comparison of two single runs; the seed's dev draws next to its test success.",
        f"Seed: its distinct dev executions (version {SEED_VERSION}, deduplicated by request ids) and test executions (native-seed runs "
        f"{', '.join(SEED_TEST_RUNS)}). Optimizers: summary.json `extra` of skillopt/gepa/skilladam/<bench>/{RUN}.",
        f"Beta-binomial prior fitted to the seed's dev and test executions; moderate prior Beta(4m, 4(1-m)). {DRAWS} draws per setting, "
        "random seed [20261012, benchmark index, prior index]. A true gain of g tasks raises the least-solved tasks.",
        "### Seed task model\n\n" + md(["bench", "dev tasks", "seed executions per dev task", "test tasks", "fitted prior (a, b)", "mean success"], priors)
        + "\n\n### The seed's dev draws and its test success\n\n" + md(["bench", "seed dev draw (solved/verified)", "success %", "runs using it",
                                                                        "runs (first three)"], draws)
        + "\n\n### Selection rules under no effect\n\n" + md(["bench", "prior", "rule", "K / iterations", "observed baseline", "observed gain",
                                                               "E[gain], random baseline", "E[gain], observed baseline", "P(gain >= obs), random",
                                                               "P(gain >= obs), observed"], rows)
        + "\n\n### Accept-if-better on the dev panel (two single runs)\n\n" + md(["bench", "prior", "dev tasks", "P(accept | no effect)",
                                                                                   "SD of net difference (tasks)", "P(accept | true gain 1 / 2 / 3 tasks)",
                                                                                   "critical net difference (one-sided 0.05)", "MDE80"], rows2)
        + "\n\n### Power of the replay from the starting draw (one candidate truly better)\n\n"
        + md(["bench", "prior", "rule", "starting draw", "observed gain", "critical gain (one-sided 0.05, tasks)"]
             + [f"power, one candidate +{g} task{'s' if g > 1 else ''}" for g in POWER_GAINS], power),
        "python -m src.analysis.selection compute\npython -m src.analysis.selection draw")


def compute():
    result, replay = {}, []
    for bench, _ in BENCHES:
        result[bench], rows = analyse(bench)
        replay += rows
    write_json("selection", result)
    rnd = lambda r: {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}
    write_json("dev2test", dict(replays=DRAWS, prior="fitted", panel_a=[rnd(r) for r in replay]))
    record(result)


# Figure 4
INK, GRID, BAR, TICK, NOTE = "#222222", "#E2E5EA", "#D5DAE0", "#5F6873", "#6B7480"
COL = {"gepa": "#2A9D4B", "skillopt": "#1F6FB5", "skilladam": "#E08A1E"}
NAME = {"gepa": "GEPA", "skillopt": "SkillOpt", "skilladam": "SkillAdam"}
MARK = {"tau3-bench": "o", "automationbench": "s", "terminal-bench-2-1": "^"}
SHORT = {"tau3-bench": "τ³-bench", "automationbench": "AutoBench", "terminal-bench-2-1": "TB2"}
ORDER = ["tau3-bench", "automationbench", "terminal-bench-2-1"]
FS = 6.8


def panel_b(rows):
    """Returned skills: reported dev gain against the test avg@3 difference to the seed (Table 1), points, one decimal."""
    t1 = json.loads((out_dir("table1") / "table1.json").read_text())
    G = {(r["bench"], r["opt"]): r["G"] for r in rows}
    pts = []
    for b in ORDER:
        c = t1[f"deepseek-flash|{b}"]
        pts += [dict(bench=b, opt=o, dev=round(G[b, o], 1), test=round(c[RULE[o]]["Q"] - c["Seed skill"]["Q"], 1)) for o in ("gepa", "skillopt", "skilladam")]
    xy = np.array([(p["dev"], p["test"]) for p in pts])
    fit = lambda a: (round(float(np.polyfit(a[:, 0], a[:, 1], 1)[0]), 3), round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3))
    (s9, c9), (s6, c6) = fit(xy), fit(np.array([(p["dev"], p["test"]) for p in pts if p["opt"] != "skilladam"]))
    return dict(returned=pts, nine_slope=s9, nine_corr=c9, skillopt_gepa_slope=s6, skillopt_gepa_corr=c6)


def mark(ax, x, y, opt, bench, s):
    ax.scatter([x], [y], s=s * (0.85 if bench == "automationbench" else 1.0), marker=MARK[bench], facecolor=COL[opt], edgecolor="white",
               lw=0.6, zorder=4, clip_on=False)


def style(ax, grid):
    ax.grid(axis=grid, color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7, pad=1.5)


def forest(ax, opts, xlabel, xlim, xticks, R, h=0.30):
    import matplotlib.transforms as T
    from matplotlib.patches import Rectangle
    ys, y = [], 0.0
    for opt in opts:
        y0 = y
        for bench in ORDER:
            r = R[bench, opt]
            ax.add_patch(Rectangle((r["p5"], y - h), max(r["p95"] - r["p5"], 0.0), 2 * h, facecolor=BAR, edgecolor="none", zorder=1))
            ax.plot([r["mean"]] * 2, [y - 1.45 * h, y + 1.45 * h], color=TICK, lw=1.1, zorder=2, solid_capstyle="butt")
            mark(ax, r["G"], y, opt, bench, 34)
            ys.append(y)
            y += 1.0
        ax.text(-0.36, (y0 + y - 1) / 2, NAME[opt], transform=T.blended_transform_factory(ax.transAxes, ax.transData), ha="right",
                va="center", fontsize=7.5, fontweight="bold", color=COL[opt])
        y += 0.5
    ax.set_yticks(ys)
    ax.set_yticklabels([SHORT[b] for _ in opts for b in ORDER])
    ax.set_ylim(y - 0.85, -0.65)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    style(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2, labelsize=7)
    ax.set_xlabel(xlabel, fontsize=7, labelpad=1.5)


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 7.5, "mathtext.fontset": "stix", "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "axes.edgecolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK})
    num = json.loads((CACHE / "dev2test.json").read_text())
    num["panel_b"] = b = panel_b(num["panel_a"])
    write_json("dev2test", num)
    R = {(r["bench"], r["opt"]): r for r in num["panel_a"]}
    ht = 2.30
    fig = plt.figure(figsize=(6.3, ht))
    ax = fig.add_axes([0.215, 1.24 / ht, 0.275, 0.88 / ht])
    forest(ax, ("skillopt", "gepa"), "dev gain, points", (-1, 27), [0, 5, 10, 15, 20, 25], R)
    sx = fig.add_axes([0.215, 0.52 / ht, 0.275, 0.39 / ht])
    forest(sx, ("skilladam",), "SkillAdam dev gain, points", (-5, 140), list(range(0, 141, 20)), R)
    fig.text(0.008, 2.21 / ht, "(a) Unchanged candidates reach every reported dev gain", fontsize=7.5, fontweight="bold", ha="left", va="center")
    ha = [Line2D([0], [0], marker="o", color="#555555", lw=0, markersize=4.5, label="marker: reported dev gain"),
          Line2D([0], [0], color=BAR, lw=4.5, solid_capstyle="butt", label="bar: 5 to 95% of the gain unchanged candidates report"),
          Line2D([0], [0], color=TICK, lw=0, marker="|", markersize=7, markeredgewidth=1.1, label="tick: its mean")]
    fig.legend(handles=ha, loc="lower left", ncol=3, frameon=False, fontsize=FS, handlelength=1.0, columnspacing=0.7, handletextpad=0.3,
               bbox_to_anchor=(0.008, 0.02 / ht), borderaxespad=0.0, borderpad=0.0)
    bx = fig.add_axes([0.655, 0.66 / ht, 0.335, 1.46 / ht])
    bx.axhline(0, color=TICK, lw=0.7, zorder=1)
    for p in b["returned"]:
        mark(bx, p["dev"], p["test"], p["opt"], p["bench"], 52)
    bx.set_xlim(-1.5, 43)
    bx.set_ylim(-13, 6)
    bx.set_yticks([-10, -5, 0, 5])
    bx.set_xticks([0, 10, 20, 30, 40])
    bx.text(42.5, 5.6, f"slope {b['nine_slope']:.2f}, correlation {b['nine_corr']:.2f}", fontsize=FS, color=NOTE, ha="right", va="top")
    bx.set_xlabel("reported dev gain, points", fontsize=7, labelpad=1.5)
    bx.set_ylabel("test difference to the seed, points", fontsize=7, labelpad=1)
    style(bx, "y")
    bx.tick_params(axis="x", length=2.5)
    fig.text(0.585, 2.21 / ht, "(b) Test does not follow the dev gain", fontsize=7.5, fontweight="bold", ha="left", va="center")
    hb = []
    for opt, bench in zip(("skillopt", "gepa", "skilladam"), ORDER):
        hb += [Line2D([0], [0], marker="o", color=COL[opt], lw=0, markersize=5, label=NAME[opt]),
               Line2D([0], [0], marker=MARK[bench], color="#555555", lw=0, markersize=4.5 if bench == "automationbench" else 5, label=SHORT[bench])]
    fig.legend(handles=hb, loc="lower left", ncol=3, frameon=False, fontsize=FS, handlelength=0.8, columnspacing=1.2, handletextpad=0.3,
               labelspacing=0.35, bbox_to_anchor=(0.70, 0.02 / ht), borderaxespad=0.0, borderpad=0.0)
    save_figure(fig, "dev2test4")


if __name__ == "__main__":
    {"compute": compute, "draw": draw}[sys.argv[1]]()
