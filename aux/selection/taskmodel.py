"""The task model of dev selection, shared by every replay (selection, edits, three-word seed).

Each dev task has a success probability. Under the fitted model it is drawn from the task's posterior under a Beta prior,
fitted by beta-binomial maximum likelihood to the seed's distinct dev and test executions; the moderate model uses the
prior Beta(4m, 4(1 - m)) with m the seed's mean success. The prior-free models use each task's own executions: plug-in
s / n, the Bayesian bootstrap Beta(s, n - s) (point masses for unanimous tasks), or the uniform and Jeffreys posteriors.

A replay draws task probabilities p (draws, tasks), then applies an optimizer's selection rule to fresh outcomes drawn
from p with no candidate different from the seed: SkillOpt keeps the best of K candidates if strictly above its baseline,
GEPA takes the argmax over the seed and K candidates, SkillAdam accepts the first of up to `budget` patches that beats
the current skill on a random dev subset. A true gain of g tasks raises the least-solved tasks.

Every function that draws takes its generator; a caller draws p once per draw set and computes every rule on it."""
import json, math
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils import layout
from src.analysis import settings as S

MODERATE = 4.0  # prior strength of the moderate model
KEPT_DEV = layout.ROOT / "aux/selection/data/seed_dev_extra.json"


# ---- the seed's executions

def signature(e):
    """Request ids of an episode: one cached episode has one signature in every file."""
    return tuple(sorted(r["request_id"] for r in e["requests"]))


def verified(e):
    """An execution the task model counts: it ran (has requests) and has a verified outcome."""
    return bool(e["requests"]) and bool(e.get("outcome_valid")) and e.get("success") is not None


def executions(files):
    """{task: [0/1 per distinct execution]} over evaluation files."""
    seen = defaultdict(dict)
    for path in files:
        for e in json.loads(Path(path).read_text())["episodes"]:
            if verified(e):
                seen[e["task_id"]][signature(e)] = int(bool(e["success"]))
    return {task: list(runs.values()) for task, runs in seen.items()}


def run_name(path, model):
    """<method>/<bench>/<run> of an evaluation file."""
    return str(Path(path).relative_to(layout.EXECUTION / model)).split("/results")[0]


def dev_files(model, bench, version=S.SEED_VERSION):
    """Dev evaluation files of a skill version (default: the seed) in the seed dev pool (aux_runs seed_dev_pool); version
    None takes every version."""
    pattern = f"{version[:12]}-*/evaluation.json" if version else "*/evaluation.json"
    return sorted(str(f) for run in S.extra("seed_dev_pool", model, bench) for f in (run / "results/evaluations/dev").glob(pattern))


def kept_dev(bench, files):
    """{runs sharing one evaluation: [(task, 0/1, input tokens)]} of the DeepSeek seed dev executions kept in KEPT_DEV, whose
    runs the assets no longer hold; an evaluation whose runs are among files is left out."""
    kept = json.loads(KEPT_DEV.read_text())
    found = {run_name(f, S.DEEPSEEK) for f in files}
    out = defaultdict(list)
    for b, task, outcome, run, x in kept["executions"]:
        if b == bench and not found & set(kept["runs"][run]):
            out[tuple(kept["runs"][run])].append((task, outcome, x))
    return out


def seed_executions(model, bench):
    """(dev, test, dev files): the seed's {task: [0/1 per distinct execution]} on the dev panel (the seed dev pool, plus the
    kept DeepSeek executions) and on the test panel (Table 1's Seed row files)."""
    dev_f = dev_files(model, bench)
    dev = executions(dev_f)
    if model == S.DEEPSEEK:
        for rows in kept_dev(bench, dev_f).values():
            for task, outcome, _ in rows:
                dev.setdefault(task, []).append(outcome)
    test = executions([f for fs in S.cell_files(model, bench, "native-seed").values() for f in fs])
    return dev, test, dev_f


def null_executions(model, bench):
    """{task: [0/1]} over every dev evaluation, of any version, in the seed dev pool: under no effect every candidate's dev
    execution is a seed draw (the prior-free GPT-6-Luna models)."""
    return executions(dev_files(model, bench, version=None))


def dev_draws(model, bench, files, n_dev):
    """The seed's distinct full-panel dev draws ({runs, solved, verified}), the kept ones included, sorted by success."""
    draws = defaultdict(lambda: {"runs": []})
    for path in files:
        eps = json.loads(Path(path).read_text())["episodes"]
        if len(eps) != n_dev:
            continue
        ok = [e for e in eps if verified(e)]
        draw = draws[frozenset((e["task_id"], signature(e)) for e in ok)]
        draw.update(solved=sum(bool(e["success"]) for e in ok), verified=len(ok))
        draw["runs"].append(run_name(path, model))
    kept = [{"runs": list(runs), "solved": sum(o for _, o, _ in rows), "verified": len(rows)}
            for runs, rows in (kept_dev(bench, files).items() if model == S.DEEPSEEK else [])]
    return sorted([*draws.values(), *kept], key=lambda d: d["solved"] / d["verified"])


# ---- priors and task probabilities

def fit_prior(tasks):
    """(a, b) maximizing the beta-binomial likelihood of [(successes, executions)] on a grid."""
    def loglik(a, b):
        return sum(math.lgamma(k + a) + math.lgamma(n - k + b) - math.lgamma(n + a + b) - math.lgamma(a) - math.lgamma(b)
                   + math.lgamma(a + b) for k, n in tasks)
    best = None
    for m in np.linspace(0.01, 0.99, 99):
        for log_c in np.linspace(-2.0, 7.0, 91):
            c = math.exp(log_c)
            value = loglik(m * c, (1 - m) * c)
            if best is None or value > best[0]:
                best = (value, m * c, (1 - m) * c)
    return best[1], best[2]


def priors(dev, test):
    """({"fitted": (a, b), "moderate": (a, b)}, mean success) over the seed's dev and test executions."""
    pooled = [(sum(v), len(v)) for v in list(dev.values()) + list(test.values())]
    mean = sum(k for k, _ in pooled) / sum(n for _, n in pooled)
    return {"fitted": fit_prior(pooled), "moderate": (MODERATE * mean, MODERATE * (1 - mean))}, mean


def posterior_mean(runs, a, b):
    """Expected success (percent) of the tasks of {task: [0/1]} under their Beta(a, b) posteriors."""
    return 100 * float(np.mean([(a + sum(v)) / (a + b + len(v)) for v in runs.values()]))


def task_probabilities(dev, prior, size, draws, rng):
    """(draws, size) task probabilities: a random subset of the dev posteriors, padded with prior draws beyond the panel."""
    pa, pb = prior
    tasks = sorted(dev)
    p = rng.beta(np.array([pa + sum(dev[t]) for t in tasks]), np.array([pb + len(dev[t]) - sum(dev[t]) for t in tasks]),
                 size=(draws, len(tasks)))
    if size <= len(tasks):
        return np.take_along_axis(p, np.argsort(rng.random((draws, len(tasks))), axis=1)[:, :size], axis=1)
    return np.concatenate([p, rng.beta(pa, pb, size=(draws, size - len(tasks)))], axis=1)


def prior_free(kind, execs, tasks, draws, rng):
    """(draws, tasks) probabilities from each task's own executions: "plug-in" s / n (fixed), "bayes-boot" Beta(s, n - s)
    with point masses at 0 and 1, "uniform" Beta(1 + s, 1 + n - s), "jeffreys" Beta(0.5 + s, 0.5 + n - s)."""
    s = np.array([sum(execs[t]) for t in tasks], dtype=float)
    n = np.array([len(execs[t]) for t in tasks], dtype=float)
    if kind == "plug-in":
        return np.broadcast_to(s / n, (draws, len(tasks))).copy()
    if kind == "uniform":
        return rng.beta(1 + s, 1 + n - s, size=(draws, len(tasks)))
    if kind == "jeffreys":
        return rng.beta(0.5 + s, 0.5 + n - s, size=(draws, len(tasks)))
    if kind == "bayes-boot":
        p = rng.beta(np.maximum(s, 1e-12), np.maximum(n - s, 1e-12), size=(draws, len(tasks)))
        p[:, s == 0], p[:, s == n] = 0.0, 1.0
        return p
    raise ValueError(kind)


# ---- what the runs recorded

def batch_size(scores):
    """Smallest n <= 20 that makes every score a whole number of tasks."""
    return next(n for n in range(1, 21) if all(abs(s * n - round(s * n)) < 1e-6 for s in scores))


def skilladam_observed(scored):
    """SkillAdam's dev-gated iterations: batch size, budgets, attempts, accepted gains and the summed reported gain."""
    n = batch_size([i["baseline"] for i in scored] + [i["candidate"] for i in scored])
    diffs = [round((i["candidate"] - i["baseline"]) * n) for i in scored]
    gains = [d for d, i in zip(diffs, scored) if i["accepted"]]
    return {"iterations": len(scored), "batch": n, "budgets": [i["budget"] for i in scored], "attempts": sum(i["attempts"] for i in scored),
            "accepted": sum(1 for i in scored if i["accepted"]), "accepted_gains": gains, "reported_gain_points": 100 * sum(gains) / n,
            "better": sum(d > 0 for d in diffs), "worse": sum(d < 0 for d in diffs), "tied": sum(d == 0 for d in diffs)}


def observed(model, bench, n_dev, seed_solved=None):
    """Candidates scored, baseline (dev tasks) and reported gain of each dev-gated optimizer run the pins give. SkillOpt's
    baseline is its recorded starting score, else the seed draw it started at (seed_solved); a run that gated on the
    training batch is None."""
    def extra(method):
        return json.loads((S.run(model, bench, method) / "summary.json").read_text())["extra"]
    so, ge = extra("skillopt"), extra("gepa")
    history_file = S.run(model, bench, "skillopt") / "skillopt/history.json"
    history = json.loads(history_file.read_text()) if history_file.is_file() else []
    baseline = round(so["baseline_selection_hard"] * n_dev) if so.get("baseline_selection_hard") is not None else seed_solved
    vals = ge["val_scores"]
    out = {"SkillOpt": {"K": so["total_accepts"] + so["total_rejects"], "baseline": baseline,
                        "candidates": [round(h["selection_hard"] * n_dev) for h in history],
                        "gain": max(round(so["best_selection_hard"] * n_dev) - baseline, 0), "returned": so.get("best_origin")},
           "GEPA": {"K": len(vals) - 1, "baseline": round(vals[0] * n_dev), "candidates": [round(v * n_dev) for v in vals[1:]],
                    "gain": round((max(vals) - vals[0]) * n_dev), "returned": f"index {ge.get('best_idx')}"},
           "SkillAdam": None}
    if S.dev_gated(model, bench, "skilladam"):
        out["SkillAdam"] = skilladam_observed([i for i in extra("skilladam")["iterations"]
                                               if isinstance(i.get("baseline"), (int, float)) and isinstance(i.get("candidate"), (int, float))])
    return out


# ---- selection rules on fresh outcomes

def outcomes(p, rng):
    """Solved tasks of one run per draw; p is (draws, tasks)."""
    return (rng.random(p.shape) < p).sum(axis=1)


def best_of(p, k, baseline, rng):
    """Gain over baseline (tasks, at least 0) of the best of k candidates identical to the seed."""
    best = (rng.random((p.shape[0], k, p.shape[1])) < p[:, None, :]).sum(axis=2).max(axis=1)
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


def paired(p, q, rng):
    """Net solved tasks of one run with probabilities q against one with p."""
    return outcomes(q, rng) - outcomes(p, rng)


def critical_value(p, rng, level=0.05):
    """The smallest net difference that two identical single runs reach with probability at most level."""
    null = paired(p, p, rng)
    return next(t for t in range(0, p.shape[1] + 2) if (null >= t).mean() <= level)


def mde80(p, rng, step=0.25):
    """(one-sided 0.05 critical net difference, smallest gain detected with probability 0.8)."""
    critical = critical_value(p, rng)
    for gain in np.arange(step, p.shape[1] + step, step):
        if (paired(p, raised(p, gain), rng) >= critical).mean() >= 0.8:
            return critical, float(gain)
    return critical, float("nan")


def replay_power(p, k, baseline, null_gain, rng, gains=S.GAINS):
    """Critical gain of the null replay and P(reaching it) when one of the k candidates truly solves g more tasks."""
    critical = next(t for t in range(0, p.shape[1] + 2) if (null_gain >= t).mean() <= 0.05)
    out = {"critical_gain_tasks": critical}
    for g in gains:
        q = raised(p, float(g))
        scores = (rng.random((p.shape[0], k, p.shape[1])) < p[:, None, :]).sum(axis=2)
        scores[:, 0] = (rng.random(p.shape) < q).sum(axis=1)
        out[str(g)] = float((np.maximum(scores.max(axis=1) - baseline, 0) >= critical).mean())
    return out


def batch_gain_tail(p_dev, batch, gain, rng):
    """P(candidate - current >= gain) for one null comparison on a random dev subset."""
    pick = np.argsort(rng.random(p_dev.shape), axis=1)[:, :batch]
    p = np.take_along_axis(p_dev, pick, axis=1)
    return float((paired(p, p, rng) >= gain).mean())


def skilladam_accepts(p_dev, batch, budgets, rng):
    """Iterations that accept a null patch, with fresh outcomes for the current skill."""
    draws, n_dev = p_dev.shape
    accepted = np.zeros(draws, dtype=int)
    for budget in budgets:
        pick = np.argsort(rng.random((draws, n_dev)), axis=1)[:, :batch]
        p = np.take_along_axis(p_dev, pick, axis=1)
        current = outcomes(p, rng)
        accepted += ((rng.random((draws, budget, batch)) < p[:, None, :]).sum(axis=2) > current[:, None]).any(axis=1)
    return accepted


def skilladam_reported_gain(p_dev, batch, budgets, rng):
    """(summed dev gain in points, accepted iterations) for patches identical to the seed; a skill-task outcome is drawn
    once and reused."""
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


def skilladam_null(p_dev, sa, rng):
    """E[summed gain], P(summed gain >= the reported one) and E[accepts] of identical patches."""
    gain, accepts = skilladam_reported_gain(p_dev, sa["batch"], sa["budgets"], rng)
    return {"E_reported_gain_points": float(gain.mean()),
            "P_reported_gain_ge_observed": float((gain >= sa["reported_gain_points"] - 1e-9).mean()),
            "E_accepts_cached": float(accepts.mean())}


def runs_total(p, repeats, rng):
    """Tasks solved summed over `repeats` independent runs with task probabilities p."""
    total = np.zeros(p.shape[0], dtype=int)
    for _ in range(repeats):
        total += outcomes(p, rng)
    return total


def acceptance(p, q, critical, rng):
    """{rule: acceptance rate} of one candidate with probabilities q against its parent with p, on the same draws of p:
    (a) better on one run, (b) gain >= the critical value, (c) better, then better again, (d) mean of 3 runs."""
    parent = outcomes(p, rng)
    net = outcomes(q, rng) - parent
    net_again = outcomes(q, rng) - outcomes(p, rng)
    net_mean3 = runs_total(q, 3, rng) - runs_total(p, 3, rng)
    return {"better": float((net > 0).mean()), "critical": float((net >= critical).mean()),
            "confirm": float(((net > 0) & (net_again > 0)).mean()), "mean3": float((net_mean3 > 0).mean())}


def mid_quantile(sample, value):
    """The mid-quantile of value in sample."""
    return float((sample < value).mean() + (sample == value).mean() / 2)


def rank_test(baseline, candidates, gain):
    """Model-free P(G0 >= gain) under exchangeable scores: over each of the K + 1 scores as the baseline, max(others) - it."""
    scores = [baseline] + list(candidates)
    return float(np.mean([max(scores[:i] + scores[i + 1:]) - s >= gain for i, s in enumerate(scores)]))
