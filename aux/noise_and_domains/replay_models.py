"""Prior-free robustness of the selection replay (aux.selection_replay.selection).

Re-runs selection.py's replay (its own best_of / skilladam_null / mde80 / paired, unchanged) per optimizer x benchmark under task models
that drop the Beta-prior exchangeability assumption (A2):

  M0  fitted      Beta(a + s, b + n - s), (a, b) beta-binomial ML over dev + test tasks  -- the paper; reproduced exactly (same RNG stream)
  M4  moderate    Beta(4m + s, 4(1 - m) + n - s)                                           -- the paper's sensitivity prior
  M1  plug-in     p_v = s_v / n_v, fixed; no prior, no pooling (0/n and n/n tasks get zero variance)
  M2u Beta(1+s, 1+n-s)      per task, uniform prior, not fitted, no pooling
  M2j Beta(.5+s, .5+n-s)    per task, Jeffreys prior, not fitted, no pooling
  M3  resample    each simulated outcome resampled from the task's own observed seed executions (== Bernoulli(s/n), M1 in law;
                  run as a check that the replay code agrees)
  M3b Bayes boot  per task Dirichlet(1,...,1) weights over its own executions == Beta(s, n - s) (degenerate at 0/n, n/n)

Dev executions are restricted to the replay's inputs: the post-upgrade seed run (native-seed/tau3-bench/native-20261020-1017,
model fingerprint e72d3c) and the LODO runs (a 27-task dev panel, 5 tasks outside the 30-task panel) are excluded; selection.py's
glob picks both up.

Run: python -m aux.noise_and_domains.replay_models
Writes replay_models.json to OUT/noise_and_domains.
"""
import json
import sys

import numpy as np

from aux.edit_gate_and_design import design as D
from aux.selection_replay import devsize as DV
from aux.selection_replay import selection as S
from src import layout

OUT = layout.OUT / "noise_and_domains"
EXCLUDE = ("native-20261020-1017", "lodo")
DRAWS = S.DRAWS
QUICK = "--quick" in sys.argv
if QUICK:
    DRAWS = 20_000


def seed_data(bench):
    dev_files, test_files = S.seed_files(bench)
    dev_files = [f for f in dev_files if not any(x in f for x in EXCLUDE)]
    return S.executions(dev_files), S.executions(test_files), dev_files


def probabilities(model, dev, prior_fit, mean, draws, rng):
    """(draws, n_dev) per-task success probabilities under a task model (tasks sorted as in selection.task_probabilities)."""
    tasks = sorted(dev)
    s = np.array([sum(dev[t]) for t in tasks], dtype=float)
    n = np.array([len(dev[t]) for t in tasks], dtype=float)
    if model in ("M1", "M3"):
        return np.broadcast_to(s / n, (draws, len(tasks))).copy()
    if model == "M2u":
        return rng.beta(1 + s, 1 + n - s, size=(draws, len(tasks)))
    if model == "M2j":
        return rng.beta(0.5 + s, 0.5 + n - s, size=(draws, len(tasks)))
    if model == "M3b":
        # Beta(s, n - s) with point masses at 0 / 1 for unanimous tasks
        a, b = np.maximum(s, 1e-12), np.maximum(n - s, 1e-12)
        p = rng.beta(a, b, size=(draws, len(tasks)))
        p[:, s == 0] = 0.0
        p[:, s == n] = 1.0
        return p
    raise ValueError(model)


def m3_outcomes(dev, draws, rng):
    """M3 as literal resampling: a function that draws one run's per-task outcomes by picking one of the task's observed executions."""
    tasks = sorted(dev)
    obs = [np.array(dev[t]) for t in tasks]

    def run(shape_prefix=(draws,)):
        cols = [o[rng.integers(0, len(o), size=shape_prefix)] for o in obs]
        return np.stack(cols, axis=-1)
    return run


def rules_block(p, obs, n_dev, index, j):
    """The paper's rule replay on probabilities p (uses S.RNG for SkillOpt/GEPA, a dedicated rng for SkillAdam as in selection.analyse)."""
    rules = {}
    for rule in ("SkillOpt", "GEPA"):
        k = obs[rule]["K"]
        gain_random = S.best_of(p, k, S.outcomes(p))
        gain_fixed = S.best_of(p, k, obs[rule]["baseline"])
        rules[rule] = {"K": k, "g_obs": obs[rule]["gain"], "baseline": obs[rule]["baseline"],
                       "E_random": float(gain_random.mean()), "E_observed": float(gain_fixed.mean()),
                       "P_random": float((gain_random >= obs[rule]["gain"]).mean()),
                       "P_observed": float((gain_fixed >= obs[rule]["gain"]).mean())}
    sa = obs["SkillAdam"]
    accepts = S.skilladam_accepts(p, sa["batch"], sa["budgets"])
    rules["SkillAdam"] = {"E_accepts_fresh": float(accepts.mean()), "accepted_obs": sa["accepted"],
                          "P_better_one_comparison": S.batch_gain_tail(p, sa["batch"], 1),
                          "P_gain_ge_accepted": {str(g): S.batch_gain_tail(p, sa["batch"], g) for g in sa["accepted_gains"]},
                          "g_obs_points": sa["reported_gain_points"],
                          **S.skilladam_null(p, sa, np.random.default_rng([20261012, index, j, 7]))}
    null = S.paired(p, p)
    critical, mde = S.mde80(p)
    rules["accept_if_better"] = {"P_accept_no_effect": float((null > 0).mean()), "SD_net": float(null.std()),
                                 "critical": critical, "MDE80_tasks": mde, "MDE80_points": 100 * mde / n_dev}
    return rules


def selrules_block(p, mde_tasks):
    """design.selrules acceptance rates (rules a-d) for identical candidates and a true +1/+2/+4-task gain, on probabilities p."""
    out = {"null": D.acceptance(p, p, mde_tasks)}
    for g in (1, 2, 4):
        out[str(g)] = D.acceptance(p, S.raised(p, g), mde_tasks)
    return out


def m3_literal(dev, obs, draws, seed):
    """M3 run literally: every simulated run resamples each task's outcome from the task's observed seed executions (SkillOpt, GEPA)."""
    rng = np.random.default_rng(seed)
    run = m3_outcomes(dev, draws, rng)
    out = {}
    for rule in ("SkillOpt", "GEPA"):
        k = obs[rule]["K"]
        best = np.max(np.stack([run().sum(axis=1) for _ in range(k)], axis=1), axis=1)
        base = run().sum(axis=1)
        gr, gf = np.maximum(best - base, 0), np.maximum(best - obs[rule]["baseline"], 0)
        out[rule] = {"E_random": float(gr.mean()), "E_observed": float(gf.mean()),
                     "P_random": float((gr >= obs[rule]["gain"]).mean()), "P_observed": float((gf >= obs[rule]["gain"]).mean())}
    return out


def devsize_nonparam(dev, test, size, draws, seed, model):
    """Dev-size replay beyond the panel WITHOUT the Beta form: extra tasks drawn from the empirical distribution of the seed's dev+test tasks
    (still exchangeability of tasks, A2-weak, but no parametric prior).  model 'M1': plug-in p of the resampled task; 'M2j': Jeffreys posterior
    of the resampled task.  Returns E[gain] in points of SkillOpt best-of-4 and GEPA best-of-7 (random baseline) and MDE80 points."""
    rng = np.random.default_rng(seed)
    pool = list(dev.values()) + list(test.values())
    s = np.array([sum(v) for v in pool], dtype=float)
    n = np.array([len(v) for v in pool], dtype=float)
    tasks = sorted(dev)
    sd = np.array([sum(dev[t]) for t in tasks], dtype=float)
    nd = np.array([len(dev[t]) for t in tasks], dtype=float)
    if size <= len(tasks):
        pick = np.argsort(rng.random((draws, len(tasks))), axis=1)[:, :size]
        ss, nn = sd[pick], nd[pick]
    else:
        idx = rng.integers(0, len(pool), size=(draws, size - len(tasks)))
        ss = np.concatenate([np.broadcast_to(sd, (draws, len(tasks))), s[idx]], axis=1)
        nn = np.concatenate([np.broadcast_to(nd, (draws, len(tasks))), n[idx]], axis=1)
    p = ss / nn if model == "M1" else rng.beta(0.5 + ss, 0.5 + nn - ss)
    S.RNG = np.random.default_rng(seed + 1)
    base = S.outcomes(p)
    row = {r: 100 * float(S.best_of(p, k, base).mean()) / size for r, k in DV.DEVSIZE_RULES}
    critical, mde = S.mde80(p, step=0.5 if size > 30 else 0.25)
    row["MDE80_points"] = 100 * mde / size
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for index, (bench, _) in enumerate(S.BENCHES):
        dev, test, dev_files = seed_data(bench)
        n_dev = len(dev)
        pooled = [(sum(v), len(v)) for v in list(dev.values()) + list(test.values())]
        a, b = S.fit_prior(pooled)
        mean = sum(k for k, _ in pooled) / sum(n for _, n in pooled)
        obs = S.observed(bench, n_dev)
        r = {"n_dev": n_dev, "dev_exec_per_task": sorted({len(v) for v in dev.values()}), "prior": [a, b], "mean": mean,
             "dev_s_over_n": sorted(f"{sum(v)}/{len(v)}" for v in dev.values()),
             "share_unanimous_dev": float(np.mean([sum(v) in (0, len(v)) for v in dev.values()])),
             "models": {}, "selrules": {}, "devsize": {}}
        # M0 / M4: exactly selection.analyse's stream
        for j, name in enumerate(("fitted", "moderate")):
            prior = (a, b) if name == "fitted" else (S.MODERATE * mean, S.MODERATE * (1 - mean))
            S.reseed(index, j)
            p = S.task_probabilities(dev, prior, n_dev, DRAWS)
            r["models"]["M0" if j == 0 else "M4"] = rules_block(p, obs, n_dev, index, j)
        # prior-free models
        for j, name in enumerate(("M1", "M2u", "M2j", "M3b"), start=10):
            S.reseed(index, j)
            p = probabilities(name, dev, (a, b), mean, DRAWS, np.random.default_rng([20261020, index, j]))
            r["models"][name] = rules_block(p, obs, n_dev, index, j)
        r["models"]["M3_literal"] = m3_literal(dev, obs, DRAWS, [20261020, index, 99])
        # selrules (Fig. implications a) under M0 (paper), M1, M2j, each with its own MDE80
        for j, name in enumerate(("M0", "M1", "M2j"), start=30):
            S.reseed(index, j)
            if name == "M0":
                p = S.task_probabilities(dev, (a, b), n_dev, DRAWS)
            else:
                p = probabilities(name, dev, (a, b), mean, DRAWS, np.random.default_rng([20261020, index, j]))
            mde = r["models"][name]["accept_if_better"]["MDE80_tasks"]
            mde = mde if np.isfinite(mde) else float(n_dev + 1)
            r["selrules"][name] = selrules_block(p, mde)
        # dev-size curve, nonparametric task population (still exchangeable tasks)
        for name in ("M1", "M2j"):
            r["devsize"][name] = {str(size): devsize_nonparam(dev, test, size, 10_000 if QUICK else DV.DEVSIZE_DRAWS,
                                                              20261020 + 1000 * index + size, name) for size in DV.DEV_SIZES}
        result[bench] = r
        print(bench, json.dumps({m: {k: v for k, v in x.items() if k in ("SkillOpt", "GEPA")} for m, x in r["models"].items()}), flush=True)
    (OUT / ("replay_models_quick.json" if QUICK else "replay_models.json")).write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
