"""The dev gain each optimizer's own acceptance rule reports when every candidate is unchanged, against the gain its run
reported: the replay of dev selection for DeepSeek-V4.1-Flash (mech_selection) and GPT-6-Luna (mech_selection_luna), how
it changes with the size of the dev set (mech_devsize), and how much it rests on the task model (mech_assumptions).

Inputs. The seed's distinct dev executions are the seed dev pool of the experiments file (aux_runs seed_dev_pool, plus
DeepSeek's kept executions in aux/data/seed_dev_extra.json); its test executions are the files of Table 1's Seed row.
What each run recorded comes from its summary.json (SkillOpt also from skillopt/history.json): candidates scored K,
starting score, candidate scores and reported gain g. Only dev-gated runs are replayed (settings.dev_gated): a SkillAdam
run that gated its edits on the training batch reports no dev gain.

Task models. fitted: a Beta prior fitted by beta-binomial maximum likelihood to the seed's dev and test executions, each
dev task drawn from its posterior; moderate: the prior Beta(4m, 4(1 - m)) with m the seed's mean success. DeepSeek also
runs models without the pooled prior: one prior per domain, a prior fitted to the dev tasks only, and the prior-free
plug-in s / n, Bayesian bootstrap Beta(s, n - s), uniform and Jeffreys posteriors of each task's own executions. Under
GPT-6-Luna the seed has one dev execution per task, so its prior-free models are estimated under the null: every dev
execution of every version in the seed dev pool is a seed draw if no candidate differs from the seed.

Rules. SkillOpt keeps the best of K candidates if it strictly beats its starting score; GEPA takes the best of the seed
and its K candidates; SkillAdam accepts the first of up to `budget` patches that beats the current skill on a random dev
subset, reusing a task's outcome once drawn. P(G0 >= g) is how often unchanged candidates report at least the run's
gain (at least one task where the run reported none). A true gain of g tasks raises the least-solved tasks. Power: the
chance that the replay flags a run in which one candidate truly solves g more tasks. Accept-if-better compares two
single runs on the dev set; its critical value is the smallest margin unchanged candidates reach in at most one in twenty
draws, and MDE80 the smallest true gain such a comparison detects with probability four in five. The rank test needs no
model: under no effect the starting score and the K candidate scores are exchangeable.

Dev-set size: the dev tasks are a random subset of the posteriors at most the size of the dev set, else all of them
plus tasks drawn from the prior (or, in mech_assumptions, resampled from the seed's dev and test tasks); SkillOpt's best
of four and GEPA's best of seven, against a random baseline and against the baseline at the run's own mid-quantile.

Checks of the pooled prior (mech_assumptions): likelihood-ratio tests of one Beta against one per group of tasks
(domain, dev or test set), and the fit of the pooled prior to the counts of solved executions.

Draws: settings.N_DRAWS per task model, from one stream per (model, benchmark, task model): every rule of a benchmark is
replayed on the same task probabilities."""
import json, math

import numpy as np
from scipy import optimize, special, stats

from aux.selection import taskmodel as T
from src.analysis import settings as S
from src.utils.logger import md_table, record

DS, LUNA = S.DEEPSEEK, S.LUNA
BENCHES = ("terminal-bench-2-1", "tau3-bench", "automationbench")
RULES = ("SkillOpt", "GEPA")
MODELS = {DS: ("fitted", "moderate", "per domain", "dev only", "plug-in", "Bayesian bootstrap", "uniform", "Jeffreys"),
          LUNA: ("fitted", "moderate", "null plug-in", "null Bayesian bootstrap")}
PRIOR_FREE = {"plug-in": "plug-in", "Bayesian bootstrap": "bayes-boot", "uniform": "uniform", "Jeffreys": "jeffreys"}
SIZES = (9, 15, 30, 60, 120)
SIZE_K = (("SkillOpt", 4), ("GEPA", 7))  # the candidates the tau3 and AutoBench runs scored
OUT = S.cache("selection")


# ---- inputs

def ml_fit(k, n):
    """(a, b, log-likelihood) of the beta-binomial maximum likelihood fit to arrays of successes k of n executions."""
    def ll(m, c):
        a, b = m * c, (1 - m) * c
        with np.errstate(invalid="ignore"):
            return float((special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)
                          + special.betaln(k + a, n - k + b) - special.betaln(a, b)).sum())
    start = max((ll(m, math.exp(c)), m, c) for m in np.linspace(0.02, 0.98, 25) for c in np.linspace(-3, 6, 19))
    best = optimize.minimize(lambda x: -ll(special.expit(x[0]), math.exp(min(x[1], 12))), [special.logit(start[1]), start[2]],
                             method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
    m, c = special.expit(best.x[0]), math.exp(min(best.x[1], 12))
    return m * c, (1 - m) * c, -best.fun


def inputs(model, bench):
    """The seed's executions, its priors and draws, and what the benchmark's runs recorded."""
    dev, test, files = T.seed_executions(model, bench)
    priors, mean = T.priors(dev, test)
    draws = T.dev_draws(model, bench, files, len(dev))
    tasks = [(t, "dev") for t in sorted(dev)] + [(t, "test") for t in sorted(test)]
    table1 = json.loads((S.cache("table1") / "table1.json").read_text())[f"{model}|{bench}"]
    return dict(bench=bench, dev=dev, test=test, priors=priors, mean=mean, draws=draws,
                obs=T.observed(model, bench, len(dev), draws[0]["solved"] if len(draws) == 1 else None),
                k=np.array([sum((dev if s == "dev" else test)[t]) for t, s in tasks]),
                n=np.array([len((dev if s == "dev" else test)[t]) for t, s in tasks]),
                domain=np.array([S.domain_of(bench, t) for t, _ in tasks]), split=np.array([s for _, s in tasks]),
                null=T.null_executions(model, bench) if model == LUNA else None, test_avg3=round(table1["Seed skill"]["Q"], 1))


def probabilities(kind, x, size, r):
    """(draws, size) task probabilities under a task model."""
    dev, tasks = x["dev"], sorted(x["dev"])
    if kind in x["priors"]:
        return T.task_probabilities(dev, x["priors"][kind], size, S.N_DRAWS, r)
    if kind in ("per domain", "dev only"):
        groups = x["domain"] if kind == "per domain" else x["split"]
        fits = {g: ml_fit(x["k"][groups == g], x["n"][groups == g])[:2] for g in set(groups) if kind == "per domain" or g == "dev"}
        a, b = np.array([fits[S.domain_of(x["bench"], t) if kind == "per domain" else "dev"] for t in tasks]).T
        s, n = np.array([sum(dev[t]) for t in tasks]), np.array([len(dev[t]) for t in tasks])
        return r.beta(a + s, b + n - s, size=(S.N_DRAWS, len(tasks)))
    return T.prior_free(PRIOR_FREE[kind.replace("null ", "")], x["null"] if kind.startswith("null") else dev, tasks, S.N_DRAWS, r)


# ---- the replay

def replay(p, x, r, power):
    """Every statistic of one task model, on one draw of the task probabilities p; the fixed-baseline gains for Figure 4."""
    n_dev, obs, out, gains = p.shape[1], x["obs"], {"rules": {}}, {}
    for rule in RULES:
        o = obs[rule]
        rand, fixed = T.best_of(p, o["K"], T.outcomes(p, r), r), T.best_of(p, o["K"], o["baseline"], r)
        g = out["rules"][rule] = {
            "K": o["K"], "E_gain_random_baseline": float(rand.mean()), "E_gain_observed_baseline": float(fixed.mean()),
            "P_ge_observed_random_baseline": float((rand >= max(o["gain"], 1)).mean()),
            "P_ge_observed_observed_baseline": float((fixed >= max(o["gain"], 1)).mean()),
            "hist_random_baseline": np.bincount(rand, minlength=n_dev + 1).tolist(),
            "hist_observed_baseline": np.bincount(fixed, minlength=n_dev + 1).tolist()}
        if power:
            g["power_observed_baseline"] = T.replay_power(p, o["K"], o["baseline"], fixed, r)
        gains[rule] = 100 * fixed / n_dev
    if sa := obs["SkillAdam"]:
        accepts = T.skilladam_accepts(p, sa["batch"], sa["budgets"], r)
        gain, cached = T.skilladam_reported_gain(p, sa["batch"], sa["budgets"], r)
        out["rules"]["SkillAdam"] = {
            "E_accepts": float(accepts.mean()), "P_accepts_ge_observed": float((accepts >= max(sa["accepted"], 1)).mean()),
            "P_better_one_comparison": T.batch_gain_tail(p, sa["batch"], 1, r),
            "P_gain_ge_accepted": {str(g): T.batch_gain_tail(p, sa["batch"], g, r) for g in sorted(set(sa["accepted_gains"]))},
            "E_reported_gain_points": float(gain.mean()), "E_accepts_cached": float(cached.mean()),
            "P_reported_gain_ge_observed": float((gain >= sa["reported_gain_points"] - 1e-9).mean() if sa["accepted"] else (gain > 0).mean())}
        gains["SkillAdam"] = gain
    null = T.paired(p, p, r)
    critical, mde = T.mde80(p, r)
    out["accept_if_better"] = {"P_accept_no_effect": float((null > 0).mean()), "SD_net_tasks": float(null.std()),
                               "P_accept_true_gain": {str(g): float((T.paired(p, T.raised(p, g), r) > 0).mean()) for g in S.GAINS},
                               "critical_net_tasks": critical, "MDE80_tasks": mde, "MDE80_points": 100 * mde / n_dev}
    out["seed_draw_mid_quantile"] = T.mid_quantile(T.outcomes(p, r), obs["SkillOpt"]["baseline"])
    return out, gains


def size_probabilities(kind, x, size, r):
    """Task probabilities of a dev set of `size` tasks; beyond the dev set, prior draws (fitted, moderate) or tasks resampled
    from the seed's dev and test tasks (plug-in, Jeffreys)."""
    if kind in x["priors"]:
        return T.task_probabilities(x["dev"], x["priors"][kind], size, S.N_DRAWS, r)
    s_dev, n_dev = (np.array([f(x["dev"][t]) for t in sorted(x["dev"])], dtype=float) for f in (sum, len))
    if size <= len(s_dev):
        pick = np.argsort(r.random((S.N_DRAWS, len(s_dev))), axis=1)[:, :size]
        s, n = s_dev[pick], n_dev[pick]
    else:
        extra = r.integers(0, len(x["k"]), size=(S.N_DRAWS, size - len(s_dev)))
        s = np.concatenate([np.broadcast_to(s_dev, (S.N_DRAWS, len(s_dev))), x["k"][extra]], axis=1)
        n = np.concatenate([np.broadcast_to(n_dev, (S.N_DRAWS, len(s_dev))), x["n"][extra]], axis=1)
    return s / n if kind == "plug-in" else r.beta(0.5 + s, 0.5 + n - s)


def devsize(model, bench, kind, x, quantile):
    rows = {}
    for size in SIZES:
        r = S.rng("devsize", model, bench, kind, size)
        p = size_probabilities(kind, x, size, r)
        base = T.outcomes(p, r)
        row = rows[str(size)] = {}
        if quantile is not None:
            row["baseline_at_quantile"] = at = float(np.quantile(base, quantile, method="nearest"))
        for rule, k in SIZE_K:
            row[rule] = {"E_gain_random_points": 100 * float(T.best_of(p, k, base, r).mean()) / size}
            if quantile is not None:
                row[rule]["E_gain_quantile_points"] = 100 * float(T.best_of(p, k, at, r).mean()) / size
        row["critical_net_tasks"], mde = T.mde80(p, r, step=0.5 if size > 30 else 0.25)
        row["MDE80_points"] = 100 * mde / size
    return rows


def analyse(model, bench):
    x = inputs(model, bench)
    dev, test, (a, b) = x["dev"], x["test"], x["priors"]["fitted"]
    out = {"dev_tasks": len(dev), "dev_executions_per_task": sorted({len(v) for v in dev.values()}), "test_tasks": len(test),
           "dev_executions": sum(map(len, dev.values())), "test_executions": sum(map(len, test.values())),
           "prior_fitted": [a, b], "mean_success": x["mean"], "observed": x["obs"], "seed_dev_draws": x["draws"],
           "test_avg3": x["test_avg3"], "priors": {k: list(v) for k, v in x["priors"].items()}, "models": {}, "devsize": {},
           "expected_seed_success": {"dev": T.posterior_mean(dev, a, b), "test": T.posterior_mean(test, a, b), "prior_mean": 100 * a / (a + b)}}
    panel = []
    for kind in (m for m in MODELS[model] if m != "per domain" or len(set(x["domain"])) > 1):
        r = S.rng("selection", model, bench, kind)
        out["models"][kind], gains = replay(probabilities(kind, x, len(dev), r), x, r, power=kind in x["priors"])
        if model == DS and kind == "fitted":
            for opt, g0 in gains.items():
                o = x["obs"][opt]
                G = o["gain"] * 100 / len(dev) if opt in RULES else o["reported_gain_points"]
                head = {"K": o["K"], "start": f"{o['baseline']}/{len(dev)}"} if opt in RULES else {"K": len(o["budgets"]), "batch": o["batch"]}
                panel.append({"bench": bench, "opt": opt.lower(), **head, "G": G, "P": float((g0 >= G - 1e-9).mean()), "mean": float(g0.mean()),
                              "p5": float(np.percentile(g0, 5)), "p95": float(np.percentile(g0, 95))})
    if model == DS:
        for kind in ("fitted", "moderate", "plug-in", "Jeffreys"):
            q = out["models"][kind]["seed_draw_mid_quantile"] if kind in x["priors"] else None
            out["devsize"][kind] = devsize(model, bench, kind, x, q)
        out["checks"] = checks(x)
        out["steps"] = steps(bench)
    else:
        nx, tasks = x["null"], sorted(dev)
        runs = [sum(v) for v in test.values()]
        out["test_always_sometimes_never"] = [runs.count(3), sum(0 < k < 3 for k in runs), runs.count(0)]
        out["null_executions"] = {"per_task_min": min(len(nx[t]) for t in tasks), "per_task_max": max(len(nx[t]) for t in tasks),
                                  "unanimous": sum(sum(nx[t]) in (0, len(nx[t])) for t in tasks), "s_over_n": {t: f"{sum(nx[t])}/{len(nx[t])}" for t in tasks}}
        out["rank_test"] = {rule: T.rank_test(x["obs"][rule]["baseline"], x["obs"][rule]["candidates"], max(x["obs"][rule]["gain"], 1)) for rule in RULES}
        out["aborted"] = aborted(bench, dev)
    return out, panel


def checks(x):
    """Likelihood-ratio tests of one Beta against one per group, and the pooled fit to the counts of solved executions."""
    k, n = x["k"], x["n"]
    a0, b0, l0 = ml_fit(k, n)
    groupings = {"panel": x["split"]} | ({} if len(set(x["domain"])) == 1 else
                                         {"domain": x["domain"], "domain x panel": np.char.add(x["domain"], x["split"])})
    lrt = []
    for name, g in groupings.items():
        fits = {G: ml_fit(k[g == G], n[g == G]) for G in np.unique(g)}
        lr, df = 2 * (sum(f[2] for f in fits.values()) - l0), 2 * (len(fits) - 1)
        lrt.append({"grouping": name, "groups": len(fits), "LR": lr, "df": df, "p_chi2": float(stats.chi2.sf(lr, df))})
    cells = sorted({(int(m), s) for m in np.unique(n) for s in range(m + 1)})
    obs = np.array([((n == m) & (k == s)).sum() for m, s in cells], float)
    exp = np.array([(n == m).sum() * math.comb(m, s) * math.exp(special.betaln(s + a0, m - s + b0) - special.betaln(a0, b0)) for m, s in cells])
    pbar, rho, m0 = k.sum() / n.sum(), 1 / (a0 + b0 + 1), a0 / (a0 + b0)
    return {"lrt": lrt, "prior": [a0, b0], "cells": cells, "obs": obs.tolist(), "exp": exp.tolist(), "pearson": float(((obs - exp) ** 2 / exp).sum()),
            "tarone_Z": float(((((k - n * pbar) ** 2) / (pbar * (1 - pbar))).sum() - n.sum()) / math.sqrt(2 * (n * (n - 1)).sum())),
            "var_s_n3_obs": float(k[n == 3].var(ddof=1)), "var_s_n3_pred": 3 * m0 * (1 - m0) * (1 + 2 * rho),
            "unanimous_dev": float(np.mean([sum(v) in (0, len(v)) for v in x["dev"].values()]))}


def steps(bench):
    """SkillAdam's steps: edits applied, steps whose patch was empty or did not parse, and each kept edit's batch score next to
    the kept skill's score on the following step's batch (a lucky win is never paid back)."""
    its = json.loads((S.run(DS, bench, "skilladam") / "summary.json").read_text())["extra"]["iterations"]
    applied = [i for i in its if i.get("patch_mode") not in ("empty", "failed")]
    margin = [round(i["candidate"] - i["baseline"], 9) for i in applied]
    n = T.batch_size([i["baseline"] for i in its] + [i["candidate"] for i in its])
    return {"steps": len(its), "applied": len(applied), "better": sum(m > 0 for m in margin), "worse": sum(m < 0 for m in margin),
            "tied": margin.count(0), "empty": sum(i.get("patch_mode") == "empty" for i in its), "failed": sum(i.get("patch_mode") == "failed" for i in its),
            "batch": n, "kept": [(round(i["candidate"] * n), round(its[k + 1]["baseline"] * n)) for k, i in enumerate(its) if i["accepted"] and k + 1 < len(its)]}


def aborted(bench, dev):
    """The aborted GPT-6-Luna SkillOpt run's seed dev draw against the pool's, on the tasks verified in both."""
    other = T.executions([str(f) for run in S.extra("aborted_seed_dev", LUNA, bench)
                          for f in (run / "results/evaluations/dev").glob(f"{S.SEED_VERSION[:12]}-*/evaluation.json")])
    common = sorted(set(dev) & set(other))
    return {"verified": len(other), "solved": sum(v[0] for v in other.values()), "common": len(common),
            "native_solved_common": sum(dev[t][0] for t in common), "aborted_solved_common": sum(other[t][0] for t in common),
            "native_1_aborted_0": sum(dev[t][0] > other[t][0] for t in common), "native_0_aborted_1": sum(dev[t][0] < other[t][0] for t in common)}


# ---- records

def runs(model, other=()):
    cells = [(model, b, m) for b in BENCHES for m in ("native-seed", *S.OPTIMIZERS) if m == "native-seed" or S.dev_gated(model, b, m)]
    return S.runs_text(cells, [p for b in BENCHES for name in ("seed_dev_pool", *other) for p in S.extra(name, model, b)])


def pts(o, n):
    return f"{o['baseline']}/{n}", f"+{o['gain']}"


def power_cells(x):
    pw = x["power_observed_baseline"]
    return [pw["critical_gain_tasks"], *(f"{pw[str(k)]:.3f}" for k in S.GAINS)]


POWER_HEAD = ["critical gain (one-sided 0.05, tasks)"] + [f"power, one candidate +{k} task{'s' if k > 1 else ''}" for k in S.GAINS]


def record_selection(res):
    model_rows, grid, draws, rules, accept, power, steps_rows = [], [], [], [], [], [], []
    for b in BENCHES:
        r, n = res[b], res[b]["dev_tasks"]
        (a_, b_), e, o, m = r["prior_fitted"], r["expected_seed_success"], r["observed"], max(r["dev_executions_per_task"])
        model_rows.append([b, n, r["dev_executions_per_task"], r["test_tasks"], f"({a_:.2f}, {b_:.2f})", f"{r['mean_success']:.3f}",
                           r["dev_executions"], r["test_executions"]])
        grid.append([b, f"{a_ + b_:.2f}", ", ".join(f"{s} of {m}: {(a_ + s) / (a_ + b_ + m):.2f}" for s in range(m, -1, -1))])
        qs = [100 * d["solved"] / d["verified"] for d in r["seed_dev_draws"]]
        draws += [[b, f"{d['solved']}/{d['verified']}", f"{q:.1f}", len(d["runs"]), ", ".join(d["runs"][:3]) + (" ..." if len(d["runs"]) > 3 else "")]
                  for d, q in zip(r["seed_dev_draws"], qs)]
        draws.append([b, "summary", f"mean {np.mean(qs):.1f}, min {min(qs):.1f}, max {max(qs):.1f}",
                      f"run's start draw {o['SkillOpt']['baseline']}/{n} (mid-quantile {r['models']['fitted']['seed_draw_mid_quantile']:.2f})",
                      f"expected seed success under the fitted posteriors: dev {e['dev']:.1f}, test {e['test']:.1f}; prior mean {e['prior_mean']:.1f}; "
                      f"test avg@3 {r['test_avg3']}"])
        for prior in ("fitted", "moderate"):
            g = r["models"][prior]
            for rule in RULES:
                x = g["rules"][rule]
                rules.append([b, prior, rule, x["K"], *pts(o[rule], n), f"{x['E_gain_random_baseline']:.2f}", f"{x['E_gain_observed_baseline']:.2f}",
                              f"{x['P_ge_observed_random_baseline']:.3f}", f"{x['P_ge_observed_observed_baseline']:.3f}"])
                power.append([b, prior, rule, *pts(o[rule], n), *power_cells(x)])
            x, s = g["rules"]["SkillAdam"], o["SkillAdam"]
            rules.append([b, prior, "SkillAdam", f"{s['iterations']} iterations on {s['batch']}-task dev subsets, {s['attempts']} patches",
                          f"better/worse/tied {s['better']}/{s['worse']}/{s['tied']}",
                          f"{s['accepted']} accepted: " + ", ".join(f"+{k} (P {x['P_gain_ge_accepted'][str(k)]:.3f})" for k in s["accepted_gains"]),
                          f"E[accepts] {x['E_accepts']:.2f} fresh outcomes, {x['E_accepts_cached']:.2f} with reuse",
                          f"P(better) per comparison {x['P_better_one_comparison']:.3f}", f"P(accepts >= observed) {x['P_accepts_ge_observed']:.3f}",
                          f"summed gain {s['reported_gain_points']:.1f} pts; with reuse E {x['E_reported_gain_points']:.1f} pts, "
                          f"P(>= observed) {x['P_reported_gain_ge_observed']:.2f}"])
            ab = g["accept_if_better"]
            accept.append([b, prior, n, f"{ab['P_accept_no_effect']:.3f}", f"{ab['SD_net_tasks']:.2f}",
                           " / ".join(f"{ab['P_accept_true_gain'][str(k)]:.3f}" for k in S.GAINS), ab["critical_net_tasks"],
                           f"{ab['MDE80_tasks']:.2f} ({ab['MDE80_points']:.1f} pts)"])
        st = r["steps"]
        steps_rows.append([b, st["steps"], st["applied"], f"{st['worse']} / {st['better']} / {st['tied']}", st["empty"], st["failed"],
                           ", ".join(f"{w}/{st['batch']} then {x}/{st['batch']}" for w, x in st["kept"])])
    total = {k: sum(res[b]["steps"][k] for b in BENCHES) for k in ("applied", "worse", "better", "tied", "empty", "failed")}
    numbers = "\n\n".join([
        f"{S.N_DRAWS} draws per task model and benchmark.",
        "### Seed task model\n\n" + md_table(["bench", "dev tasks", "seed executions per dev task", "test tasks", "fitted prior (a, b)",
                                              "mean success", "seed dev executions", "seed test executions"], model_rows),
        "### A dev task's p_v under the fitted prior: posterior mean after s solved of its executions\n\n" + md_table(["bench", "a + b", "s of n: p_v"], grid),
        "### The seed's dev draws and its test success\n\n" + md_table(["bench", "seed dev draw (solved/verified)", "success %", "runs using it",
                                                                        "runs (first three)"], draws),
        "### Selection rules under no effect\n\n" + md_table(["bench", "prior", "rule", "K / iterations", "observed baseline", "observed gain",
                                                               "E[gain], random baseline", "E[gain], observed baseline", "P(gain >= obs), random",
                                                               "P(gain >= obs), observed"], rules),
        "### SkillAdam's steps\n\n" + md_table(["bench", "steps", "edits applied", "scored below / above / tied with their parent", "patch empty",
                                                "no patch parsed", "kept edit's batch score, then the kept skill's on the next batch"], steps_rows)
        + f"\n\nOver the three benchmarks: {total['applied']} edits applied, {total['worse']} scored below their parent, {total['better']} above "
          f"and {total['tied']} tied; {total['empty']} steps with an empty patch, {total['failed']} with no parsed patch.",
        "### Accept-if-better on the dev panel (two single runs)\n\n"
        + md_table(["bench", "prior", "dev tasks", "P(accept | no effect)", "SD of net difference (tasks)",
                    f"P(accept | true gain {' / '.join(map(str, S.GAINS))} tasks)", "critical net difference (one-sided 0.05)", "MDE80"], accept),
        "### Power of the replay from the starting draw (one candidate truly better)\n\n"
        + md_table(["bench", "prior", "rule", "starting draw", "observed gain"] + POWER_HEAD, power)])
    record(__spec__.name, "selection", numbers, runs(DS),
           what="The dev gain each optimizer's selection rule reports when every candidate is identical to the seed, against the gain its "
                "Table 1 run reported; an accept-if-better comparison of two single runs; the seed's dev draws next to its test success "
                "(DeepSeek-V4.1-Flash).")


def record_luna(res):
    inputs_rows, run_rows, rules, adam, power, checks_rows = [], [], [], [], [], []
    label = {"terminal-bench-2-1": "TB2", "tau3-bench": "tau3", "automationbench": "AutoBench"}
    for b in BENCHES:
        r, lab, n = res[b], label[b], res[b]["dev_tasks"]
        d, e, nx, o = r["seed_dev_draws"][0], r["expected_seed_success"], r["null_executions"], r["observed"]
        inputs_rows.append([lab, n, r["dev_executions_per_task"], f"{d['solved']}/{d['verified']}", len(d["runs"]), r["test_tasks"],
                            " / ".join(map(str, r["test_always_sometimes_never"])), "({:.2f}, {:.2f})".format(*r["prior_fitted"]),
                            f"{r['mean_success']:.3f}", f"{r['models']['fitted']['seed_draw_mid_quantile']:.2f}",
                            f"dev {e['dev']:.1f}, test {e['test']:.1f}; test avg@3 {r['test_avg3']}",
                            f"{nx['per_task_min']}-{nx['per_task_max']} per task, {nx['unanimous']}/{n} unanimous", r["dev_executions"], r["test_executions"]])
        run_rows += [[lab, rule, o[rule]["K"], pts(o[rule], n)[0], " ".join(map(str, o[rule]["candidates"])), pts(o[rule], n)[1], o[rule]["returned"]]
                     for rule in RULES]
        if s := o["SkillAdam"]:
            run_rows.append([lab, "SkillAdam", f"{s['iterations']} iterations, {s['batch']}-task subsets, {s['attempts']} patches",
                             f"better/worse/tied {s['better']}/{s['worse']}/{s['tied']}",
                             "accepted gains " + (" ".join(f"+{g}" for g in s["accepted_gains"]) or "none"), f"{s['reported_gain_points']:.1f} pts summed",
                             f"{s['accepted']} accepted"])
        for name in MODELS[LUNA]:
            m = r["models"][name]
            for rule in RULES:
                x = m["rules"][rule]
                rules.append([lab, name, rule, x["K"], *pts(o[rule], n), f"{x['E_gain_observed_baseline']:.2f}", f"{x['P_ge_observed_observed_baseline']:.3f}",
                              f"{x['E_gain_random_baseline']:.2f}", f"{x['P_ge_observed_random_baseline']:.3f}"])
                if name == "fitted":
                    power.append([lab, rule, *pts(o[rule], n), *power_cells(x)])
            if s := o["SkillAdam"]:
                x = m["rules"]["SkillAdam"]
                adam.append([lab, name, f"{s['accepted']} ({s['reported_gain_points']:.1f} pts)", f"{x['E_accepts_cached']:.2f}", f"{x['E_reported_gain_points']:.1f}",
                             f"{x['P_reported_gain_ge_observed']:.3f}", f"{x['P_better_one_comparison']:.3f}",
                             ", ".join(f"+{g}: {p:.3f}" for g, p in x["P_gain_ge_accepted"].items()) or "-"])
        bo = r["aborted"]
        checks_rows.append([lab] + [c for rule in RULES for c in (f"{o[rule]['baseline']}/{n} vs " + " ".join(map(str, o[rule]["candidates"])),
                                                                   f"{r['rank_test'][rule]:.3f}")]
                           + [f"{bo['solved']}/{bo['verified']}; on {bo['common']} common tasks native {bo['native_solved_common']} vs aborted "
                              f"{bo['aborted_solved_common']} ({bo['native_1_aborted_0']} solved only natively, {bo['native_0_aborted_1']} only in the aborted run)"])
    gated = [label[b] for b in BENCHES if res[b]["observed"]["SkillAdam"]]
    tb2 = res["terminal-bench-2-1"]["null_executions"]["s_over_n"]
    parts = [f"{S.N_DRAWS} draws per task model and benchmark. With one seed execution per dev task, a model of the seed's own executions "
             "gives p_v in {0, 1} and G0 = 0, so the prior-free models pool every dev execution of the seed dev pool. Where g = 0 the P columns "
             "give P(G0 >= 1), the chance that selection alone would have kept a candidate.",
             "### Seed inputs\n\n" + md_table(["bench", "dev tasks", "seed executions per dev task", "seed dev draw", "runs using it", "test tasks",
                                               "seed test always / sometimes / never (3 runs)", "fitted prior (a, b)", "mean success",
                                               "mid-quantile of the seed dev draw (fitted)", "expected seed success, fitted posteriors",
                                               "dev executions of all versions (null models)", "seed dev executions", "seed test executions"], inputs_rows),
             "### What each run recorded\n\n" + md_table(["bench", "optimizer", "K / iterations", "baseline (starting seed draw)",
                                                          "candidate dev scores (tasks)", "reported dev gain g", "returned"], run_rows)
             + ("" if len(gated) == len(BENCHES) else "\n\nSkillAdam gated its edits on the training batch"
                + (f" except on {', '.join(gated)}" if gated else "") + ": it reports no dev gain to replay."),
             "### SkillOpt and GEPA rules under no effect\n\n" + md_table(["bench", "task model", "rule", "K", "starting draw", "g", "E[G0], starting draw",
                                                                            "P(G0 >= max(g,1)), starting draw", "E[G0], random seed draw",
                                                                            "P(G0 >= max(g,1)), random seed draw"], rules)]
    if adam:
        parts.append("### SkillAdam's gate under no effect (identical patches, outcomes reused as in the runs)\n\n"
                     + md_table(["bench", "task model", "accepted (summed gain)", "E[accepts], identical patches", "E[summed gain] (pts)",
                                 "P(summed gain >= observed) (TB2: > 0)", "P(better) per comparison", "P(gain >= accepted gain) per comparison"], adam))
    parts += ["### Model-free checks\n\n" + md_table(["bench", "SkillOpt scores (seed vs candidates)", "rank test P", "GEPA scores (seed vs children)",
                                                      "rank test P", "aborted run's seed dev draw (not a replicate)"], checks_rows),
              "### Power of the replay from the starting draw (fitted model; one candidate truly better, the others identical)\n\n"
              + md_table(["bench", "rule", "starting draw", "g"] + POWER_HEAD, power),
              "TB2 dev tasks over every evaluation of every version (successes / executions): " + ", ".join(f"{t} {v}" for t, v in tb2.items()) + ".",
              notes(res)]
    record(__spec__.name, "selection_luna", "\n\n".join(parts), runs(LUNA, ("aborted_seed_dev",)),
           what="The selection replay of mech_selection for GPT-6-Luna: the dev gain each optimizer's own rule reports when every candidate is "
                "identical to the seed, from the seed draw each run started at, against the gain the run reported; on TB2, where no candidate "
                "scored above the starting draw, how often selection alone would have kept one.")


def notes(res):
    """The SkillOpt run on AutoBench and the TB2 gate, written from the numbers."""
    r, t = res["automationbench"], res["terminal-bench-2-1"]
    o, ot = r["observed"]["SkillOpt"], t["observed"]["SkillOpt"]
    ps = ", ".join(f"{r['models'][m]['rules']['SkillOpt']['P_ge_observed_observed_baseline']:.3f} ({m})" for m in MODELS[LUNA])
    never = [k for k, v in t["null_executions"]["s_over_n"].items() if v.startswith("0/")]
    return ("Notes:\n\n"
            f"- SkillOpt on AutoBench: the kept candidate scored {max(o['candidates'])}/{r['dev_tasks']} against the starting draw "
            f"{o['baseline']}/{r['dev_tasks']} (+{o['gain']} task), its candidates {' '.join(map(str, o['candidates']))}; identical candidates report "
            f"at least +{max(o['gain'], 1)} from that draw with probability {ps}, and {r['rank_test']['SkillOpt']:.3f} by the rank test. The replay "
            f"would flag a candidate that truly solved two more dev tasks with probability "
            f"{r['models']['fitted']['rules']['SkillOpt']['power_observed_baseline']['2']:.3f}.\n"
            f"- TB2: the seed draw {ot['baseline']}/{t['dev_tasks']} failed {', '.join(never)}, which every version failed in every dev execution, and "
            f"solved the other {t['dev_tasks'] - len(never)}; under the null models P(G0 >= 1) from that draw is "
            f"{t['models']['null plug-in']['rules']['SkillOpt']['P_ge_observed_observed_baseline']:.0f}, against "
            f"{t['models']['fitted']['rules']['SkillOpt']['P_ge_observed_observed_baseline']:.3f} (SkillOpt) under the fitted model, which gives these "
            "tasks a nonzero rate.")


def record_devsize(res):
    rows = []
    for b in BENCHES:
        for prior in ("fitted", "moderate"):
            q = res[b]["models"][prior]["seed_draw_mid_quantile"]
            for size in SIZES:
                r = res[b]["devsize"][prior][str(size)]
                rows.append([b, prior, size, f"{q:.2f} -> {r['baseline_at_quantile']:.0f}/{size}",
                             *(f"{r[rule][k]:.1f}" for rule in RULES for k in ("E_gain_random_points", "E_gain_quantile_points")),
                             f"{r['MDE80_points']:.1f}", r["critical_net_tasks"]])
    record(__spec__.name, "devsize", f"{S.N_DRAWS} draws per task model, benchmark and dev size.\n\n"
           + md_table(["bench", "prior", "dev size", "seed-draw quantile -> baseline", "SkillOpt E[gain] random (pts)", "SkillOpt E[gain] at quantile (pts)",
                       "GEPA E[gain] random (pts)", "GEPA E[gain] at quantile (pts)", "MDE80 (pts)", "critical net difference (tasks)"], rows), runs(DS),
           what="How the gain a success-only selector announces under no effect, and the smallest effect a single-run dev comparison detects, "
                "change with the size of the dev panel (SkillOpt's best of four, GEPA's best of seven), in points of the dev panel.")


def record_assumptions(res):
    lrt, fit, rules, single, sizes = [], [], [], [], []
    for b in BENCHES:
        r, c = res[b], res[b]["checks"]
        lrt += [[b, x["grouping"], x["groups"], f"{x['LR']:.2f}", x["df"], f"{x['p_chi2']:.3f}"] for x in c["lrt"]]
        fit.append([b, "({:.3f}, {:.3f})".format(*c["prior"]), " ".join(f"({m}, {s}) {o:.0f}/{e:.2f}" for (m, s), o, e in zip(c["cells"], c["obs"], c["exp"])),
                    f"{c['pearson']:.2f}", f"{c['tarone_Z']:.1f}", f"{c['var_s_n3_obs']:.3f} / {c['var_s_n3_pred']:.3f}", f"{c['unanimous_dev']:.2f}"])
        for kind, m in r["models"].items():
            rules += [[b, kind, rule] + [f"{m['rules'][rule][k]:.{d}f}" for k, d in (("E_gain_observed_baseline", 2), ("P_ge_observed_observed_baseline", 2),
                                                                                    ("E_gain_random_baseline", 2), ("P_ge_observed_random_baseline", 2))]
                      for rule in RULES]
            x, ab = m["rules"]["SkillAdam"], m["accept_if_better"]
            single.append([b, kind, f"{x['E_reported_gain_points']:.1f}", f"{x['P_reported_gain_ge_observed']:.2f}", f"{ab['P_accept_no_effect']:.3f}",
                           f"{ab['MDE80_points']:.1f}"])
        sizes += [[b, kind, size, *(f"{r['devsize'][kind][str(size)][rule]['E_gain_random_points']:.1f}" for rule in RULES),
                   f"{r['devsize'][kind][str(size)]['MDE80_points']:.1f}"] for kind in ("plug-in", "Jeffreys") for size in SIZES]
    numbers = "\n\n".join([
        "### One Beta across tasks: likelihood-ratio tests against one Beta per group (seed's dev and test tasks)\n\n"
        + md_table(["bench", "grouping", "groups", "LR", "df", "p (chi-square)"], lrt),
        "### Fit of the pooled prior to the counts of solved executions\n\n"
        + md_table(["bench", "prior (a, b)", "(executions, solved) observed/expected tasks", "Pearson X2", "Tarone Z",
                    "var(solved), three executions: observed / predicted", "share of dev tasks solved in all or no executions"], fit),
        "### Replay under each task model (SkillOpt, GEPA; K, starting draws and gains as in mech_selection)\n\n"
        + md_table(["bench", "task model", "rule", "E[G0], starting draw", "P(G0 >= g), starting draw", "E[G0], random seed draw",
                    "P(G0 >= g), random seed draw"], rules),
        "### SkillAdam's summed gain and one dev comparison under each task model\n\n"
        + md_table(["bench", "task model", "SkillAdam E[summed gain] (pts)", "SkillAdam P(summed gain >= observed)", "P(accept), identical candidate, one comparison",
                    "MDE80, one comparison (pts)"], single),
        "### Dev-set size with the added tasks resampled from the seed's dev and test tasks (random baseline)\n\n"
        + md_table(["bench", "task model", "dev size", "SkillOpt E[gain] (pts), best of 4", "GEPA E[gain] (pts), best of 7", "MDE80 (pts)"], sizes)])
    record(__spec__.name, "assumptions", numbers, runs(DS),
           what="How much the selection replay rests on its task model: tests of one Beta prior across the tasks of a benchmark, the fit of the "
                "pooled prior, and the replay, a single dev comparison and the dev-set-size curve under task models without the pooled prior "
                "(DeepSeek-V4.1-Flash).")


def main():
    res, panel = {DS: {}, LUNA: {}}, []
    for model in (DS, LUNA):
        for b in BENCHES:
            res[model][b], rows = analyse(model, b)
            panel += rows
    keep = ("dev_tasks", "dev_executions_per_task", "test_tasks", "prior_fitted", "mean_success", "seed_dev_draws", "test_avg3", "expected_seed_success")
    selection = {b: {**{k: r[k] for k in keep}, "observed": {o: {k: v[k] for k in ("K", "baseline", "gain")} if o in RULES else v for o, v in r["observed"].items()},
                     "priors": {p: {"prior": r["priors"][p], **r["models"][p]} for p in ("fitted", "moderate")}} for b, r in res[DS].items()}
    (OUT / "selection.json").write_text(json.dumps(selection, indent=1))
    (OUT / "replay.json").write_text(json.dumps(res, indent=1))
    rnd = lambda row: {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
    (OUT / "dev2test.json").write_text(json.dumps({"replays": S.N_DRAWS, "prior": "fitted", "panel_a": [rnd(x) for x in panel]}, indent=1))
    record_selection(res[DS])
    record_luna(res[LUNA])
    record_devsize(res[DS])
    record_assumptions(res[DS])


if __name__ == "__main__":
    main()
