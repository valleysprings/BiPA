"""(d, part 2) Replay of the paper's own selection model (src/analysis/selection.py) on the from-scratch runs.

Model (selection.py docstring): per benchmark a Beta(a, b) prior over per-task success probability, fitted by beta-binomial maximum likelihood to the SEED's distinct
executions of the dev and test tasks; each dev task's probability is the posterior Beta(a + s, b + n - s) from its own distinct dev executions; a candidate identical to
the seed draws fresh outcomes from these probabilities; best-of-K (kept only if strictly above the baseline; GEPA: ties go to the seed) announces
gain = max(best - baseline, 0), measured against the run's starting score fixed at its observed value ('observed baseline') or drawn like a candidate ('random baseline').

Adaptation to the from-scratch runs (this is the Luna replay of aux/selection_replay/luna.py with the 3-word seed cf7b932190ed in place of the hand-written seed 4fd5ca441c55):
  * seed dev executions  = every dev evaluation of cf7b932190ed in the benchmark's two from-scratch runs (SkillOpt run, GEPA run), distinct by request-id signature;
  * seed test executions = the 3-word seed's test evaluations: AutoBench 3 reps (test-reps/automationbench__native-seed__fromscratch), tau3 one per run (2);
  * K and baseline: SkillOpt K = total_accepts + total_rejects (4), baseline = its own seed dev evaluation (history.json starting current_score; the summary leaves it null),
    GEPA K = len(val_scores) - 1, baseline = val_scores[0].
  Sensitivity: moderate prior Beta(4m, 4(1-m)) as in selection.py; and the 'null' models of the Luna replay (plug-in s/n and Bayesian bootstrap Beta(s, n-s)) estimated from
  every dev evaluation (any version) of the benchmark's two runs, which treat ALL candidates as draws of one distribution (conservative: includes the returned skill's real gain).
Run: python -m aux.three_word_seed.selection_replay
"""
import glob, json
from pathlib import Path
import numpy as np
from src.analysis import selection as S

from aux.selection_replay.luna import null_probabilities
from src import layout

EXE = layout.EXECUTION / "gpt-6-luna"
OUT = layout.OUT / "three_word_seed"
SEED3 = "cf7b932190ed"
DRAWS = 200_000
CHUNK = 25_000


def seed_files(bench):
    dev = sorted(glob.glob(str(EXE / f"*/{bench}/fromscratch/results/evaluations/dev/{SEED3}-*/evaluation.json")))
    if bench == "automationbench":
        rr = json.load(open(EXE / "test-reps/automationbench__native-seed__fromscratch/results.json"))
        test = [p for r in rr["runs"] for p in r["evaluation_paths"]]
    else:
        test = sorted(glob.glob(str(EXE / f"*/{bench}/fromscratch/results/evaluations/test/{SEED3}-*/evaluation.json")))
    return dev, test


def all_dev_files(bench):
    return sorted(glob.glob(str(EXE / f"*/{bench}/fromscratch/results/evaluations/dev/*/evaluation.json")))


def chunked(fn, draws=DRAWS, chunk=CHUNK):
    parts = [fn(chunk) for _ in range(draws // chunk)]
    return np.concatenate(parts)


def summarize(g, n_dev, observed_tasks):
    pts = 100 * g / n_dev
    return dict(mean=float(pts.mean()), p05=float(np.percentile(pts, 5)), p50=float(np.percentile(pts, 50)), p95=float(np.percentile(pts, 95)),
                P_ge_observed=float((g >= max(observed_tasks, 1)).mean()), P_zero=float((g == 0).mean()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    collect = json.load(open(OUT / "dev_collect.json"))


    res = {}
    for bi, bench in enumerate(("automationbench", "tau3-bench")):
        dev_files, test_files = seed_files(bench)
        dev, test = S.executions(dev_files), S.executions(test_files)
        n_dev = len(dev)
        pooled = [(sum(v), len(v)) for v in list(dev.values()) + list(test.values())]
        a, b = S.fit_prior(pooled)
        mean = sum(k for k, _ in pooled) / sum(n for _, n in pooled)
        null_exec = S.executions(all_dev_files(bench))
        tasks = sorted(dev)
        info = dict(n_dev=n_dev, seed_dev_files=[Path(p).parent.name for p in dev_files],
                    seed_dev_executions_per_task={str(k): sum(1 for v in dev.values() if len(v) == k) for k in sorted({len(v) for v in dev.values()})},
                    seed_dev_distinct_executions=sum(map(len, dev.values())), seed_dev_solved_by_task_first_exec=sum(v[0] for v in dev.values()),
                    seed_test_files=[Path(p).parent.name for p in test_files], seed_test_distinct_executions=sum(map(len, test.values())),
                    prior_fitted=[a, b], mean_success=mean, prior_mean=a / (a + b), prior_concentration=a + b,
                    posterior_mean_dev_success_pct=S.posterior_mean(dev, a, b), posterior_mean_test_success_pct=S.posterior_mean(test, a, b),
                    observed_seed_dev_pct=100 * sum(v[0] for v in dev.values()) / n_dev, rules={})
        for opt in ("skillopt", "gepa"):
            c = collect[f"{opt}|{bench}"]
            K = c["K"]
            base = c["seed_dev_solved"][0] if opt == "skillopt" else c["baseline_tasks_val_scores0"]
            returned = c["returned_dev_solved"][0]
            gain_obs = max(returned - base, 0)
            cand = c.get("candidate_dev_tasks") or c["val_scores_tasks"][1:]
            rule = dict(K=K, baseline_tasks=base, returned_dev_tasks=returned, observed_gain_tasks=gain_obs, observed_gain_points=100 * gain_obs / n_dev, candidate_dev_tasks=cand, models={})
            for j, name in enumerate(("fitted", "moderate", "null plug-in", "null Bayesian bootstrap")):
                S.RNG = np.random.default_rng([20261024, bi, j, 0 if opt == "skillopt" else 1])
                if name in ("fitted", "moderate"):
                    prior = (a, b) if name == "fitted" else (S.MODERATE * mean, S.MODERATE * (1 - mean))
                    mk = lambda n: S.task_probabilities(dev, prior, n_dev, n)
                else:
                    rng = np.random.default_rng([20261024, bi, j, 9])
                    mk = lambda n: null_probabilities(name, null_exec, tasks, n, rng)
                gfix, grand = [], []
                for _ in range(DRAWS // CHUNK):
                    p = mk(CHUNK)
                    gfix.append(S.best_of(p, K, base))
                    grand.append(S.best_of(p, K, S.outcomes(p)))
                gfix, grand = np.concatenate(gfix), np.concatenate(grand)
                rule["models"][name] = dict(observed_baseline=summarize(gfix, n_dev, gain_obs), random_baseline=summarize(grand, n_dev, gain_obs))
            # K sensitivity for GEPA AutoBench (a 17th dev evaluation 0b4175610f30, 28 of 30 verified, exists outside val_scores)
            S.RNG = np.random.default_rng([20261024, bi, 77, 0 if opt == "skillopt" else 1])
            rule["fitted_best_of_K_range"] = {}
            for kk in sorted({max(K - 1, 1), K, K + 1}):
                g = np.concatenate([S.best_of(S.task_probabilities(dev, (a, b), n_dev, CHUNK), kk, base) for _ in range(DRAWS // CHUNK)])
                rule["fitted_best_of_K_range"][str(kk)] = summarize(g, n_dev, gain_obs)
            info["rules"][opt] = rule
        res[bench] = info
    json.dump(res, open(OUT / "selection_replay.json", "w"), indent=1)
    for bench, info in res.items():
        print(bench, {k: info[k] for k in ("n_dev", "seed_dev_executions_per_task", "seed_dev_distinct_executions", "seed_test_distinct_executions", "prior_fitted", "mean_success", "observed_seed_dev_pct", "posterior_mean_dev_success_pct", "posterior_mean_test_success_pct")})
        for opt, r in info["rules"].items():
            print("  ", opt, "K", r["K"], "base", r["baseline_tasks"], "returned", r["returned_dev_tasks"], "observed gain pts %.1f" % r["observed_gain_points"], "cands", r["candidate_dev_tasks"])
            for m, v in r["models"].items():
                o, rd = v["observed_baseline"], v["random_baseline"]
                print("      %-24s obs-base mean %.1f [%.1f, %.1f] P>=obs %.3f | rand-base mean %.1f [%.1f, %.1f] P>=obs %.3f" % (m, o["mean"], o["p05"], o["p95"], o["P_ge_observed"], rd["mean"], rd["p05"], rd["p95"], rd["P_ge_observed"]))
            print("      K sens:", {k: (round(v['mean'], 1), round(v['p05'], 1), round(v['p95'], 1), round(v['P_ge_observed'], 3)) for k, v in r["fitted_best_of_K_range"].items()})


if __name__ == "__main__":
    main()
