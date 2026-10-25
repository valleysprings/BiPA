"""Maximum-likelihood estimate of the deepseek-flash model-version shift (old aeb564... -> new 1017 e72d3c...) on the seed skill.

Old-model data: the seed's executions that src/analysis/selection.py uses (dev: executions(seed_files(bench)[0]); test: its three
test runs, seed_files(bench)[1]; checked by check_inputs).  New-model data: the seed calibration runs of run_path_test.py
(out/deepseek-flash/seed__<bench>/results/evaluations/{dev,test}/4fd5ca441c55-*/evaluation.json), plus any --extra-new evaluation files.

Model.  Per benchmark the beta-binomial task model of selection.py is fitted to the old executions (fit_prior on dev+test pooled:
Beta(a, b), mean m, concentration c = a + b).  Per split s, the old baseline mean m_s is the ML mean of that split's old counts with c held
fixed.  The new model's tasks are Beta(m_s' c, (1 - m_s') c) with logit m_s' = logit m_s + delta: one shift of the logit of the mean
success, concentration fixed; delta is fitted by maximum likelihood to the new-model counts (per split, and one delta for dev+test).
Points = 100 (m_s' - m_s).  LR = 2 [l(delta_hat) - l(0)], p from chi-square(1).
Sensitivity (paired): each task keeps its own old posterior Beta(a + k, b + n - k) and its success probability p becomes
sigmoid(logit p + delta); this uses the task identity (same tasks on both models), which the marginal model ignores.

  python -m aux.path_test.estimate_gap              # needs the calibration runs; prints one block per benchmark
  python -m aux.path_test.estimate_gap --selftest   # synthetic counts with known delta; no data needed
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import optimize, special, stats

from src import layout
from src.analysis import selection as S

PATH_TEST = layout.ARTIFACTS / "path-test"
os.environ.setdefault("MPLCONFIGDIR", str(layout.OUT / ".mplconfig"))

BENCHES = ("tau3-bench", "automationbench", "terminal-bench-2-1")
SEED = "4fd5ca441c55"
NEW_FINGERPRINT = "e72d3c7f663d3e1383b00b3884cb4501"
OLD_FINGERPRINT = "aeb56401ca74e127821c4f9126dcb669"
QUAD = (np.arange(4000) + 0.5) / 4000  # equal-weight quantile quadrature (handles the U-shaped Beta's endpoint mass)


def logit(p):
    return math.log(p / (1 - p))


def betabinom_ll(counts, m, c):
    a, b = m * c, (1 - m) * c
    return float(sum(special.gammaln(k + a) + special.gammaln(n - k + b) - special.gammaln(n + a + b)
                     - special.gammaln(a) - special.gammaln(b) + special.gammaln(a + b) for k, n in counts))


def split_mean(counts, c):
    """ML mean of one split's old counts with the concentration held at c."""
    res = optimize.minimize_scalar(lambda x: -betabinom_ll(counts, special.expit(x), c), bounds=(-9, 9), method="bounded")
    return float(special.expit(res.x))


def fit_shift(loglik):
    """delta_hat, l(delta_hat), LR statistic, p-value for H0 delta = 0."""
    res = optimize.minimize_scalar(lambda d: -loglik(d), bounds=(-6, 6), method="bounded", options={"xatol": 1e-5})
    best, null = -res.fun, loglik(0.0)
    lr = max(0.0, 2 * (best - null))
    return float(res.x), best, lr, float(stats.chi2.sf(lr, 1))


def marginal(old_split, new_split, c):
    """{split: (m_s, delta, points, LR, p)} and the joint delta over the splits present in new_split."""
    base = {s: split_mean(list(old_split[s].values()), c) for s in new_split}
    def ll(s):
        counts = list(new_split[s].values())
        return lambda d: betabinom_ll(counts, special.expit(logit(base[s]) + d), c)
    out = {}
    for s in new_split:
        d, _, lr, p = fit_shift(ll(s))
        out[s] = (base[s], d, 100 * (special.expit(logit(base[s]) + d) - base[s]), lr, p)
    d, _, lr, p = fit_shift(lambda x: sum(ll(s)(x) for s in new_split))
    out["dev+test"] = (None, d, None, lr, p)
    return out


def paired(old_split, new_split, a, b):
    """Same as marginal, with each task's own old posterior (task identity kept)."""
    grids = {}
    for s, tasks in new_split.items():
        rows = []
        for task, (k, n) in tasks.items():
            ko, no = old_split[s].get(task, (0, 0))
            p = stats.beta.ppf(QUAD, a + ko, b + no - ko)
            rows.append((np.clip(p, 1e-12, 1 - 1e-12), k, n))
        grids[s] = rows
    def ll(s):
        def f(d):
            total = 0.0
            for p, k, n in grids[s]:
                q = special.expit(special.logit(p) + d)
                total += math.log(max(np.mean(q ** k * (1 - q) ** (n - k)), 1e-300)) + math.log(math.comb(n, k))
            return total
        return f
    out = {}
    for s in new_split:
        d, _, lr, p = fit_shift(ll(s))
        before = np.mean([g[0].mean() for g in grids[s]])
        after = np.mean([special.expit(special.logit(g[0]) + d).mean() for g in grids[s]])
        out[s] = (before, d, 100 * (after - before), lr, p)
    d, _, lr, p = fit_shift(lambda x: sum(ll(s)(x) for s in new_split))
    out["dev+test"] = (None, d, None, lr, p)
    return out


def counts(executions):
    return {task: (sum(v), len(v)) for task, v in executions.items()}


def fingerprints(files, limit=10):
    seen = {}
    for path in files:
        for e in json.load(open(path))["episodes"][:limit]:
            for r in e.get("requests", [])[:1]:
                try:
                    fp = (json.load(open(r["record_path"])).get("response") or {}).get("system_fingerprint")
                except (OSError, ValueError, KeyError):
                    fp = "unreadable"
                seen[fp] = seen.get(fp, 0) + 1
    return seen


def show(bench, prior, rows_m, rows_p, n_new):
    a, b = prior
    print(f"== {bench}: old prior Beta({a:.3f}, {b:.3f}) mean {a/(a+b):.3f} concentration {a+b:.3f}; new-model executions {n_new}")
    for name, rows in (("marginal (asked)", rows_m), ("paired (sensitivity)", rows_p)):
        for s, (base, d, pts, lr, p) in rows.items():
            head = f"   {name:<21} {s:<8} delta {d:+.3f} logit"
            body = f"  = {pts:+.1f} points from {100*base:.1f}%" if pts is not None else "  (one shift for both splits)"
            print(f"{head}{body};  LR {lr:.2f}, p {p:.3g} -> {'differs from 0' if p < 0.05 else 'not distinguishable from 0'}")


def real(extra_new):
    status = 0
    for bench in BENCHES:
        dev_files, test_files = S.seed_files(bench)
        dev, test = S.executions(dev_files), S.executions(test_files)
        S.check_inputs(bench, dev, test)
        a, b = S.fit_prior([(sum(v), len(v)) for v in list(dev.values()) + list(test.values())])
        root = PATH_TEST / "out/deepseek-flash" / f"seed__{bench}/results/evaluations"
        new_files = {s: sorted(glob.glob(str(root / s / f"{SEED}-*/evaluation.json"))) for s in ("dev", "test")}
        for item in extra_new:  # BENCH=PATH
            name, _, path = item.partition("=")
            data = json.load(open(path))
            if name != bench:
                continue
            if not data.get("version_id", "").startswith(SEED) or data.get("split") not in new_files:
                raise SystemExit(f"--extra-new {item}: not a seed dev/test evaluation")
            new_files[data["split"]].append(path)
        new_split = {s: counts(S.executions(f)) for s, f in new_files.items() if f}
        if not new_split:
            print(f"== {bench}: no new-model calibration runs under {root} yet; run run_path_test.py --model deepseek-flash --execute first")
            status = 1
            continue
        fp = fingerprints([f for fs in new_files.values() for f in fs])
        if set(fp) - {NEW_FINGERPRINT}:
            print(f"   WARNING {bench}: new-model files carry fingerprints {fp}; expected only {NEW_FINGERPRINT}")
        old_split = {"dev": counts(dev), "test": counts(test)}
        show(bench, (a, b), marginal(old_split, new_split, a + b), paired(old_split, new_split, a, b),
             sum(n for s in new_split.values() for _, n in s.values()))
    return status


def shifted(p, a, b, delta, generator):
    """New-model task probabilities: 'mean' = the asked model (the prior's mean logit shifted by delta, concentration fixed; each task
    keeps its quantile), 'task' = every task's own logit shifted by delta."""
    if generator == "task":
        return special.expit(special.logit(p) + delta)
    c, m = a + b, a / (a + b)
    m2 = special.expit(logit(m) + delta)
    return stats.beta.ppf(stats.beta.cdf(p, a, b), m2 * c, (1 - m2) * c)


def true_points(a, b, delta, generator):
    p = np.clip(stats.beta.ppf(QUAD, a, b), 1e-12, 1 - 1e-12)
    return 100 * float(np.mean(shifted(p, a, b, delta, generator)) - a / (a + b))


def synthetic(rng, a, b, delta, generator, n_dev=30, n_test=75, old_reps=3, new_reps=1):
    """Tasks p ~ Beta(a, b); old counts with old_reps executions; new counts with one execution at the shifted probabilities."""
    p = np.clip(rng.beta(a, b, n_dev + n_test), 1e-9, 1 - 1e-9)
    q = np.clip(shifted(p, a, b, delta, generator), 0, 1)
    names = [f"t{i}" for i in range(n_dev + n_test)]
    old = {t: [int(x) for x in rng.random(old_reps) < pi] for t, pi in zip(names, p)}
    new = {t: (int((rng.random(new_reps) < qi).sum()), new_reps) for t, qi in zip(names, q)}
    split = lambda d: {"dev": {t: d[t] for t in names[:n_dev]}, "test": {t: d[t] for t in names[n_dev:]}}  # noqa: E731
    return split(old), split(new)


def selftest(reps):
    rng = np.random.default_rng(20261024)
    # the tau3 seed prior of the paper's inputs: U-shaped, most tasks nearly always or never solved
    a0, b0 = 0.548, 0.193
    print(f"selftest: tasks ~ Beta({a0}, {b0}) (tau3 seed prior); 30 dev + 75 test tasks, 3 old executions each, 1 new execution; "
          f"{reps} replicates per cell.  Columns: mean delta_hat (sd) of the joint dev+test fit, mean test-split points vs the truth, "
          f"P(LR rejects delta=0 at 0.05)")
    failures = 0
    for generator in ("mean", "task"):
        for true in (0.0, -0.8, 0.8):
            est = {"marginal": [], "paired": []}
            for _ in range(reps):
                old, new = synthetic(rng, a0, b0, true, generator)
                old_split = {s: {t: (sum(v), len(v)) for t, v in d.items()} for s, d in old.items()}
                a, b = S.fit_prior(list(old_split["dev"].values()) + list(old_split["test"].values()))
                for name, rows in (("marginal", marginal(old_split, new, a + b)), ("paired", paired(old_split, new, a, b))):
                    est[name].append((rows["dev+test"][1], rows["test"][2], rows["dev+test"][4]))
            truth = true_points(a0, b0, true, generator)
            for name, rows in est.items():
                d, pts, p = (np.array(x) for x in zip(*rows))
                print(f"   generator {generator:<4} true delta {true:+.1f} ({truth:+5.1f} pts)  {name:<8}: delta_hat {d.mean():+.3f} "
                      f"(sd {d.std():.3f}), test points {pts.mean():+5.1f}, P(reject) {np.mean(p < 0.05):.2f}")
                own = (generator == "mean") == (name == "marginal")  # each estimator under its own generating model
                if own and abs(d.mean() - true) > 3 * d.std() / math.sqrt(reps) + 0.1:
                    failures += 1
                    print(f"      FAIL: {name} is off its own model's delta")
    print(f"selftest {'passed' if not failures else 'FAILED'}: each estimator recovers delta under its own model")
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--reps", type=int, default=40, help="selftest replicates per true delta")
    parser.add_argument("--extra-new", nargs="*", default=[], help="BENCH=PATH: extra new-model (1017) seed evaluation.json files to include")
    args = parser.parse_args(argv)
    return selftest(args.reps) if args.selftest else real(args.extra_new)


if __name__ == "__main__":
    sys.exit(main())
