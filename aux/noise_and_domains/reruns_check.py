"""Model-free checks that need neither the Beta prior (A2) nor per-task probabilities.

(1) Extra seed executions: every seed-version (4fd5ca441c55) test evaluation on the pre-upgrade model (run tags up to native-20261012; the
    *-1017 runs are a different model release) -- how many distinct executions they add beyond the paper's three runs.
(2) Real-rerun selection: the seed's own distinct full-panel runs used as identical candidates. For every choice of a baseline run and
    k candidate runs from the others, gain = max(0, best candidate - baseline) (SkillOpt / GEPA rule).  Also the gains the seed's other real
    dev draws show over the starting draw every optimizer run used.
(3) A1 (no run-level shock): Cochran's Q over the seed's runs on tasks with an outcome in every run (exact permutation p-value: per task,
    permute its outcomes across runs).  Also the SD of the run totals against the A1 (within-task Bernoulli) prediction.

Run: python -m aux.noise_and_domains.reruns_check
"""
import itertools
import json
import os

import numpy as np

from src.analysis import selection as S

from src import layout

OUT = layout.OUT / "noise_and_domains"
E = S.EXECUTION
# every seed-version test evaluation found by `find ... -path '*/evaluations/test/4fd5ca441c55-*'`, main file and retries
EXTRA_TEST = {
    "tau3-bench": ["native-seed/tau3-bench/native-20261003-trace-final", "gepa/tau3-bench/native-20261004-upstream-batch",
                   "gepa/tau3-bench/native-20261008-https", "skillopt/tau3-bench/native-20261004-upstream-batch",
                   "skillopt/tau3-bench/native-20261008-https", "skilladam/tau3-bench/native-20261005", "skilladam/tau3-bench/native-20261008-https",
                   "gepa/tau3-bench/ood-from-gpt-6-luna", "skillopt/tau3-bench/ood-from-gpt-6-luna"],
    "automationbench": ["native-seed/automationbench/native-20261003-trace-final", "native-seed/automationbench/ood-bench-seed",
                        "native-seed/automationbench/trace-remeasurement-20261003", "gepa/automationbench/native-20261004-upstream-batch",
                        "gepa/automationbench/native-20261008-https", "skillopt/automationbench/native-20261004-upstream-batch",
                        "skillopt/automationbench/native-20261008-https", "skilladam/automationbench/native-20261005",
                        "skilladam/automationbench/native-20261008-https"],
    "terminal-bench-2-1": ["native-seed/terminal-bench-2-1/native-20261003-trace-final", "native-seed/terminal-bench-2-1/ood-bench-seed",
                           "native-seed/terminal-bench-2-1/trace-remeasurement-20261003", "gepa/terminal-bench-2-1/native-20261004-upstream-batch",
                           "gepa/terminal-bench-2-1/native-20261008-https", "skillopt/terminal-bench-2-1/native-20261004-upstream-batch",
                           "skillopt/terminal-bench-2-1/native-20261008-https", "skilladam/terminal-bench-2-1/native-20261005",
                           "skilladam/terminal-bench-2-1/native-20261008-https", "gepa/terminal-bench-2-1/ood-from-gpt-6-luna",
                           "skillopt/terminal-bench-2-1/ood-from-gpt-6-luna"],
}
PAPER_TEST = {b: [f"native-seed/{b}/{r}" for r in S.SEED_TEST_RUNS] for b, _ in S.BENCHES}


def eval_files(run):
    base = E / run / "results/evaluations/test"
    out = []
    for d in sorted(base.glob("4fd5ca441c55-*")):
        out += [str(d / "evaluation.json")] + sorted(str(p) for p in d.glob("retry-*/evaluation.json"))
    return [f for f in out if os.path.exists(f)]


def episodes(files):
    """{(task, signature): success} over files."""
    out = {}
    for f in files:
        for e in json.load(open(f))["episodes"]:
            if S.verified(e):
                out[(e["task_id"], S.signature(e))] = int(bool(e["success"]))
    return out


def run_outcomes(run):
    """One run's per-task outcome (main file, then retries filling unverified tasks): {task: 0/1}."""
    out = {}
    for f in eval_files(run):
        for e in json.load(open(f))["episodes"]:
            if S.verified(e) and e["task_id"] not in out:
                out[e["task_id"]] = (int(bool(e["success"])), S.signature(e))
    return out


def cochran(matrix, rng, perms=20000):
    """Cochran's Q of a (tasks, runs) 0/1 matrix and its exact permutation p-value (outcomes permuted across runs within each task)."""
    def q(m):
        k = m.shape[1]
        col, row = m.sum(axis=0), m.sum(axis=1)
        den = k * row.sum() - (row ** 2).sum()
        return (k - 1) * (k * (col ** 2).sum() - row.sum() ** 2) / den if den > 0 else 0.0
    q0 = q(matrix)
    idx = np.argsort(rng.random((perms, *matrix.shape)), axis=2)
    perm = np.take_along_axis(np.broadcast_to(matrix, (perms, *matrix.shape)), idx, axis=2)
    k = matrix.shape[1]
    col = perm.sum(axis=1)
    row = matrix.sum(axis=1)
    den = k * row.sum() - (row ** 2).sum()
    qs = (k - 1) * (k * (col ** 2).sum(axis=1) - row.sum() ** 2) / den
    return float(q0), float((qs >= q0 - 1e-9).mean())


def real_best_of(totals, k):
    gains = []
    for b in range(len(totals)):
        others = [t for i, t in enumerate(totals) if i != b]
        for c in itertools.combinations(others, k):
            gains.append(max(0.0, max(c) - totals[b]))
    return gains


DISTINCT = {   # distinct pre-upgrade full test runs of the seed (distinct_runs.py; copies in optimizer dirs collapsed, fingerprint aeb564)
    "tau3-bench": ["native-seed/tau3-bench/native-20261003", "native-seed/tau3-bench/rep-1", "native-seed/tau3-bench/rep-2",
                   "gepa/tau3-bench/native-20261004-upstream-batch"],
    "automationbench": ["native-seed/automationbench/native-20261003", "native-seed/automationbench/rep-1", "native-seed/automationbench/rep-2",
                        "native-seed/automationbench/ood-bench-seed", "gepa/automationbench/native-20261004-upstream-batch"],
    "terminal-bench-2-1": ["native-seed/terminal-bench-2-1/native-20261003", "native-seed/terminal-bench-2-1/rep-1",
                           "native-seed/terminal-bench-2-1/rep-2", "native-seed/terminal-bench-2-1/ood-bench-seed",
                           "gepa/terminal-bench-2-1/native-20261004-upstream-batch"],
}


def model_p(model, s, n, prior, draws, rng):
    if model == "M0":
        return rng.beta(prior[0] + s, prior[1] + n - s, size=(draws, len(s)))
    if model == "M1":
        return np.broadcast_to(s / n, (draws, len(s)))
    if model == "M2j":
        return rng.beta(0.5 + s, 0.5 + n - s, size=(draws, len(s)))


def model_best_of(p, k, rng):
    base = (rng.random(p.shape) < p).sum(axis=1)
    best = np.max(np.stack([(rng.random(p.shape) < p).sum(axis=1) for _ in range(k)], axis=1), axis=1)
    return float(np.maximum(best - base, 0).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261020)
    result = {}
    for bench, _ in S.BENCHES:
        r = {"runs": DISTINCT[bench]}
        runs = {run: run_outcomes(run) for run in DISTINCT[bench]}
        common = sorted(t for t in runs[DISTINCT[bench][0]] if all(t in o for o in runs.values()))
        m = np.array([[runs[run][t][0] for run in DISTINCT[bench]] for t in common])
        R3 = m[:, :3]
        r["common_tasks"] = len(common)
        r["run_success_pct"] = (100 * m.mean(axis=0)).round(1).tolist()
        r["always_sometimes_never_allruns"] = [int((m.sum(1) == m.shape[1]).sum()), int(((m.sum(1) > 0) & (m.sum(1) < m.shape[1])).sum()),
                                               int((m.sum(1) == 0).sum())]
        for name, mat in (("paper3", R3), ("all", m)):
            q, p = cochran(mat, rng)
            within = mat.var(axis=1, ddof=1).sum()
            r[f"A1_{name}"] = {"cochran_Q": q, "df": mat.shape[1] - 1, "p_perm": p,
                               "sd_run_pct_obs": float(100 * mat.mean(axis=0).std(ddof=1)),
                               "sd_run_pct_A1": float(100 * np.sqrt(within) / len(common))}
        # out of sample: models fitted on the paper's three runs (+ dev executions for the M0 prior, as the paper) predict runs 4, 5
        dev_files, test_files = S.seed_files(bench)
        dev_files = [f for f in dev_files if "1017" not in f and "lodo" not in f]
        dev, test = S.executions(dev_files), S.executions(test_files)
        prior = S.fit_prior([(sum(v), len(v)) for v in list(dev.values()) + list(test.values())])
        s3, n3 = R3.sum(1).astype(float), np.full(len(common), 3.0)
        r["predict_new_runs"] = {}
        r["best_of_k"] = {"real_all_runs": {}, "real_paper3": {}}
        totals = m.sum(axis=0).tolist()
        for k in range(1, m.shape[1]):
            r["best_of_k"]["real_all_runs"][str(k)] = float(np.mean(real_best_of(totals, k)))
        for k in (1, 2):
            r["best_of_k"]["real_paper3"][str(k)] = float(np.mean(real_best_of(totals[:3], k)))
        for model in ("M0", "M1", "M2j"):
            P = model_p(model, s3, n3, prior, 50_000, rng)
            tot = (rng.random(P.shape) < P).sum(axis=1)
            r["predict_new_runs"][model] = {"mean": float(tot.mean()), "sd": float(tot.std()),
                                            "new_runs": [{"run": DISTINCT[bench][i], "solved": int(totals[i]),
                                                          "mid_quantile": S.mid_quantile(tot, totals[i])} for i in range(3, m.shape[1])]}
            r["best_of_k"][model] = {str(k): model_best_of(P, k, rng) for k in range(1, m.shape[1])}
        # dev draws (pre-upgrade): the seed's own reruns against the starting draw
        n_dev = {"terminal-bench-2-1": 9}.get(bench, 30)
        draws = S.dev_draws(dev_files, n_dev)
        r["dev_draws"] = [f"{d['solved']}/{d['verified']}" for d in draws]
        dtot = [d["solved"] * n_dev / d["verified"] for d in draws]
        r["dev_other_draws_minus_start"] = [round(x - min(dtot), 2) for x in dtot][1:]
        r["dev_real_best_of"] = {str(k): float(np.mean(real_best_of(dtot, k))) for k in range(1, len(dtot))}
        dm = {}
        for f in dev_files:
            eps = json.load(open(f))["episodes"]
            if len(eps) == n_dev:
                key = frozenset((e["task_id"], S.signature(e)) for e in eps if S.verified(e))
                dm[key] = {e["task_id"]: int(bool(e["success"])) for e in eps if S.verified(e)}
        mats = list(dm.values())
        cd = [t for t in mats[0] if all(t in x for x in mats)]
        q, p = cochran(np.array([[x[t] for x in mats] for t in cd]), rng)
        r["A1_dev"] = {"cochran_Q": q, "df": len(mats) - 1, "p_perm": p, "tasks": len(cd)}
        result[bench] = r
        print(bench, json.dumps(r, indent=1), flush=True)
    (OUT / "reruns_check.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
