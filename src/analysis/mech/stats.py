"""Paired statistics behind paper 1's claims (kept in the repro package, not printed).

Role: recompute from the raw evaluation.json files, for each benchmark and each of GEPA, SkillOpt, SkillAdam and No skill, the paired comparison against the Seed skill row.
Data and task sets are IDENTICAL to Table 1: the loaders `rep_files`, `rows_of`, `mean_rows` and `geo` of analysis.table1 are reused, the runs are those of table1.compute
(seed = native-seed run + rep-1/2; optimizers = native-20261008-https + test-reps), and the common task set is the one of table1.geo (verified outcome, N and X > 0 on both sides).

f) per-run sign agreement of the nine optimizer skills (run-k difference vs seed run k / seed mean, against the avg@3 sign).
 g) dev-to-test shrinkage: dev gain announced by the SkillOpt / GEPA runs (tasks and points of the dev set) next to the test avg@3 difference.

Outputs: assets/cache/analysis/mech/stats.json and paper/assets/data/mech/mech_stats.md. Table 1 itself is never written (table1.record is not called).

Sections of the result (per benchmark x method):
 a) cost ratios for input tokens X, tool calls N and turns R: rho_geo (geometric mean of per-task ratios), rho_sum (ratio of sums of per-task means), 95% paired bootstrap
    intervals over tasks (10,000 resamples, seed 20261012), share of tasks with ratio > 1, Wilcoxon signed-rank p on the per-task log ratios.
 b) success: Q difference vs seed (avg@3, paired bootstrap over tasks), help / harm in rep units (d = method successes - seed successes over the runs of a task).
    Success uses every run with a verified outcome (per_task_outcomes), so the task-mean dQ equals Table 1's dQ whenever each run covers the same tasks.
 c) survivor-bias control: rho_geo on tasks both sides solve in every run, and on tasks both fail in every run.
 d) per benchmark: seed headroom, seed run-to-run noise, and the minimum detectable effect (MDE) at 80% power, two-sided alpha 0.05.
    MDE = (z_{0.975} + z_{0.80}) * sqrt(v / n), with v the variance of the per-task difference of the two success estimates and n the number of tasks.
    Single run vs single run: v1 = mean over the three seed run pairs of mean_i (y_ia - y_ib)^2 (the per-task difference between two seed runs, centred at zero because both are the same skill).
    avg@3 vs avg@3: v3 = v1 / 3 (each side averages three independent runs, so the run-noise variance of the difference drops by three).
    The dev panel uses v1 with n = 30 / 30 / 9 (tau3 / AutoBench / TB2). Task-by-skill interaction is not in v, so these MDEs are lower bounds.
"""
import argparse
import json
import math

import numpy as np
from scipy import stats as sps

from src.analysis import main as T
from src.utils.logger import paperout as PO
from src.utils.dataloader import BENCHES, H, SEED_TEXT, current_run, final_skill, gmean, vid

MODEL = "deepseek-flash"
METHODS = (("No skill", "native-noseed"), ("GEPA", "gepa"), ("SkillOpt", "skillopt"), ("SkillAdam", "skilladam"))
METRICS = (("X", "input tokens"), ("N", "tool calls"), ("R", "turns"))
DEV_SIZE = {"tau3-bench": 30, "automationbench": 30, "terminal-bench-2-1": 9}
N_BOOT = 10000
SEED = 20261012
OUT = H / "assets/cache/analysis/mech"
RECORD = H / "paper/assets/data/mech/mech_stats.md"
Z = sps.norm.ppf(0.975) + sps.norm.ppf(0.80)


def per_task_runs(reps):
    """{task: [run rows]} over the runs {rep: file} of one skill, exactly the rows Table 1 averages (verified outcome and N, X known); R = number of distinct requests."""
    per = {}
    for _, f in sorted(reps.items()):
        for (task, _), row in T.rows_of(f).items():
            if row["success"] is None or row["N"] is None or row["X"] is None:
                continue
            per.setdefault(task, []).append(dict(row, R=row["nreq"]))
    return per


def per_task_outcomes(reps):
    """{task: [bool per run]} over every run with a verified outcome, whether or not its N and X are known.
    The success statistics (b) use these outcomes: per_task_runs also drops a run whose N or X is missing (needed for the cost ratios only),
    which silently removed successes/failures from the per-task success mean and made dQ (task mean) disagree with Table 1's Q."""
    per = {}
    for _, f in sorted(reps.items()):
        for (task, _), row in T.rows_of(f).items():
            if row["success"] is None:
                continue
            per.setdefault(task, []).append(row["success"] is True)
    return per


def task_means(per):
    """{task: {S: success rate, N, X, R: mean over runs}}."""
    out = {}
    for task, runs in per.items():
        out[task] = {k: float(np.mean([r[k] if k != "S" else float(r["success"] is True) for r in runs])) for k in ("S", "N", "X", "R")}
    return out


def common_tasks(ref, own):
    """Table 1's common task set: both sides present and N, X > 0 on the per-task means."""
    keep = []
    for task in sorted(ref):
        if task not in own:
            continue
        values = (ref[task]["N"], ref[task]["X"], own[task]["N"], own[task]["X"])
        if all(v > 0 for v in values):
            keep.append(task)
    return keep


def boot_idx(n, rng):
    """(N_BOOT, n) resample indices."""
    return rng.integers(0, n, size=(N_BOOT, n))


def ratio_block(ref, own, tasks, key, rng):
    """rho_geo, rho_sum with 95% paired bootstrap intervals, share of tasks above 1, Wilcoxon p on log ratios."""
    a = np.array([own[t][key] for t in tasks])
    b = np.array([ref[t][key] for t in tasks])
    logs = np.log(a / b)
    idx = boot_idx(len(tasks), rng)
    geo_boot = np.exp(logs[idx].mean(axis=1))
    sum_boot = a[idx].sum(axis=1) / b[idx].sum(axis=1)
    if np.all(logs == 0):
        p = 1.0
    else:
        p = float(sps.wilcoxon(logs).pvalue)
    return {
        "n": len(tasks),
        "rho_geo": float(np.exp(logs.mean())),
        "rho_geo_ci": [float(x) for x in np.percentile(geo_boot, [2.5, 97.5])],
        "rho_sum": float(a.sum() / b.sum()),
        "rho_sum_ci": [float(x) for x in np.percentile(sum_boot, [2.5, 97.5])],
        "share_gt1": float(np.mean(a / b > 1)),
        "wilcoxon_p": p,
    }


def geo_ci(ref, own, tasks, key, rng):
    """rho_geo and its bootstrap interval on a task subset (survivor control)."""
    if len(tasks) == 0:
        return {"n": 0, "rho_geo": None, "rho_geo_ci": None}
    logs = np.log(np.array([own[t][key] / ref[t][key] for t in tasks]))
    boot = np.exp(logs[boot_idx(len(tasks), rng)].mean(axis=1))
    return {"n": len(tasks), "rho_geo": float(np.exp(logs.mean())), "rho_geo_ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])]}


def wins(runs):
    """Number of successful runs of a task."""
    return sum(r["success"] is True for r in runs)


def success_block(out_ref, out_own, q_ref, q_own, rng):
    """Q difference (Table 1 Q), paired bootstrap over tasks on per-task success rates, help / harm in rep units, survivor-style counts.
    out_ref / out_own = per_task_outcomes (all runs with a verified outcome)."""
    tasks = sorted(set(out_ref) & set(out_own))
    s_ref = np.array([np.mean(out_ref[t]) for t in tasks])
    s_own = np.array([np.mean(out_own[t]) for t in tasks])
    diff = s_own - s_ref
    boot = 100 * diff[boot_idx(len(tasks), rng)].mean(axis=1)
    d = np.array([sum(out_own[t]) - sum(out_ref[t]) for t in tasks])
    full_ref = np.array([sum(out_ref[t]) == len(out_ref[t]) == 3 for t in tasks])
    zero_ref = np.array([sum(out_ref[t]) == 0 and len(out_ref[t]) == 3 for t in tasks])
    wins_own = np.array([sum(out_own[t]) for t in tasks])
    return {
        "n_tasks": len(tasks),
        "Q_own": q_own,
        "Q_seed": q_ref,
        "dQ_table1": q_own - q_ref,
        "dQ_task_mean": float(100 * diff.mean()),
        "dQ_ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "help_reps": int(d[d > 0].sum()),
        "harm_reps": int(d[d < 0].sum()),
        "net_reps": int(d.sum()),
        "tasks_helped": int((d > 0).sum()),
        "tasks_harmed": int((d < 0).sum()),
        "seed_3of3_tasks": int(full_ref.sum()),
        "harm_reps_on_seed_3of3": int(d[full_ref & (d < 0)].sum()),
        "tasks_harmed_on_seed_3of3": int((full_ref & (d < 0)).sum()),
        "seed_0of3_tasks": int(zero_ref.sum()),
        "seed_0of3_solved_ge1": int((zero_ref & (wins_own >= 1)).sum()),
        "seed_0of3_reps_won": int(wins_own[zero_ref].sum()),
    }


def sign(x, tol=1e-9):
    """-1, 0, +1 with a tolerance for float noise."""
    return 0 if abs(x) < tol else (1 if x > 0 else -1)


def sign_agreement(res):
    """Per-run sign agreement for the nine optimizer skills (GEPA, SkillOpt, SkillAdam x 3 benchmarks): the sign of (skill run-k Q - seed run-k Q),
    and of (skill run-k Q - seed 3-run mean Q), against the sign of the avg@3 difference (Table 1 dQ). Disagreement = opposite non-zero sign; tie = per-run difference exactly 0 while avg@3 is not."""
    rows = []
    for b in BENCHES:
        seed_runs = res[b]["seed_run_Q"]
        seed_mean = float(np.mean(seed_runs))
        for name, _ in METHODS[1:]:
            m = res[b]["methods"][name]
            avg_sign = sign(m["success"]["dQ_table1"])
            for k, q in enumerate(m["run_Q"]):
                d_run = q - seed_runs[k]
                d_mean = q - seed_mean
                rows.append({"bench": b, "method": name, "k": k, "avg3_dQ": m["success"]["dQ_table1"], "d_vs_seed_run_k": d_run, "d_vs_seed_mean": d_mean,
                             "sign_avg3": avg_sign, "sign_run_k": sign(d_run), "sign_mean": sign(d_mean)})
    summary = {}
    for k in range(3):
        sel = [r for r in rows if r["k"] == k]
        summary[str(k)] = {
            "n": len(sel),
            "disagree_vs_seed_run_k": sum(r["sign_run_k"] * r["sign_avg3"] == -1 for r in sel),
            "tie_vs_seed_run_k": sum(r["sign_run_k"] == 0 and r["sign_avg3"] != 0 for r in sel),
            "disagree_vs_seed_mean": sum(r["sign_mean"] * r["sign_avg3"] == -1 for r in sel),
            "tie_vs_seed_mean": sum(r["sign_mean"] == 0 and r["sign_avg3"] != 0 for r in sel),
        }
    return rows, summary


def dev_to_test(res):
    """Dev gain announced by the run (SkillOpt: best_selection_hard - baseline_selection_hard; GEPA: val_scores[best_idx] - val_scores[0]) in tasks and points of the dev set,
    next to the test avg@3 difference (Table 1 dQ) and the run's own announced test number."""
    rows = []
    for b in BENCHES:
        size = DEV_SIZE[b]
        for name, m in METHODS[1:3]:
            run = current_run(m, MODEL, b)
            extra = json.loads((run / "summary.json").read_text())["extra"]
            if m == "skillopt":
                base, best = extra["baseline_selection_hard"], extra["best_selection_hard"]
            else:
                base, best = extra["val_scores"][0], extra["val_scores"][extra["best_idx"]]
            gain = best - base
            rows.append({"bench": b, "method": name, "dev_size": size, "dev_seed": base, "dev_selected": best, "dev_gain_tasks": gain * size, "dev_gain_pts": 100 * gain,
                         "test_dQ_avg3": res[b]["methods"][name]["success"]["dQ_table1"]})
    return rows


def survivor_block(ref, own, per_ref, per_own, tasks, rng):
    """rho_geo restricted to tasks both sides solve in every run, and tasks both fail in every run."""
    both_win, both_fail = [], []
    for t in tasks:
        wr, wo = wins(per_ref[t]), wins(per_own[t])
        if wr == len(per_ref[t]) == 3 and wo == len(per_own[t]) == 3:
            both_win.append(t)
        if wr == 0 and wo == 0 and len(per_ref[t]) == 3 and len(per_own[t]) == 3:
            both_fail.append(t)
    out = {}
    for label, subset in (("both_solve_all", both_win), ("both_fail_all", both_fail)):
        out[label] = {key: geo_ci(ref, own, subset, key, rng) for key, _ in METRICS}
    return out


def seed_block(per_ref, ref, tasks):
    """Seed headroom, seed run-to-run noise, MDE."""
    runs3 = [t for t in sorted(per_ref) if len(per_ref[t]) == 3]
    counts = [wins(per_ref[t]) for t in runs3]
    head = {"n_tasks": len(runs3), "zero": counts.count(0), "one_two": counts.count(1) + counts.count(2), "three": counts.count(3)}
    # per-task SD of log X over the 3 runs
    sd = np.array([np.std(np.log([r["X"] for r in per_ref[t]]), ddof=1) for t in runs3])
    sd_calls = np.array([np.std(np.log([r["N"] for r in per_ref[t]]), ddof=1) for t in runs3 if all(r["N"] > 0 for r in per_ref[t])])
    # run-level geometric means over tasks with all three runs, pairwise max ratio
    gm = []
    for i in range(3):
        gm.append(float(np.exp(np.mean([math.log(per_ref[t][i]["X"]) for t in runs3]))))
    pair_ratios = [max(gm[i], gm[j]) / min(gm[i], gm[j]) for i in range(3) for j in range(i + 1, 3)]
    # success noise between seed runs
    y = np.array([[float(r["success"] is True) for r in per_ref[t]] for t in runs3])
    v1 = float(np.mean([np.mean((y[:, i] - y[:, j]) ** 2) for i in range(3) for j in range(i + 1, 3)]))
    v3 = v1 / 3
    n = len(runs3)
    return {
        "headroom": head,
        "sd_logX_mean": float(sd.mean()),
        "sd_logX_median": float(np.median(sd)),
        "sd_logN_mean": float(sd_calls.mean()),
        "run_geomean_X": gm,
        "pairwise_max_ratio_X": float(max(pair_ratios)),
        "run_success_pct": [float(100 * y[:, i].mean()) for i in range(3)],
        "var_diff_single_run": v1,
        "var_diff_avg3": v3,
        "n_test_tasks": n,
        "mde_avg3_pts": float(100 * Z * math.sqrt(v3 / n)),
        "dev_size": None,
        "mde_dev_single_pts": None,
    }


def compute():
    """All numbers, {bench: {...}}."""
    rng = np.random.default_rng(SEED)
    res = {}
    for bench in BENCHES:
        seed_run = current_run("native-seed", MODEL, bench)
        seed_dirs = [seed_run, seed_run.parent / "rep-1", seed_run.parent / "rep-2"]
        seed_reps = T.rep_files(seed_dirs, vid(SEED_TEXT))
        per_ref = per_task_runs(seed_reps)
        out_ref = per_task_outcomes(seed_reps)
        ref = task_means(per_ref)
        cell = {"methods": {}, "runs": {"seed": {str(k): str(v) for k, v in seed_reps.items()}}}
        # Table 1 Q of the seed row, same formula as table1.compute
        seed_q = table1_q(seed_reps)
        for name, m in METHODS:
            run = current_run(m, MODEL, bench)
            text = final_skill(m, run)
            reps = T.rep_files([run, T.EX / MODEL / "test-reps" / f"{bench}__{m}__{run.name}"], vid(text))
            per_own = per_task_runs(reps)
            own = task_means(per_own)
            tasks = common_tasks(ref, own)
            q_own = table1_q(reps)
            block = {
                "run": run.name,
                "files": {str(k): str(v) for k, v in sorted(reps.items())},
                "n_common": len(tasks),
                "run_Q": run_q(reps),
                "ratios": {key: ratio_block(ref, own, tasks, key, rng) for key, _ in METRICS},
                "success": success_block(out_ref, per_task_outcomes(reps), seed_q, q_own, rng),
                "survivor": survivor_block(ref, own, per_ref, per_own, tasks, rng),
            }
            cell["methods"][name] = block
            cell["runs"][name] = {str(k): str(v) for k, v in sorted(reps.items())}
        sb = seed_block(per_ref, ref, None)
        sb["dev_size"] = DEV_SIZE[bench]
        sb["mde_dev_single_pts"] = float(100 * Z * math.sqrt(sb["var_diff_single_run"] / DEV_SIZE[bench]))
        cell["seed"] = sb
        cell["seed_Q"] = seed_q
        cell["seed_run_Q"] = run_q(seed_reps)
        res[bench] = cell
    return res


def run_q(reps):
    """Q (percent) of every run separately, in repetition order."""
    out = []
    for _, f in sorted(reps.items()):
        known = [r for r in T.rows_of(f).values() if r["success"] is not None]
        out.append(100 * sum(r["success"] is True for r in known) / len(known))
    return out


def table1_q(reps):
    """Q exactly as table1.compute: 100 * mean over runs of the success rate among tasks with a verified outcome."""
    rates = []
    for _, f in sorted(reps.items()):
        known = [r for r in T.rows_of(f).values() if r["success"] is not None]
        rates.append(sum(r["success"] is True for r in known) / len(known))
    return 100 * sum(rates) / len(rates)


def fmt_ci(v, d=2):
    return f"[{v[0]:.{d}f}, {v[1]:.{d}f}]"


def pfmt(p):
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def markdown(res):
    """Readable result tables."""
    short = {"tau3-bench": "tau3", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}
    L = []
    for key, label in METRICS:
        L += [f"### a) {label} ({key}) vs the Seed skill row", "",
              "rho_geo = geometric mean of per-task ratios (Table 1 for X and N); rho_sum = sum of per-task means / seed sum; intervals = paired bootstrap over tasks; >1 = share of tasks with ratio above 1; p = Wilcoxon signed-rank on per-task log ratios.", ""]
        rows = []
        for b in BENCHES:
            for name, _ in METHODS:
                r = res[b]["methods"][name]["ratios"][key]
                rows.append([short[b], name, r["n"], f"{r['rho_geo']:.3f}", fmt_ci(r["rho_geo_ci"], 3), f"{r['rho_sum']:.3f}", fmt_ci(r["rho_sum_ci"], 3), f"{100 * r['share_gt1']:.0f}%", pfmt(r["wilcoxon_p"])])
        L += [PO.mdtable(["Bench", "Method", "n", "rho_geo", "95% CI", "rho_sum", "95% CI", ">1", "p"], rows), ""]
    L += ["### b) success vs the Seed skill row", "",
          "Q = Table 1 avg@3; dQ CI = paired bootstrap over tasks of the per-task mean success difference; help / harm = sum of positive / negative per-task d (method wins - seed wins over the 3 runs, rep units); harm@3/3 = harm reps (tasks) on tasks the seed solves 3/3; 0/3 solved = tasks the seed solves 0/3 that the method solves at least once (of all seed 0/3 tasks).", ""]
    rows = []
    for b in BENCHES:
        for name, _ in METHODS:
            s = res[b]["methods"][name]["success"]
            rows.append([short[b], name, s["n_tasks"], f"{s['Q_own']:.1f}", f"{s['Q_seed']:.1f}", f"{s['dQ_table1']:+.1f}", f"{s['dQ_task_mean']:+.1f}", fmt_ci(s["dQ_ci"], 1),
                         f"+{s['help_reps']} / {s['harm_reps']}", f"{s['harm_reps_on_seed_3of3']} ({s['tasks_harmed_on_seed_3of3']} of {s['seed_3of3_tasks']})", f"{s['seed_0of3_solved_ge1']} of {s['seed_0of3_tasks']}"])
    L += [PO.mdtable(["Bench", "Method", "n", "Q", "Q seed", "dQ (T1)", "dQ (task mean)", "95% CI", "help / harm reps", "harm@seed 3/3", "seed 0/3 solved >=1"], rows), ""]
    L += ["### c) survivor-bias control: rho_geo on tasks both sides solve in all 3 runs / both fail in all 3 runs", "",
          "Cells: rho_geo [95% CI] (n tasks).", ""]
    rows = []
    for b in BENCHES:
        for name, _ in METHODS:
            sv = res[b]["methods"][name]["survivor"]
            row = [short[b], name]
            for grp in ("both_solve_all", "both_fail_all"):
                for key in ("X", "N", "R"):
                    g = sv[grp][key]
                    row.append("-" if g["rho_geo"] is None else f"{g['rho_geo']:.2f} {fmt_ci(g['rho_geo_ci'])} (n={g['n']})")
            rows.append(row)
    L += [PO.mdtable(["Bench", "Method", "solve: X", "solve: N", "solve: R", "fail: X", "fail: N", "fail: R"], rows), ""]
    L += ["### d) Seed headroom, run-to-run noise and MDE", "",
          "headroom = tasks with 3 runs by seed successes. SD = per-task SD of log input tokens over the 3 seed runs (mean / median). max ratio = largest pairwise ratio of the seed runs' task-geometric-mean input tokens. v1, v3 = variance of the per-task success difference between two seed runs (single run) and for avg@3 vs avg@3 (v1/3). MDE (pts) = (1.96 + 0.84) sqrt(v / n) * 100, 80% power, two-sided alpha 0.05.", ""]
    rows = []
    for b in BENCHES:
        s = res[b]["seed"]
        h = s["headroom"]
        rows.append([short[b], f"{h['zero']} / {h['one_two']} / {h['three']} (of {h['n_tasks']})", f"{s['sd_logX_mean']:.3f} / {s['sd_logX_median']:.3f}", f"{s['pairwise_max_ratio_X']:.3f}",
                     " / ".join(f"{x:.1f}" for x in s["run_success_pct"]), f"{s['var_diff_single_run']:.3f}", f"{s['var_diff_avg3']:.3f}", f"{s['mde_avg3_pts']:.1f} (n={s['n_test_tasks']})", f"{s['mde_dev_single_pts']:.1f} (n={s['dev_size']})"])
    L += [PO.mdtable(["Bench", "seed 0/3 / 1-2/3 / 3/3", "SD log X mean / median", "max pairwise ratio", "seed run success %", "v1", "v3", "MDE avg@3 vs avg@3 (pts)", "MDE single-run dev (pts)"], rows), ""]
    L += ["### e) Per-run Q (rep 0 / 1 / 2; rep 0 = the optimizer run's own first evaluation) and first-run gain vs the seed's rep 0", "",
          "Q per run among tasks with a verified outcome; first-run dQ = rep-0 Q of the method - rep-0 Q of the seed; avg@3 dQ = Table 1.", ""]
    rows = []
    for b in BENCHES:
        seed_runs = res[b]["seed_run_Q"]
        rows.append([short[b], "Seed skill", " / ".join(f"{x:.1f}" for x in seed_runs), "-", "-"])
        for name, _ in METHODS:
            m = res[b]["methods"][name]
            rq = m["run_Q"]
            rows.append([short[b], name, " / ".join(f"{x:.1f}" for x in rq), f"{rq[0] - seed_runs[0]:+.1f}", f"{m['success']['dQ_table1']:+.1f}"])
    L += [PO.mdtable(["Bench", "Method", "Q per run", "first-run dQ", "avg@3 dQ"], rows), ""]
    sign_rows, sign_sum = res["_sign"]["rows"], res["_sign"]["summary"]
    L += ["### f) Per-run sign agreement: does one run's difference to the seed have the sign of the avg@3 difference?", "",
          "Nine skills (GEPA, SkillOpt, SkillAdam x 3 benchmarks). For run k: sign(skill run-k Q - seed run-k Q) and sign(skill run-k Q - seed 3-run mean Q) against sign(avg@3 dQ, Table 1). disagree = opposite sign; tie = per-run difference exactly 0 while avg@3 is not.", ""]
    L += [PO.mdtable(["run k", "disagree vs seed run k", "tie vs seed run k", "disagree vs seed mean", "tie vs seed mean"],
                     [[k, v["disagree_vs_seed_run_k"], v["tie_vs_seed_run_k"], v["disagree_vs_seed_mean"], v["tie_vs_seed_mean"]] for k, v in sign_sum.items()]), ""]
    L += [PO.mdtable(["Bench", "Method", "avg@3 dQ", "run0 - seed run0", "run1 - seed run1", "run2 - seed run2", "run0 - seed mean", "run1 - seed mean", "run2 - seed mean"],
                     [[short[b], name] + [f"{[r for r in sign_rows if r['bench'] == b and r['method'] == name][0]['avg3_dQ']:+.1f}"]
                      + [f"{[r for r in sign_rows if r['bench'] == b and r['method'] == name and r['k'] == k][0]['d_vs_seed_run_k']:+.1f}" for k in range(3)]
                      + [f"{[r for r in sign_rows if r['bench'] == b and r['method'] == name and r['k'] == k][0]['d_vs_seed_mean']:+.1f}" for k in range(3)]
                      for b in BENCHES for name, _ in METHODS[1:]]), ""]
    L += ["### g) Dev-to-test shrinkage (SkillOpt, GEPA)", "",
          "Dev gain announced by the run: SkillOpt best_selection_hard - baseline_selection_hard; GEPA val_scores[best_idx] - val_scores[0] (seed). Tasks = gain x dev size (30 / 30 / 9); points = 100 x gain. Test dQ = avg@3 difference vs the seed row (Table 1).", ""]
    L += [PO.mdtable(["Bench", "Method", "dev size", "dev seed", "dev selected", "dev gain (tasks)", "dev gain (pts)", "test dQ avg@3 (pts)"],
                     [[short[r["bench"]], r["method"], r["dev_size"], f"{100 * r['dev_seed']:.1f}", f"{100 * r['dev_selected']:.1f}", f"{r['dev_gain_tasks']:+.1f}", f"{r['dev_gain_pts']:+.1f}", f"{r['test_dQ_avg3']:+.1f}"] for r in res["_dev"]]), ""]
    return "\n".join(L)


def check_table1(res):
    """Compare Q, rho_trj, rho_inv of every method with the live table1.compute (same run selection) and return rows and the number of mismatches at print precision."""
    t1 = T.compute()
    rows = []
    bad = 0
    for b in BENCHES:
        for name, _ in METHODS:
            ref = t1[f"{MODEL}|{b}"][name]
            mine = res[b]["methods"][name]
            q_ok = f"{ref['Q']:.1f}" == f"{mine['success']['Q_own']:.1f}"
            x_ok = f"{ref['rho_trj']:.2f}" == f"{mine['ratios']['X']['rho_geo']:.2f}" and abs(ref["rho_trj"] - mine["ratios"]["X"]["rho_geo"]) < 1e-9
            n_ok = f"{ref['rho_inv']:.2f}" == f"{mine['ratios']['N']['rho_geo']:.2f}" and abs(ref["rho_inv"] - mine["ratios"]["N"]["rho_geo"]) < 1e-9
            n_same = ref["n"] == mine["n_common"]
            bad += (not q_ok) + (not x_ok) + (not n_ok) + (not n_same)
            rows.append([b, name, f"{ref['Q']:.3f}", f"{mine['success']['Q_own']:.3f}", f"{ref['rho_trj']:.4f}", f"{mine['ratios']['X']['rho_geo']:.4f}", f"{ref['rho_inv']:.4f}", f"{mine['ratios']['N']['rho_geo']:.4f}", f"{ref['n']} / {mine['n_common']}", "ok" if (q_ok and x_ok and n_ok and n_same) else "MISMATCH"])
    return rows, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()
    res = compute()
    rows, bad = check_table1(res)
    res["_table1_check"] = {"mismatches": bad}
    sign_rows, sign_sum = sign_agreement(res)
    res["_sign"] = {"rows": sign_rows, "summary": sign_sum}
    res["_dev"] = dev_to_test(res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(res, indent=1))
    check = PO.mdtable(["Bench", "Method", "Q T1", "Q here", "rho_trj T1", "rho_trj here", "rho_inv T1", "rho_inv here", "n common T1 / here", "status"], rows)
    raw = []
    for b in BENCHES:
        raw += list(res[b]["runs"]["seed"].values())
        for name, _ in METHODS:
            raw += list(res[b]["runs"][name].values())
    when, _ = PO._when(sorted(set(raw)))
    text = ["# mech_stats", "",
            "Audit record, written by `bipa/src/analysis/mech/stats.py`; rerunning it refreshes this file. Do not edit by hand.", "",
            "## 1. What it shows", "",
            "Paired statistics of the four comparison rows (No skill, GEPA, SkillOpt, SkillAdam) against the Seed skill row on DeepSeek-V4.1-Flash, per benchmark: cost ratios for input tokens, tool calls and turns (geometric mean and ratio of sums, bootstrap intervals, share above 1, Wilcoxon), success differences with help / harm counts, survivor-bias controls, seed headroom and noise, and minimum detectable effects. Kept in the repro package, not printed. Task sets and runs are those of Table 1 (`sec05_table1`).", "",
            "## 2. Runs and aggregation", "",
            "Per task the mean over the 3 runs of each side (seed: native-seed run + rep-1 + rep-2; methods: native-20261008-https + test-reps; No skill: its latest native-noseed run). Common task set = Table 1's (verified outcome, N and X > 0 on both sides). Bootstrap: 10,000 task resamples, numpy default_rng seed 20261012, percentile 95% intervals. Turns R = number of distinct requests of the episode.", "",
            "## 3. Regenerate (from `/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation/bipa`)", "", "```", ".venv/bin/python -m src.analysis.mech.stats", "```", "",
            "## 4. Table 1 reproduction check", "", check, "", f"Mismatches: {bad}.", "",
            "## 5. Numbers", "", markdown(res), "",
            "## 6. When (underlying evaluation files)", "", when, ""]
    RECORD.write_text("\n".join(text))
    print("table1 mismatches:", bad)
    print(RECORD)


if __name__ == "__main__":
    main()
