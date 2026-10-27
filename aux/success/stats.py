"""Paired statistics of No skill, GEPA, SkillOpt and SkillAdam against the Seed row on Table 1's DeepSeek-V4.1-Flash runs and
task sets.

a) Cost ratios (X input tokens, N tool calls, R turns) with paired bootstrap intervals. b) Success difference, help and harm.
c) Ratios on the tasks both sides always solve or always fail. d) Seed headroom, run noise and MDE. e) Q per run.
f) Per-run sign agreement with the avg@3 difference. g) The dev gain each run reported against its test difference.

Per task, the mean over the three test runs of each side; the common task set and the files of each repetition are Table 1's
(settings.cell_files). Intervals are percentile intervals over task resamples, all drawn from one sequential generator seeded
with settings.SEED, so they stay the intervals printed beside Table 1. MDE = (z_0.975 + z_0.80) sqrt(v / n). Turns R = the
distinct requests of an episode."""
import json, math

import numpy as np
from scipy import stats as sps

from src.analysis import settings as S
from src.analysis import table1 as T
from src.utils import logger

MODEL = S.DEEPSEEK
BENCHES = ("tau3-bench", "automationbench", "terminal-bench-2-1")
METHODS = (("No skill", "native-noseed"), ("GEPA", "gepa"), ("SkillOpt", "skillopt"), ("SkillAdam", "skilladam"))
METRICS = (("X", "input tokens"), ("N", "tool calls"), ("R", "turns"))
Z = sps.norm.ppf(0.975) + sps.norm.ppf(0.80)


def runs_of(files):
    """[{task: row}] per repetition of {repetition: [evaluation.json]}, read as Table 1 reads them."""
    return list(S.rows(files, MODEL).values())


def per_task_runs(files):
    """{task: [run rows with a verdict and N, X]}, R = distinct requests."""
    per = {}
    for rows in runs_of(files):
        for task, row in rows.items():
            if row["success"] is not None and row["N"] is not None and row["X"] is not None:
                per.setdefault(task, []).append(dict(row, R=row["nreq"]))
    return per


def per_task_outcomes(files):
    """{task: [success per run with a verdict]}."""
    per = {}
    for rows in runs_of(files):
        for task, row in rows.items():
            if row["success"] is not None:
                per.setdefault(task, []).append(row["success"] is True)
    return per


def task_means(per):
    return {t: {k: float(np.mean([r[k] if k != "S" else float(r["success"] is True) for r in runs])) for k in ("S", "N", "X", "R")}
            for t, runs in per.items()}


def common_tasks(ref, own):
    """Table 1's common task set: both sides present, N and X > 0."""
    return [t for t in sorted(ref) if t in own and all(v > 0 for v in (ref[t]["N"], ref[t]["X"], own[t]["N"], own[t]["X"]))]


def boot_idx(n, rng):
    return rng.integers(0, n, size=(S.N_BOOT, n))


def ci(x):
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def ratio_block(ref, own, tasks, key, rng):
    a, b = np.array([own[t][key] for t in tasks]), np.array([ref[t][key] for t in tasks])
    logs = np.log(a / b)
    idx = boot_idx(len(tasks), rng)
    return {"n": len(tasks), "rho_geo": float(np.exp(logs.mean())), "rho_geo_ci": ci(np.exp(logs[idx].mean(axis=1))),
            "rho_sum": float(a.sum() / b.sum()), "rho_sum_ci": ci(a[idx].sum(axis=1) / b[idx].sum(axis=1)),
            "share_gt1": float(np.mean(a / b > 1)), "wilcoxon_p": 1.0 if np.all(logs == 0) else float(sps.wilcoxon(logs).pvalue)}


def geo_ci(ref, own, tasks, key, rng):
    if not tasks:
        return {"n": 0, "rho_geo": None, "rho_geo_ci": None}
    logs = np.log(np.array([own[t][key] / ref[t][key] for t in tasks]))
    return {"n": len(tasks), "rho_geo": float(np.exp(logs.mean())), "rho_geo_ci": ci(np.exp(logs[boot_idx(len(tasks), rng)].mean(axis=1)))}


def wins(runs):
    return sum(r["success"] is True for r in runs)


def success_block(out_ref, out_own, q_ref, q_own, rng):
    tasks = sorted(set(out_ref) & set(out_own))
    diff = np.array([np.mean(out_own[t]) for t in tasks]) - np.array([np.mean(out_ref[t]) for t in tasks])
    boot = 100 * diff[boot_idx(len(tasks), rng)].mean(axis=1)
    d = np.array([sum(out_own[t]) - sum(out_ref[t]) for t in tasks])
    full = np.array([sum(out_ref[t]) == len(out_ref[t]) == 3 for t in tasks])
    zero = np.array([sum(out_ref[t]) == 0 and len(out_ref[t]) == 3 for t in tasks])
    won = np.array([sum(out_own[t]) for t in tasks])
    return {"n_tasks": len(tasks), "Q_own": q_own, "Q_seed": q_ref, "dQ_table1": q_own - q_ref, "dQ_task_mean": float(100 * diff.mean()),
            "dQ_ci": ci(boot), "help_reps": int(d[d > 0].sum()), "harm_reps": int(d[d < 0].sum()), "net_reps": int(d.sum()),
            "tasks_helped": int((d > 0).sum()), "tasks_harmed": int((d < 0).sum()), "seed_3of3_tasks": int(full.sum()),
            "harm_reps_on_seed_3of3": int(d[full & (d < 0)].sum()), "tasks_harmed_on_seed_3of3": int((full & (d < 0)).sum()),
            "seed_0of3_tasks": int(zero.sum()), "seed_0of3_solved_ge1": int((zero & (won >= 1)).sum()), "seed_0of3_reps_won": int(won[zero].sum())}


def survivor_block(ref, own, per_ref, per_own, tasks, rng):
    """Ratios on tasks both sides solve in all three runs, and on tasks both fail in all three."""
    def every(t, n):
        return len(per_ref[t]) == 3 and len(per_own[t]) == 3 and wins(per_ref[t]) == wins(per_own[t]) == n
    groups = (("both_solve_all", [t for t in tasks if every(t, 3)]), ("both_fail_all", [t for t in tasks if every(t, 0)]))
    return {label: {key: geo_ci(ref, own, sub, key, rng) for key, _ in METRICS} for label, sub in groups}


def dev_size(bench):
    """Tasks in the dev panel the optimizers selected on."""
    return json.loads((S.run(MODEL, bench, "native-seed") / "summary.json").read_text())["panels"]["sel"]


def seed_block(per_ref, bench):
    """Seed headroom, run-to-run noise of log X and success, and the MDE at 80% power."""
    runs3 = [t for t in sorted(per_ref) if len(per_ref[t]) == 3]
    counts = [wins(per_ref[t]) for t in runs3]
    sd = np.array([np.std(np.log([r["X"] for r in per_ref[t]]), ddof=1) for t in runs3])
    sd_calls = np.array([np.std(np.log([r["N"] for r in per_ref[t]]), ddof=1) for t in runs3 if all(r["N"] > 0 for r in per_ref[t])])
    gm = [float(np.exp(np.mean([math.log(per_ref[t][i]["X"]) for t in runs3]))) for i in range(3)]
    pairs = [(i, j) for i in range(3) for j in range(i + 1, 3)]
    y = np.array([[float(r["success"] is True) for r in per_ref[t]] for t in runs3])
    v1 = float(np.mean([np.mean((y[:, i] - y[:, j]) ** 2) for i, j in pairs]))
    n_dev = dev_size(bench)
    return {"headroom": {"n_tasks": len(runs3), "zero": counts.count(0), "one_two": counts.count(1) + counts.count(2), "three": counts.count(3)},
            "sd_logX_mean": float(sd.mean()), "sd_logX_median": float(np.median(sd)), "sd_logN_mean": float(sd_calls.mean()),
            "run_geomean_X": gm, "pairwise_max_ratio_X": float(max(max(gm[i], gm[j]) / min(gm[i], gm[j]) for i, j in pairs)),
            "run_success_pct": [float(100 * y[:, i].mean()) for i in range(3)], "var_diff_single_run": v1, "var_diff_avg3": v1 / 3,
            "n_test_tasks": len(runs3), "mde_avg3_pts": float(100 * Z * math.sqrt(v1 / 3 / len(runs3))), "dev_size": n_dev,
            "mde_dev_single_pts": float(100 * Z * math.sqrt(v1 / n_dev))}


def run_q(files):
    """Q of each run: 100 * solved / verified."""
    known = [[r for r in rows.values() if r["success"] is not None] for rows in runs_of(files)]
    return [100 * sum(r["success"] is True for r in k) / len(k) for k in known]


def compute():
    rng = S.rng()  # the one sequential generator of settings.SEED (module docstring)
    res = {}
    for bench in BENCHES:
        seed_files = S.cell_files(MODEL, bench, "native-seed")
        per_ref, out_ref = per_task_runs(seed_files), per_task_outcomes(seed_files)
        ref, seed_q = task_means(per_ref), float(np.mean(run_q(seed_files)))
        cell = {"methods": {}, "runs": {"seed": {str(k): [str(f) for f in v] for k, v in seed_files.items()}}}
        for name, m in METHODS:
            files = S.cell_files(MODEL, bench, m)
            per_own = per_task_runs(files)
            own = task_means(per_own)
            tasks = common_tasks(ref, own)
            listed = {str(k): [str(f) for f in v] for k, v in sorted(files.items())}
            ratios = {key: ratio_block(ref, own, tasks, key, rng) for key, _ in METRICS}
            success = success_block(out_ref, per_task_outcomes(files), seed_q, float(np.mean(run_q(files))), rng)
            cell["methods"][name] = {"run": S.run(MODEL, bench, m).name, "files": listed, "n_common": len(tasks), "run_Q": run_q(files),
                                     "ratios": ratios, "success": success, "survivor": survivor_block(ref, own, per_ref, per_own, tasks, rng)}
            cell["runs"][name] = listed
        cell.update(seed=seed_block(per_ref, bench), seed_Q=seed_q, seed_run_Q=run_q(seed_files))
        res[bench] = cell
    return res


def sign(x, tol=1e-9):
    return 0 if abs(x) < tol else (1 if x > 0 else -1)


def sign_agreement(res):
    """Sign of each run's difference to the seed (run k, and the seed mean) against the sign of the avg@3 difference."""
    rows = []
    for b in BENCHES:
        seed_runs = res[b]["seed_run_Q"]
        for name, _ in METHODS[1:]:
            m = res[b]["methods"][name]
            dq = m["success"]["dQ_table1"]
            for k, q in enumerate(m["run_Q"]):
                rows.append({"bench": b, "method": name, "k": k, "avg3_dQ": dq, "d_vs_seed_run_k": q - seed_runs[k],
                             "d_vs_seed_mean": q - float(np.mean(seed_runs)), "sign_avg3": sign(dq), "sign_run_k": sign(q - seed_runs[k]),
                             "sign_mean": sign(q - float(np.mean(seed_runs)))})
    summary = {}
    for k in range(3):
        sel = [r for r in rows if r["k"] == k]
        summary[str(k)] = {"n": len(sel)}
        for ref, key in (("seed_run_k", "sign_run_k"), ("seed_mean", "sign_mean")):
            summary[str(k)][f"disagree_vs_{ref}"] = sum(r[key] * r["sign_avg3"] == -1 for r in sel)
            summary[str(k)][f"tie_vs_{ref}"] = sum(r[key] == 0 and r["sign_avg3"] != 0 for r in sel)
    return rows, summary


def dev_to_test(res):
    """Dev gain the SkillOpt and GEPA runs reported, in tasks and points, next to the test avg@3 difference."""
    rows = []
    for b in BENCHES:
        n_dev = dev_size(b)
        for name, m in METHODS[1:3]:
            extra = json.loads((S.run(MODEL, b, m) / "summary.json").read_text())["extra"]
            if m == "skillopt":
                base, best = extra["baseline_selection_hard"], extra["best_selection_hard"]
            else:
                base, best = extra["val_scores"][0], extra["val_scores"][extra["best_idx"]]
            rows.append({"bench": b, "method": name, "dev_size": n_dev, "dev_seed": base, "dev_selected": best,
                         "dev_gain_tasks": (best - base) * n_dev, "dev_gain_pts": 100 * (best - base),
                         "test_dQ_avg3": res[b]["methods"][name]["success"]["dQ_table1"]})
    return rows


def check_table1(res):
    """Q, rho_trj, rho_turns, rho_inv and n of every row against table1.compute."""
    t1, rows, bad = T.compute(MODEL), [], 0
    for b in BENCHES:
        for name, _ in METHODS:
            ref, mine = t1[f"{MODEL}|{b}"][name], res[b]["methods"][name]
            x, n, r = (mine["ratios"][key]["rho_geo"] for key in ("X", "N", "R"))
            oks = [f"{ref['Q']:.1f}" == f"{mine['success']['Q_own']:.1f}",
                   f"{ref['rho_trj']:.2f}" == f"{x:.2f}" and abs(ref["rho_trj"] - x) < 1e-9,
                   f"{ref['rho_turns']:.2f}" == f"{r:.2f}" and abs(ref["rho_turns"] - r) < 1e-9,
                   f"{ref['rho_inv']:.2f}" == f"{n:.2f}" and abs(ref["rho_inv"] - n) < 1e-9, ref["n"] == mine["n_common"]]
            bad += oks.count(False)
            rows.append([b, name, f"{ref['Q']:.3f}", f"{mine['success']['Q_own']:.3f}", f"{ref['rho_trj']:.4f}", f"{x:.4f}",
                         f"{ref['rho_inv']:.4f}", f"{n:.4f}", f"{ref['n']} / {mine['n_common']}", "ok" if all(oks) else "MISMATCH"])
    return rows, bad


def fci(v, d=2):
    return f"[{v[0]:.{d}f}, {v[1]:.{d}f}]"


def markdown(res):
    md, L, BN = logger.md_table, [], S.BENCH_SHORT
    every = [(b, name) for b in BENCHES for name, _ in METHODS]
    for key, label in METRICS:
        rows = []
        for b, name in every:
            r = res[b]["methods"][name]["ratios"][key]
            p = r["wilcoxon_p"]
            rows.append([BN[b], name, r["n"], f"{r['rho_geo']:.3f}", fci(r["rho_geo_ci"], 3), f"{r['rho_sum']:.3f}", fci(r["rho_sum_ci"], 3),
                         f"{100 * r['share_gt1']:.0f}%", f"{p:.1e}" if p < 0.001 else f"{p:.3f}"])
        L += [f"### a) {label} ({key}) vs the Seed skill row", "", md(["Bench", "Method", "n", "rho_geo", "95% CI", "rho_sum", "95% CI", ">1", "p"], rows), ""]
    rows = []
    for b, name in every:
        s = res[b]["methods"][name]["success"]
        rows.append([BN[b], name, s["n_tasks"], f"{s['Q_own']:.1f}", f"{s['Q_seed']:.1f}", f"{s['dQ_table1']:+.1f}", f"{s['dQ_task_mean']:+.1f}",
                     fci(s["dQ_ci"], 1), f"+{s['help_reps']} / {s['harm_reps']}",
                     f"{s['harm_reps_on_seed_3of3']} ({s['tasks_harmed_on_seed_3of3']} of {s['seed_3of3_tasks']})",
                     f"{s['seed_0of3_solved_ge1']} of {s['seed_0of3_tasks']}"])
    L += ["### b) success vs the Seed skill row", "", md(["Bench", "Method", "n", "Q", "Q seed", "dQ (T1)", "dQ (task mean)", "95% CI",
                                                          "help / harm reps", "harm@seed 3/3", "seed 0/3 solved >=1"], rows), ""]
    rows = []
    for b, name in every:
        sv = res[b]["methods"][name]["survivor"]
        rows.append([BN[b], name] + ["-" if g["rho_geo"] is None else f"{g['rho_geo']:.2f} {fci(g['rho_geo_ci'])} (n={g['n']})"
                                     for grp in ("both_solve_all", "both_fail_all") for g in (sv[grp][k] for k in ("X", "N", "R"))])
    L += ["### c) rho_geo on tasks both sides solve / both fail in all 3 runs", "",
          md(["Bench", "Method", "solve: X", "solve: N", "solve: R", "fail: X", "fail: N", "fail: R"], rows), ""]
    rows = []
    for b in BENCHES:
        s, h = res[b]["seed"], res[b]["seed"]["headroom"]
        rows.append([BN[b], f"{h['zero']} / {h['one_two']} / {h['three']} (of {h['n_tasks']})", f"{s['sd_logX_mean']:.3f} / {s['sd_logX_median']:.3f}",
                     f"{s['pairwise_max_ratio_X']:.3f}", " / ".join(f"{x:.1f}" for x in s["run_success_pct"]), f"{s['var_diff_single_run']:.3f}",
                     f"{s['var_diff_avg3']:.3f}", f"{s['mde_avg3_pts']:.1f} (n={s['n_test_tasks']})", f"{s['mde_dev_single_pts']:.1f} (n={s['dev_size']})"])
    L += ["### d) Seed headroom, run-to-run noise and MDE", "", md(["Bench", "seed 0/3 / 1-2/3 / 3/3", "SD log X mean / median", "max pairwise ratio",
                                                                   "seed run success %", "v1", "v3", "MDE avg@3 vs avg@3 (pts)", "MDE single-run dev (pts)"], rows), ""]
    rows = []
    for b in BENCHES:
        seed_runs = res[b]["seed_run_Q"]
        rows.append([BN[b], "Seed skill", " / ".join(f"{x:.1f}" for x in seed_runs), "-", "-"])
        for name, _ in METHODS:
            m = res[b]["methods"][name]
            rows.append([BN[b], name, " / ".join(f"{x:.1f}" for x in m["run_Q"]), f"{m['run_Q'][0] - seed_runs[0]:+.1f}", f"{m['success']['dQ_table1']:+.1f}"])
    L += ["### e) Q per run (rep 0 / 1 / 2) and first-run gain", "", md(["Bench", "Method", "Q per run", "first-run dQ", "avg@3 dQ"], rows), ""]
    sign_rows, summary = res["_sign"]["rows"], res["_sign"]["summary"]
    pick = {(r["bench"], r["method"], r["k"]): r for r in sign_rows}
    L += ["### f) Per-run sign agreement with the avg@3 difference", "",
          md(["run k", "disagree vs seed run k", "tie vs seed run k", "disagree vs seed mean", "tie vs seed mean"],
             [[k, v["disagree_vs_seed_run_k"], v["tie_vs_seed_run_k"], v["disagree_vs_seed_mean"], v["tie_vs_seed_mean"]] for k, v in summary.items()]), "",
          md(["Bench", "Method", "avg@3 dQ", "run0 - seed run0", "run1 - seed run1", "run2 - seed run2", "run0 - seed mean", "run1 - seed mean",
              "run2 - seed mean"], [[BN[b], name, f"{pick[b, name, 0]['avg3_dQ']:+.1f}"] + [f"{pick[b, name, k]['d_vs_seed_run_k']:+.1f}" for k in range(3)]
                                    + [f"{pick[b, name, k]['d_vs_seed_mean']:+.1f}" for k in range(3)] for b in BENCHES for name, _ in METHODS[1:]]), ""]
    L += ["### g) Dev-to-test shrinkage (SkillOpt, GEPA)", "",
          md(["Bench", "Method", "dev size", "dev seed", "dev selected", "dev gain (tasks)", "dev gain (pts)", "test dQ avg@3 (pts)"],
             [[BN[r["bench"]], r["method"], r["dev_size"], f"{100 * r['dev_seed']:.1f}", f"{100 * r['dev_selected']:.1f}", f"{r['dev_gain_tasks']:+.1f}",
               f"{r['dev_gain_pts']:+.1f}", f"{r['test_dQ_avg3']:+.1f}"] for r in res["_dev"]]), ""]
    return "\n".join(L)


def main():
    res = compute()
    rows, bad = check_table1(res)
    sign_rows, summary = sign_agreement(res)
    res.update(_table1_check={"mismatches": bad}, _sign={"rows": sign_rows, "summary": summary}, _dev=dev_to_test(res))
    (S.cache("stats") / "stats.json").write_text(json.dumps(res, indent=1))
    check = logger.md_table(["Bench", "Method", "Q T1", "Q here", "rho_trj T1", "rho_trj here", "rho_inv T1", "rho_inv here", "n common T1 / here",
                             "status"], rows)
    numbers = (f"Bootstrap: {S.N_BOOT:,} task resamples, seed {S.SEED}, percentile {100 * S.LEVEL:.0f}% intervals.\n\n"
               f"### Table 1 reproduction check\n\n{check}\n\nMismatches: {bad}.\n\n{markdown(res)}")
    logger.record(__spec__.name, "stats", numbers, S.runs_text([(MODEL, b, m) for b in BENCHES for m in ("native-seed",) + tuple(m for _, m in METHODS)]))
    print("table1 mismatches:", bad)


if __name__ == "__main__":
    main()
