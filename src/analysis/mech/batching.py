"""Paper 1 (mechanism paper), design principle "issue independent calls in one turn": what does spreading the same tool calls over more turns cost in input tokens?

Role in the argument: the harness re-sends its history every turn, so input tokens X grow faster than turns R (turn elasticity e_R between 1 and 2).  An editor that
writes rules which cut cost should therefore batch independent calls: the same N calls in fewer turns (higher calls per turn m = N / R) bill fewer tokens.  This module
measures that on the recorded test episodes without re-running anything.

Algorithm.
 1. Regression.  Per benchmark the test episodes of the seed (three runs) and of the three learned skills GEPA, SkillOpt, SkillAdam (three runs each) are pooled.  With task
    fixed effects (and a skill fixed effect, seed included, in the headline fit)   log X = a_task [+ c_skill] + b_N log N + b_R log R.   b_R is the extra input of spreading the
    same calls over more turns; b_N the input of more calls in the same number of turns.  The same data are also fitted as  log X = a_task [+ c_skill] + g_N log N + g_m log m
    with m = N / R; this is the identical model re-parameterised (g_m = -b_R, g_N = b_N + b_R) and reads as "calls per turn at fixed calls".  Standard errors are clustered by task.
 2. Counterfactual inside the measurement.  A learned skill keeps its calls N on each task but is given the seed's calls per turn m_seed on that task, so its turns become
    R_cf = N / m_seed.  With the fitted b_R, log X falls by b_R log(R / R_cf) = -b_R (log m_skill - log m_seed) on that task (negative of a fall when the skill batches more than the seed).
    The task mean of this is the part of the skill's token tax log rho that disappears.  It is compared with the turn-splitting term of the four-term split of mech_attribution3,
    -e_R dlog m, computed on the same tasks and 3-run means: both measure the effect of a change in calls per turn, one with the within-task regression slope b_R, the other with the
    per-episode accounting elasticity e_R.
Everything uses recorded episodes; a skill that really batched would also change what the agent does next, so the numbers are accounting, not an experiment.

Commands (from bipa):           .venv/bin/python -m src.analysis.mech.batching compute  -> assets/cache/analysis/mech/batching.json
                               python3 -m src.analysis.mech.batching draw -> fig/mech/mech_batching.{pdf,svg}, data/mech/mech_batching.md
"""
import argparse
import json
import math

import numpy as np

from src.analysis import main as T
from src.analysis.mech import attribution as A
from src.utils import fig, logger, dataloader
BENCH_NAME = {"tau3-bench": r"$\tau^3$", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}


def fe_fit(task, skill, columns, y):
    """Within-task OLS of y on the columns plus skill dummies, task-clustered standard errors.

    task, skill: label arrays; columns: (n, k) array of regressors; y: (n,).  Task fixed effects are removed by demeaning every variable inside its task (Frisch-Waugh);
    the skill fixed effect enters as dummies (first level dropped) that are demeaned the same way.  Returns the coefficients and clustered SEs of the k regressors."""
    skill_levels = sorted(set(skill))[1:]
    dummies = np.array([[1.0 if s == level else 0.0 for level in skill_levels] for s in skill]).reshape(len(y), len(skill_levels))
    design = np.hstack([columns, dummies]) if skill_levels else columns
    task_ids = np.unique(task, return_inverse=True)[1]
    design = design.astype(float)
    y = y.astype(float)
    for t in np.unique(task_ids):
        rows = task_ids == t
        design[rows] -= design[rows].mean(axis=0)
        y[rows] -= y[rows].mean()
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    bread = np.linalg.inv(design.T @ design)
    meat = np.zeros((design.shape[1], design.shape[1]))
    for t in np.unique(task_ids):
        rows = task_ids == t
        score = design[rows].T @ resid[rows]
        meat += np.outer(score, score)
    n_tasks = len(np.unique(task_ids))
    # small-sample factor of the usual cluster-robust estimator
    factor = n_tasks / (n_tasks - 1) * (len(y) - 1) / (len(y) - design.shape[1])
    se = np.sqrt(np.diag(bread @ meat @ bread) * factor)
    k = columns.shape[1]
    return beta[:k], se[:k], dict(n=len(y), tasks=n_tasks)


def regressions(records):
    """Fits of one episode pool: [(N, R) fit with skill FE, (N, m) fit with skill FE, (N, R) fit without skill FE] as dicts."""
    task = np.array([r["task"] for r in records])
    skill = np.array([r["skill"] for r in records])
    log_n = np.log([r["N"] for r in records])
    log_r = np.log([r["R"] for r in records])
    log_x = np.log([r["X"] for r in records])
    log_m = log_n - log_r
    out = {}
    for name, columns, with_skill in (("NR", np.column_stack([log_n, log_r]), True),
                                      ("Nm", np.column_stack([log_n, log_m]), True),
                                      ("NR_noskill", np.column_stack([log_n, log_r]), False)):
        labels = skill if with_skill else np.zeros(len(task))
        beta, se, size = fe_fit(task, labels, columns, log_x)
        out[name] = dict(coef=beta.tolist(), se=se.tolist(), **size)
    return out


def bench_numbers(bench):
    """Episode records of the seed and the three skills on one benchmark, and the counterfactual of each skill."""
    records = []
    cells = []
    seed_added = False
    for method_disp, method in A.OPT:
        own_files, seed_files = A.run_files(method, bench)
        own_eps, own_rows = A.load_runs(own_files)
        seed_eps, seed_rows = A.load_runs(seed_files)
        reps = sorted(own_files)
        # Table 1's common task set, then tasks with usable metrics in all six runs (as mech_attribution3)
        _, own_mean = T.mean_rows(own_files)
        _, seed_mean = T.mean_rows(seed_files)
        common = [t for (t, _), r in own_mean.items() if (t, 0) in seed_mean and all(x and x > 0 for x in (r["N"], r["X"], seed_mean[(t, 0)]["N"], seed_mean[(t, 0)]["X"]))]
        usable = []
        pairs = {}
        for t in common:
            rows_ok = all(rows[r].get(t) and rows[r][t]["success"] is not None and rows[r][t]["N"] is not None and rows[r][t]["X"] is not None
                          for rows in (own_rows, seed_rows) for r in reps)
            if not rows_ok:
                continue
            if not all(t in eps[r] and eps[r][t][1] for eps in (own_eps, seed_eps) for r in reps):
                continue
            a = A.mean_metrics([seed_eps[r][t][1] for r in reps])
            b = A.mean_metrics([own_eps[r][t][1] for r in reps])
            if a is None or b is None:
                continue
            usable.append(t)
            pairs[t] = (a, b)
        # episode records for the regression: every graph episode with usable metrics
        for eps, label in ((own_eps, method_disp), (seed_eps, "seed")):
            if label == "seed" and seed_added:
                continue
            for r in reps:
                for t, (_, m) in eps[r].items():
                    if m:
                        records.append(dict(task=f"{bench}:{t}", skill=label, N=m["N"], R=m["R"], X=m["X"]))
        seed_added = True
        terms, d_x = A.four_terms([pairs[t] for t in usable])
        d_log_m = np.array([math.log((pairs[t][1]["N"] / pairs[t][1]["R"]) / (pairs[t][0]["N"] / pairs[t][0]["R"])) for t in usable])
        cells.append(dict(method=method_disp, bench=bench, n=len(usable), log_rho=float(d_x.mean()), turn_splitting_term=float(terms["turn_splitting"].mean()),
                          d_log_m=d_log_m.tolist(), dlog_x=d_x.tolist(), e_r_dlog_m=(-terms["turn_splitting"]).tolist()))
        print(method_disp, bench, "tasks", len(usable), "log rho", round(float(d_x.mean()), 3), "turn-splitting", round(float(terms["turn_splitting"].mean()), 3))
    return records, cells


def compute():
    """Fit per benchmark and pooled, apply the counterfactual with the benchmark's b_R, write batching.json."""
    fits = {}
    cells = []
    pooled_records = []
    for bench, _ in A.BENCH:
        records, bench_cells = bench_numbers(bench)
        pooled_records += records
        fits[bench] = regressions(records)
        cells += bench_cells
    fits["pooled"] = regressions(pooled_records)
    for cell in cells:
        for source in (cell["bench"], "pooled"):
            b_r = fits[source]["NR"]["coef"][1]
            # fall of log X when the skill keeps the seed's calls per turn: -b_R dlog m, task mean
            removed = float(np.mean(-b_r * np.array(cell["d_log_m"])))
            key = "removed" if source == cell["bench"] else "removed_pooled_slope"
            cell[key] = removed
            cell["log_rho_cf" if source == cell["bench"] else "log_rho_cf_pooled_slope"] = cell["log_rho"] - removed
    dataloader.write_json("batching", dict(model=A.MODEL, fits=fits, cells=cells))
    for name, fit in fits.items():
        print(name, {k: [round(c, 3) for c in v["coef"]] for k, v in fit.items()}, "n", fit["NR"]["n"])


def mean_over(cells, key):
    """Mean of a cell field over a list of cells."""
    return float(np.mean([c[key] for c in cells]))


def panel(ax, data):
    """Compact panel: per benchmark (and all nine skills) the mean token tax log rho of the three skills as recorded and with the seed's calls per turn restored."""
    groups = [(BENCH_NAME[b], [c for c in data["cells"] if c["bench"] == b]) for b, _ in A.BENCH]
    groups.append(("all", data["cells"]))
    x = np.arange(len(groups))
    width = 0.38
    measured = [mean_over(cells, "log_rho") for _, cells in groups]
    counterfactual = [mean_over(cells, "log_rho_cf") for _, cells in groups]
    ax.bar(x - width / 2, measured, width, color="#9e9e9e", label="as recorded")
    ax.bar(x + width / 2, counterfactual, width, color="#2166ac", label="at seed's calls per turn")
    ax.axhline(0, color="0.3", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in groups], fontsize=7)
    ax.set_ylabel(r"token tax ($\log\rho$)", fontsize=7, labelpad=1)
    ax.tick_params(axis="y", labelsize=7, length=2)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.set_ylim(min(0, min(counterfactual)) - 0.01, max(measured) * 1.75)
    ax.legend(fontsize=6, frameon=False, loc="upper left", handlelength=0.9, borderaxespad=0.1, labelspacing=0.2, handletextpad=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw(data):
    """Standalone 3.25 in version of the panel, saved through common.save_figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.25, 1.7))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.12, top=0.97)
    panel(ax, data)
    fig.save_figure(fig, "batching")
    # the 1x4-width version, to check legibility at the size the paper uses
    fig_small, ax_small = plt.subplots(figsize=(1.65, 1.7))
    fig_small.subplots_adjust(left=0.27, right=0.98, bottom=0.12, top=0.97)
    panel(ax_small, data)
    fig_small.savefig(dataloader.CACHE / "mech_batching_small.png", dpi=250)


def record(data):
    """Audit record mech_batching.md."""
    fits = data["fits"]
    names = {"pooled": "pooled"}
    names.update({b: n for b, n in (("tau3-bench", "tau3"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2"))})
    rows = []
    for key, name in names.items():
        f = fits[key]
        rows.append([name, f["NR"]["n"], f["NR"]["tasks"], "%.3f (%.3f)" % (f["NR"]["coef"][0], f["NR"]["se"][0]), "%.3f (%.3f)" % (f["NR"]["coef"][1], f["NR"]["se"][1]),
                     "%.3f (%.3f)" % (f["Nm"]["coef"][0], f["Nm"]["se"][0]), "%.3f (%.3f)" % (f["Nm"]["coef"][1], f["Nm"]["se"][1]),
                     "%.3f (%.3f)" % (f["NR_noskill"]["coef"][0], f["NR_noskill"]["se"][0]), "%.3f (%.3f)" % (f["NR_noskill"]["coef"][1], f["NR_noskill"]["se"][1])])
    t1 = logger.md_table(["pool", "episodes", "tasks", "b_N", "b_R", "g_N (log N)", "g_m (log N/R)", "b_N no skill FE", "b_R no skill FE"], rows)
    cell_rows = []
    for c in data["cells"]:
        cell_rows.append([c["method"], BENCH_NAME[c["bench"]].replace("$", "").replace("\\", ""), c["n"], "%+.3f" % c["log_rho"], "%+.3f" % c["removed"], "%+.3f" % c["log_rho_cf"],
                          "%+.3f" % c["turn_splitting_term"], "%+.3f" % c["removed_pooled_slope"]])
    t2 = logger.md_table(["Skill", "Bench", "tasks", "log rho", "removed by seed's m (b_R)", "log rho after", "turn-splitting term (mech_attribution3 rule)", "removed with pooled b_R"], cell_rows)
    summary = []
    for b, _ in A.BENCH:
        cs = [c for c in data["cells"] if c["bench"] == b]
        summary.append([b, "%+.3f" % mean_over(cs, "log_rho"), "%+.3f" % mean_over(cs, "removed"), "%+.3f" % mean_over(cs, "turn_splitting_term")])
    summary.append(["all nine", "%+.3f" % mean_over(data["cells"], "log_rho"), "%+.3f" % mean_over(data["cells"], "removed"), "%+.3f" % mean_over(data["cells"], "turn_splitting_term")])
    t3 = logger.md_table(["Bench", "mean log rho", "mean removed (b_R)", "mean turn-splitting term"], summary)
    table = ("### Within-task regression log X = a_task + c_skill + b_N log N + b_R log R (task-clustered SE in brackets)\n\n" + t1 +
             "\n\n### Counterfactual: each skill keeps the seed's calls per turn on every task\n\n" + t2 + "\n\n### Benchmark means against the turn-splitting term\n\n" + t3)
    logger.write_record(
        "batching",
        "Design principle 'issue independent calls in one turn'. Within a task, how many more input tokens does an episode bill when it spreads the same calls over more turns (b_R), and how much "
        "of each learned skill's token tax would disappear if it kept the seed's calls per turn on every task.  The counterfactual is compared with the turn-splitting term of mech_attribution3.",
        "DeepSeek-V4.1-Flash in OpenCode, test split: seed (3 runs) and GEPA, SkillOpt, SkillAdam (3 runs each) on tau3-bench, automationbench, terminal-bench-2-1; same run files as mech_attribution3.",
        "Episodes with R>=2, a call and positive history (slopes_checks.metrics). Regression: task fixed effects by within-task demeaning, skill fixed effect (seed + three skills) as dummies, SE clustered by task. "
        "The N, N/R fit is the same model re-parameterised (g_m = -b_R). Counterfactual: R_cf = N / m_seed on 3-run task means, removed = mean over tasks of -b_R * (log m_skill - log m_seed), "
        "using the benchmark's own b_R; tasks and 3-run means are those of mech_attribution3. Accounting only, no episode is re-run.",
        table,
        ".venv/bin/python -m src.analysis.mech.batching compute\npython3 -m src.analysis.mech.batching draw")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["compute", "draw"])
    step = ap.parse_args().step
    if step == "compute":
        compute()
        return
    data = json.loads((dataloader.CACHE / "batching.json").read_text())
    draw(data)
    record(data)


if __name__ == "__main__":
    main()
