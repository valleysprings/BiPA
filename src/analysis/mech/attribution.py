"""Paper 1 (mechanism paper): the four-term attribution of the input-token change of each baseline skill, computed on Table 1's own data (three test runs per side).

Role in the argument: Table 1 reports log rho = mean over tasks of log(Xbar_skill / Xbar_seed), with Xbar the per-task mean over three test runs. The older
appendix figure (analysis.mechanism.appc) used one run per side, so its total did not equal Table 1. Here the split is made on the same run means:

  A) headline, "3-run means": per task and side, R (requests), N (executed calls), X (input tokens) and F (fixed prefix per request, slopes_checks.metrics)
     are averaged over the three runs; h is defined by Xbar = Rbar Fbar + Rbar (Rbar - 1) / 2 h. With midpoint shares s_F, s_H = 1 - s_F, midpoint Rbar and
     e_R = 1 + s_H R / (R - 1) the per-task terms are  actions e_R dlogN,  turn-splitting -e_R dlog m (m = N / R),  content s_H dlog h,  skill text s_F dlog F.
     Totals are compared with log of Table 1's rho_trj on the same tasks.
  B) per-run stability: the same terms for the run pairs k = 0, 1, 2 (skill rep k against seed rep k, single episodes per task).
  C) mean change per request of the fixed-prefix columns (system, tool_schemas, task, skill), skill minus seed, to show the prefix term is the skill text.

Task set: Table 1's common task set (analysis.table1.mean_rows / geo), restricted to tasks with a usable graph episode (R >= 2, calls, positive history) and a
verified outcome with N, X known in all three runs of both sides. Dropped tasks are counted with their reason.
No BACH, no model calls. Runs: native-20261008-https + test-reps rep 1/2 (skills), native-seed run + rep-1/rep-2 (seed), as Table 1.

Commands (from bipa):           .venv/bin/python -m src.analysis.mech.attribution compute   -> assets/cache/analysis/mech/attribution.json
                                python3 -m src.analysis.mech.attribution draw -> fig/mech/mech_attribution3.{pdf,svg}, data/mech/mech_attribution3.md
"""
import argparse
import json
import math

import numpy as np

from src.analysis import main as T
from src.utils.dataloader import BENCHES, EX, SEED_TEXT, current_run, final_skill, vid
from src.utils import fig, logger, dataloader
MODEL = "deepseek-flash"
OPT = [("SkillOpt", "skillopt"), ("SkillAdam", "skilladam"), ("GEPA", "gepa")]
BENCH = [("tau3-bench", "tau3"), ("automationbench", "AutoB"), ("terminal-bench-2-1", "TB2")]
TERMS = ["more_actions", "turn_splitting", "content", "skill_text"]
TERM_LABEL = dict(more_actions="more actions", turn_splitting="turn-splitting", content="content per turn", skill_text="skill text")
PREFIX = ["system", "tool_schemas", "task", "skill"]


def run_files(method, bench):
    """({rep: evaluation.json} of the skill, the same for the seed) exactly as Table 1 picks them."""
    seed_run = current_run("native-seed", MODEL, bench)
    seed = T.rep_files([seed_run, seed_run.parent / "rep-1", seed_run.parent / "rep-2"], vid(SEED_TEXT))
    run = current_run(method, MODEL, bench)
    text = final_skill(method, run)
    own = T.rep_files([run, EX / MODEL / "test-reps" / f"{bench}__{method}__{run.name}"], vid(text))
    return own, seed


def load_runs(files):
    """{rep: {task: (Episode, metrics dict or None)}} for the graph episodes of each run file; also the per-run Table 1 rows."""
    from src.analysis.mechanism.slopes_checks import metrics
    from src.graph import episode as W
    episodes = {}
    rows = {}
    for rep, path in sorted(files.items()):
        episodes[rep] = {e.task: (e, metrics(e)) for e in W.load(path)}
        rows[rep] = {t: r for (t, _), r in T.rows_of(path).items()}
    return episodes, rows


def mean_metrics(per_run):
    """Metrics of one task and side from its per-run metric dicts: R, N, X, F averaged over runs; h from Xbar = Rbar Fbar + Rbar (Rbar - 1) / 2 h; None if no positive history."""
    R = float(np.mean([m["R"] for m in per_run]))
    N = float(np.mean([m["N"] for m in per_run]))
    X = float(np.mean([m["X"] for m in per_run]))
    F = float(np.mean([m["F"] for m in per_run]))
    history = X - R * F
    if history <= 0 or R <= 1:
        return None
    return dict(R=R, N=N, X=X, F=F, h=history / (R * (R - 1) / 2), sF=R * F / X)


def four_terms(pairs):
    """Per-task four terms of appc.compute_attr on a list of (seed metrics, skill metrics) pairs: arrays of the terms, the measured dlogX and e_R."""
    def d(key):
        return np.array([math.log(b[key] / a[key]) for a, b in pairs])
    dR, dN, dh, dX, dF = d("R"), d("N"), d("h"), d("X"), d("F")
    dm = dN - dR
    sF = np.array([(a["sF"] + b["sF"]) / 2 for a, b in pairs])
    sH = 1 - sF
    Rm = np.array([(a["R"] + b["R"]) / 2 for a, b in pairs])
    eR = 1 + sH * Rm / (Rm - 1)
    terms = dict(more_actions=eR * dN, turn_splitting=-eR * dm, content=sH * dh, skill_text=sF * dF)
    return terms, dX


def summarize(terms, dX):
    """Task means of the terms, of the measured total, and the residual (total minus sum of terms, mean and max abs per task)."""
    out = {k: float(v.mean()) for k, v in terms.items()}
    out["total"] = float(dX.mean())
    resid = dX - sum(terms.values())
    out["residual_mean"] = float(resid.mean())
    out["residual_max_abs"] = float(np.abs(resid).max())
    out["largest"] = max(TERMS, key=lambda k: abs(out[k]))
    return out


def prefix_columns(own_eps, seed_eps, tasks):
    """C) mean per-request tokens of each fixed-prefix column (skill minus seed), per task mean over the runs, then mean over tasks."""
    from src.graph import episode as W

    def column_means(eps_by_rep, task):
        vals = {c: [] for c in PREFIX}
        for rep in eps_by_rep:
            ep = eps_by_rep[rep][task][0]
            for c in PREFIX:
                if c in ep.comps:
                    vals[c].append(float(ep.Xrg[:, ep.comps.index(c)].mean()))
                else:
                    vals[c].append(0.0)
        return {c: float(np.mean(v)) for c, v in vals.items()}

    assert set(PREFIX) <= set(W.FIXED), W.FIXED
    diffs = {c: [] for c in PREFIX}
    seed_level = {c: [] for c in PREFIX}
    for task in tasks:
        a = column_means(seed_eps, task)
        b = column_means(own_eps, task)
        for c in PREFIX:
            diffs[c].append(b[c] - a[c])
            seed_level[c].append(a[c])
    return {c: dict(delta=float(np.mean(diffs[c])), seed=float(np.mean(seed_level[c]))) for c in PREFIX}


def compute():
    """All cells: A (3-run means), B (per run), C (prefix columns), Table 1 consistency, dropped-task accounting."""
    cells = []
    for method_disp, method in OPT:
        for bench, short in BENCH:
            own_files, seed_files = run_files(method, bench)
            own_eps, own_rows = load_runs(own_files)
            seed_eps, seed_rows = load_runs(seed_files)
            reps = sorted(own_files)
            assert reps == [0, 1, 2] and sorted(seed_files) == [0, 1, 2], (method, bench, reps)
            # Table 1's own common task set (its mean_rows + geo rule)
            _, own_mean = T.mean_rows(own_files)
            _, seed_mean = T.mean_rows(seed_files)
            t1_tasks = [t for (t, _), r in own_mean.items() if (t, 0) in seed_mean and all(x and x > 0 for x in (r["N"], r["X"], seed_mean[(t, 0)]["N"], seed_mean[(t, 0)]["X"]))]
            t1_log_rho = float(np.mean([math.log(own_mean[(t, 0)]["X"] / seed_mean[(t, 0)]["X"]) for t in t1_tasks]))
            usable, dropped = [], {"no verified outcome / N, X in all six runs": 0, "no graph episode in some run": 0, "episode without usable metrics (R<2, no call, no history)": 0, "no positive history on the 3-run mean": 0}
            per_side = {}
            for t in t1_tasks:
                # all six runs need a Table 1 row with outcome, N and X
                rows_ok = all(rows[r].get(t) and rows[r][t]["success"] is not None and rows[r][t]["N"] is not None and rows[r][t]["X"] is not None
                              for rows in (own_rows, seed_rows) for r in reps)
                if not rows_ok:
                    dropped["no verified outcome / N, X in all six runs"] += 1
                    continue
                if not all(t in eps[r] for eps in (own_eps, seed_eps) for r in reps):
                    dropped["no graph episode in some run"] += 1
                    continue
                if not all(eps[r][t][1] for eps in (own_eps, seed_eps) for r in reps):
                    dropped["episode without usable metrics (R<2, no call, no history)"] += 1
                    continue
                a = mean_metrics([seed_eps[r][t][1] for r in reps])
                b = mean_metrics([own_eps[r][t][1] for r in reps])
                if a is None or b is None:
                    dropped["no positive history on the 3-run mean"] += 1
                    continue
                usable.append(t)
                per_side[t] = (a, b)
            # A) 3-run means
            terms, dX = four_terms([per_side[t] for t in usable])
            A = summarize(terms, dX)
            # Table 1 consistency: log rho on the usable tasks from Table 1's own rows, and on its full set
            log_rho_usable = float(np.mean([math.log(own_mean[(t, 0)]["X"] / seed_mean[(t, 0)]["X"]) for t in usable]))
            # B) per run pair
            B = []
            for r in reps:
                pairs = [(seed_eps[r][t][1], own_eps[r][t][1]) for t in usable]
                pair_terms, pair_dX = four_terms([(dict(p[0], sF=p[0]["sF"]), p[1]) for p in pairs])
                B.append(dict(run=r, **summarize(pair_terms, pair_dX)))
            C = prefix_columns(own_eps, seed_eps, usable)
            cells.append(dict(method=method_disp, bench=bench, short=short, run=current_run(method, MODEL, bench).name, n=len(usable), n_table1=len(t1_tasks),
                              dropped=dropped, A=A, B=B, same_largest_all_runs=len({b["largest"] for b in B} | {A["largest"]}) == 1,
                              table1_log_rho_usable=log_rho_usable, table1_log_rho_full=t1_log_rho, table1_rho_full=math.exp(t1_log_rho),
                              total_gap_to_table1=A["total"] - log_rho_usable, C=C))
            print(method_disp, bench, "n", len(usable), "of", len(t1_tasks), {k: round(v, 3) for k, v in A.items() if k != "largest"}, A["largest"], "T1 usable", round(log_rho_usable, 4), "T1 full", round(t1_log_rho, 4))
    dataloader.write_json("attribution", dict(model=MODEL, cells=cells))
    return cells


def draw_heatmap(cells):
    """Heatmap in the style of appC_attribution: rows = skills grouped by optimizer (benchmark labels), columns = four terms + measured total, RdBu_r centred at 0."""
    import matplotlib
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    head = ["more\nactions", "turn-\nsplitting", "content\nper turn", "skill\ntext", "total\n(measured)"]
    bshort = {"tau3-bench": r"$\tau^3$", "automationbench": "AutoBench", "terminal-bench-2-1": "TB2"}
    keys = TERMS + ["total"]
    M = np.array([[c["A"][k] for k in keys] for c in cells])
    methods = [c["method"] for c in cells]
    bounds = [i for i in range(1, len(cells)) if methods[i] != methods[i - 1]]
    groups = [(m, sum(i for i, x in enumerate(methods) if x == m) / methods.count(m)) for m in dict.fromkeys(methods)]
    fig, ax = plt.subplots(figsize=(3.25, 2.6))
    fig.subplots_adjust(left=0.21, right=0.86, bottom=0.01, top=0.88)
    v = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j]) > 0.62 * v else "k", fontweight="bold" if j == 4 else "normal")
    ax.set_xticks(range(5))
    ax.set_xticklabels(head, fontsize=6.4, linespacing=0.95)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", length=0, pad=2)
    # second level of row labels: benchmark names right-aligned against the cells
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([bshort[c["bench"]] for c in cells], fontsize=7)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cells), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for y in [b - 0.5 for b in bounds]:
        ax.plot([-0.5, 4.5], [y, y], color="k", lw=0.9, clip_on=False)
    # first level: optimizer name once per group, bold, far left
    for name, yc in groups:
        ax.text(-0.33, yc, name, transform=ax.get_yaxis_transform(), ha="center", va="center", rotation=90, fontsize=7, fontweight="bold", clip_on=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02, ticks=[-0.2, 0, 0.2])
    cb.ax.set_yticklabels(["-0.2", "0", "+0.2"], fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    cb.set_label("change in log input tokens", fontsize=7)
    fig.save_figure(fig, "attribution3")


def f(x, nd=3):
    return f"{x:+.{nd}f}"


def record(cells):
    """paper/assets/data/mech/mech_attribution3.md."""
    rowsA = []
    for c in cells:
        a = c["A"]
        rowsA.append([c["method"], c["short"], f"{c['n']} / {c['n_table1']}"] + [f(a[k]) for k in TERMS] + [f(a["total"]), f(c["table1_log_rho_usable"]), f(a["residual_mean"]), f"{a['residual_max_abs']:.3f}", TERM_LABEL[a["largest"]]])
    rowsB = []
    for c in cells:
        for b in c["B"]:
            rowsB.append([c["method"], c["short"], b["run"]] + [f(b[k]) for k in TERMS] + [f(b["total"]), f(b["residual_mean"]), TERM_LABEL[b["largest"]]])
    rowsS = [[c["method"], c["short"], " / ".join(TERM_LABEL[b["largest"]] for b in c["B"]), "yes" if c["same_largest_all_runs"] else "no"] for c in cells]
    rowsD = [[c["method"], c["short"], c["n_table1"] - c["n"], "; ".join(f"{k}: {v}" for k, v in c["dropped"].items() if v) or "-"] for c in cells]
    rowsC = [[c["method"], c["short"]] + [f"{c['C'][p]['delta']:+.0f} (seed {c['C'][p]['seed']:.0f})" for p in PREFIX] for c in cells]
    rowsT = [[c["method"], c["short"], f"{c['table1_rho_full']:.3f}", f"{math.exp(c['table1_log_rho_usable']):.3f}", f"{math.exp(c['A']['total']):.3f}"] for c in cells]
    what = ("Four-term split of the change in log input tokens of each baseline skill against the seed, on the same data as Table 1: per task, R, N, X, F are averaged over the three test runs of each side "
            "before the split, so the total is Table 1's log rho_trj (mean over tasks of log(Xbar_skill / Xbar_seed)). Terms per task: more actions e_R dlogN, turn-splitting -e_R dlog(N/R), "
            "content per turn s_H dlogh, skill text s_F dlogF (midpoint shares, exact e_R = 1 + s_H R/(R-1)). The residual is total minus the sum of the four terms.")
    runs = "Skills: native-20261008-https (rep 0) + test-reps rep 1/2; seed: native-seed native-20261003-trace-final + rep-1, rep-2; DeepSeek-V4.1-Flash; GEPA, SkillOpt, SkillAdam x tau3-bench, automationbench, terminal-bench-2-1."
    agg = ("Tasks: Table 1's common task set restricted to tasks with a verified outcome and N, X in all three runs of both sides and a usable graph episode in all six runs (R>=2, an executed call, positive history). "
           "A) means over tasks of the per-task terms on 3-run means. B) the same terms on single run pairs (skill rep k vs seed rep k). C) mean over tasks of the per-request tokens of each fixed-prefix column, "
           "skill minus seed (per-task mean over the three runs).")
    table = ("### A. 3-run means (log units; n = tasks used / Table 1 common tasks)\n\n"
             + logger.md_table(["Skill", "Bench", "n", "actions", "turn-split", "content", "skill text", "total", "Table 1 log rho (same tasks)", "resid mean", "resid max |.|", "largest"], rowsA)
             + "\n\n### B. Per run pair (skill rep k vs seed rep k)\n\n"
             + logger.md_table(["Skill", "Bench", "run", "actions", "turn-split", "content", "skill text", "total", "resid mean", "largest"], rowsB)
             + "\n\nStability of the largest term (runs 0 / 1 / 2):\n\n" + logger.md_table(["Skill", "Bench", "largest per run", "same in all three (and in A)"], rowsS)
             + "\n\n### C. Fixed-prefix columns, change per request in tokens (skill minus seed)\n\n" + logger.md_table(["Skill", "Bench", "system", "tool_schemas", "task", "skill"], rowsC)
             + "\n\n### Table 1 consistency (rho_trj)\n\n" + logger.md_table(["Skill", "Bench", "Table 1 rho (full common set)", "rho on used tasks (Table 1 rows)", "exp(total) of A"], rowsT)
             + "\n\n### Dropped tasks (Table 1 common set minus used)\n\n" + logger.md_table(["Skill", "Bench", "dropped", "reason"], rowsD))
    logger.write_record("attribution3", what, runs, agg, table,
                    ".venv/bin/python -m src.analysis.mech.attribution compute\npython3 -m src.analysis.mech.attribution draw")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["compute", "draw"])
    step = ap.parse_args().step
    if step == "compute":
        compute()
        return
    cells = json.loads((dataloader.CACHE / "attribution.json").read_text())["cells"]
    draw_heatmap(cells)
    record(cells)


if __name__ == "__main__":
    main()
