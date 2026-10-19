"""Paper 1 (mechanism paper), reviewer X1: how large would the token tax be under a harness that compacts history?  A counterfactual that holds the trajectories fixed.

Role in the argument: the paper says input tokens grow with turns because every request re-sends the whole history (log X ~ e_R log R with e_R > 1). If the harness
compacted its history, the quadratic part would shrink and the token tax of a skill (rho on input tokens) should move toward its turn ratio rho_turns, i.e. the
elasticity of input tokens to turns should fall toward 1. This module recomputes every test episode's input tokens under two compaction policies and checks that.

Counterfactual, not an experiment: the agent's turns, calls and outputs are the recorded ones and are NOT re-run. Only the bill of each recorded request is
recomputed, as if the harness had sent a shorter history. A real compacting harness would change what the agent does next, so the numbers show the accounting
effect of compaction only.

Per request r of an episode (graph.episode.Episode): X_r = provider input tokens, F_r = tokens of the FIXED columns (system, tool schemas, task, skill components),
H_r = X_r - F_r = history tokens. Per-turn history increments d_r = H_r - H_{r-1} (H_0 = 0) are read off the recorded differences, so d_r is what turn r-1 added.
  (i) sliding window of k turns: H'_r = max(0, sum of the last k increments) = max(0, H_r - H_{r-k}) (H_j = 0 for j <= 0); k = inf keeps H_r.
 (ii) token cap C: H'_r = min(H_r, C); C = inf keeps H_r.
New input tokens of the episode: X' = sum_r (F_r + H'_r).  The skill text sits in F_r and is never compacted, in either policy.

Aggregation (as Table 1 / attribution): task set = Table 1's common tasks (mean_rows rule) with a graph episode in all six runs; per task the mean over the three runs of
each side; rho = geometric mean over tasks of Xbar'_skill / Xbar'_seed, the same policy applied to both sides; rho_turns = the same for the turn counts R; slope = within-task
(task fixed effects) slope of log X' on log R pooled over the six episodes of every task of the cell. At k = inf rho equals Table 1's rho_trj on these tasks.

Commands (from bipa):           .venv/bin/python -m src.analysis.mech.compaction compute  -> assets/cache/analysis/mech/compaction.json
                               python3 -m src.analysis.mech.compaction draw -> fig/mech/mech_compaction.{pdf,svg}, data/mech/mech_compaction.md
"""
import argparse
import json
import math

import numpy as np

from src.analysis import table1 as T
from src.analysis.mech import attribution as A
from src.utils import fig, logger, dataloader
WINDOWS = [1, 3, 5, 10, None]               # None = infinity
CAPS = [8000, 32000, 64000, 128000, None]   # None = infinity


def label(v):
    """Display of a policy parameter."""
    return "inf" if v is None else str(v)


def prefix_and_history(ep):
    """(F_r, H_r) arrays per request: F_r the FIXED-column tokens, H_r = X_r - F_r (never below 0)."""
    # imported here so that the draw step runs without the harness_bench package
    from src.graph import episode as W
    fixed = [i for i, c in enumerate(ep.comps) if c in W.FIXED]
    F = ep.Xrg[:, fixed].sum(axis=1) if fixed else np.zeros(ep.R)
    H = np.maximum(ep.req_in - F, 0.0)
    return F, H


def window_input(ep, k):
    """Input tokens of the episode when only the history added by the last k turns is kept (k None = all)."""
    F, H = prefix_and_history(ep)
    if k is None:
        return float((F + H).sum())
    # H shifted by k requests, zero before the start: H_{r-k}
    lagged = np.zeros_like(H)
    if k < len(H):
        lagged[k:] = H[:-k]
    kept = np.maximum(H - lagged, 0.0)
    return float((F + kept).sum())


def cap_input(ep, cap):
    """Input tokens of the episode when the history of every request is capped at cap tokens (None = no cap)."""
    F, H = prefix_and_history(ep)
    kept = H if cap is None else np.minimum(H, cap)
    return float((F + kept).sum())


def policies():
    """[(name, parameter label, function episode -> input tokens)] in report order."""
    out = []
    for k in WINDOWS:
        out.append(("window", label(k), lambda ep, k=k: window_input(ep, k)))
    for c in CAPS:
        out.append(("cap", label(c), lambda ep, c=c: cap_input(ep, c)))
    return out


def fe_slope(points):
    """Within-task slope of log y on log x over [(task, x, y)]: demean per task, tasks with fewer than two distinct x contribute nothing."""
    by_task = {}
    for task, x, y in points:
        by_task.setdefault(task, []).append((math.log(x), math.log(y)))
    sxx = sxy = 0.0
    for rows in by_task.values():
        xs = np.array([r[0] for r in rows])
        ys = np.array([r[1] for r in rows])
        xs -= xs.mean()
        ys -= ys.mean()
        sxx += float((xs * xs).sum())
        sxy += float((xs * ys).sum())
    return sxy / sxx


def cell_numbers(method_disp, method, bench):
    """Policy results of one (skill, benchmark) cell."""
    own_files, seed_files = A.run_files(method, bench)
    own_eps, _ = A.load_runs(own_files)
    seed_eps, _ = A.load_runs(seed_files)
    reps = sorted(own_files)
    _, own_mean = T.mean_rows(own_files)
    _, seed_mean = T.mean_rows(seed_files)
    common = [t for (t, _), r in own_mean.items() if (t, 0) in seed_mean and all(x and x > 0 for x in (r["N"], r["X"], seed_mean[(t, 0)]["N"], seed_mean[(t, 0)]["X"]))]
    # tasks with a graph episode in every one of the six runs
    tasks = [t for t in common if all(t in eps[r] for eps in (own_eps, seed_eps) for r in reps)]
    turns_ratio = [np.mean([own_eps[r][t][0].R for r in reps]) / np.mean([seed_eps[r][t][0].R for r in reps]) for t in tasks]
    rho_turns = float(np.exp(np.mean(np.log(turns_ratio))))
    results = []
    for name, param, fn in policies():
        log_ratio = []
        points = []
        for t in tasks:
            own_x = [fn(own_eps[r][t][0]) for r in reps]
            seed_x = [fn(seed_eps[r][t][0]) for r in reps]
            log_ratio.append(math.log(np.mean(own_x) / np.mean(seed_x)))
            for eps, xs in ((own_eps, own_x), (seed_eps, seed_x)):
                for r, x in zip(reps, xs):
                    points.append((t, eps[r][t][0].R, x))
        rho = float(np.exp(np.mean(log_ratio)))
        results.append(dict(policy=name, param=param, rho=rho, slope=fe_slope(points),
                            elasticity=math.log(rho) / math.log(rho_turns) if abs(math.log(rho_turns)) > 1e-9 else None))
    return dict(method=method_disp, bench=bench, short=dict(A.BENCH)[bench], n=len(tasks), n_table1=len(common), rho_turns=rho_turns, results=results)


def compute():
    """All nine cells."""
    cells = []
    for method_disp, method in A.OPT:
        for bench, _ in A.BENCH:
            cell = cell_numbers(method_disp, method, bench)
            cells.append(cell)
            base = [r for r in cell["results"] if r["param"] == "inf"][0]
            print(method_disp, bench, "n", cell["n"], "/", cell["n_table1"], "rho_turns", round(cell["rho_turns"], 3), "rho(inf)", round(base["rho"], 3), "slope(inf)", round(base["slope"], 3))
    dataloader.write_json("compaction", dict(model=A.MODEL, windows=[label(k) for k in WINDOWS], caps=[label(c) for c in CAPS], cells=cells))
    return cells


def pick(cell, policy, param):
    """The result dict of a cell for one policy parameter."""
    return [r for r in cell["results"] if r["policy"] == policy and r["param"] == param][0]


def draw(cells):
    """Per-skill dumbbell, rows grouped by benchmark in the row order of mech_price: rho as recorded (hollow) to rho with only the last turn kept (filled), rho_turns as a tick, reference line at 1."""
    import matplotlib
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.6})
    color = {"SkillOpt": "#0072B2", "SkillAdam": "#D55E00", "GEPA": "#009E73"}
    marker = {"SkillOpt": "o", "SkillAdam": "s", "GEPA": "^"}
    bench = [("tau3-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2")]
    order = ["GEPA", "SkillOpt", "SkillAdam"]
    by = {(c["method"], c["bench"]): c for c in cells}
    fig, ax = plt.subplots(figsize=(3.375, 2.5))
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.25, top=0.99)
    y = 0.0
    ticks, labels = [], []
    for b, disp in bench:
        ax.text(-0.36, y, disp, transform=ax.get_yaxis_transform(), ha="left", va="center", fontweight="bold", fontsize=7)
        y += 1
        for o in order:
            c = by[(o, b)]
            rec, one = pick(c, "window", "inf")["rho"], pick(c, "window", "1")["rho"]
            ax.plot([one, rec], [y, y], color=color[o], lw=1.1, zorder=2)
            ax.plot(rec, y, marker=marker[o], ms=4.5, mfc="white", mec=color[o], mew=0.9, ls="", zorder=4)
            ax.plot(one, y, marker=marker[o], ms=4.5, color=color[o], ls="", zorder=5)
            ax.plot(c["rho_turns"], y, marker="|", ms=7, mec="0.15", mew=1.3, ls="", zorder=3)
            ticks.append(y)
            labels.append(o)
            y += 1
        y += 0.4
    ax.axvline(1.0, color="0.2", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(1.0, -1.0, "seed = 1", ha="center", va="center", fontsize=7, color="0.2")
    ax.set_ylim(y - 0.3, -1.5)
    ax.set_xlim(0.97, 1.46)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2.5, width=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", ms=4.5, mew=0.9), Line2D([], [], marker="o", ls="", color="0.3", ms=4.5),
               Line2D([], [], marker="|", ls="", mec="0.15", ms=7, mew=1.3)]
    fig.legend(handles, ["as recorded", "last turn only", "turn ratio"], loc="lower center", bbox_to_anchor=(0.6, -0.005), ncol=3, frameon=False,
               fontsize=7, handlelength=1.0, columnspacing=1.0, handletextpad=0.3)
    ax.annotate("input-token ratio to the seed", xy=(0.5, -0.1), xycoords="axes fraction", ha="center", va="top", fontsize=7)
    fig.save_figure(fig, "compaction")


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def record(cells):
    """paper/assets/data/mech/mech_compaction.md."""
    sections = []
    for policy, params, title in (("window", [label(k) for k in WINDOWS], "(i) sliding window: history of the last k turns"), ("cap", [label(c) for c in CAPS], "(ii) token cap C on history per request")):
        rows_rho, rows_slope, rows_el = [], [], []
        for c in cells:
            head = [c["method"], c["short"], c["n"], fmt(c["rho_turns"])]
            rows_rho.append(head + [fmt(pick(c, policy, p)["rho"]) for p in params])
            rows_slope.append(head[:3] + [fmt(pick(c, policy, p)["slope"]) for p in params])
            rows_el.append(head[:3] + [fmt(pick(c, policy, p)["elasticity"], 2) for p in params])
        sections.append(f"### {title}\n\nrho (geometric mean over tasks of per-task 3-run means, skill vs seed)\n\n"
                        + logger.md_table(["Skill", "Bench", "n", "rho_turns"] + [f"{policy} {p}" for p in params], rows_rho)
                        + "\n\nwithin-task slope of log input on log turns\n\n" + logger.md_table(["Skill", "Bench", "n"] + [f"{policy} {p}" for p in params], rows_slope)
                        + "\n\nlog rho / log rho_turns (1 = the whole token tax is the turn ratio)\n\n" + logger.md_table(["Skill", "Bench", "n"] + [f"{policy} {p}" for p in params], rows_el))
    # mean over cells, to read the trend at a glance
    summary = []
    for policy, params in (("window", [label(k) for k in WINDOWS]), ("cap", [label(c) for c in CAPS])):
        for p in params:
            gap = [abs(math.log(pick(c, policy, p)["rho"] / c["rho_turns"])) for c in cells]
            summary.append([policy, p, fmt(float(np.mean([pick(c, policy, p)["slope"] for c in cells]))), fmt(float(np.mean(gap)))])
    sections.append("### Trend over the nine skills\n\n" + logger.md_table(["policy", "parameter", "mean slope", "mean |log(rho / rho_turns)|"], summary))
    what = ("Reviewer X1, a counterfactual: how large would the token tax be if the harness compacted its history? Each test episode's input tokens are recomputed from its per-request records under a sliding window "
            "(keep only the history added by the last k turns) and under a token cap C on history per request, with the agent's turns held fixed. The expectation is that rho moves toward rho_turns and the "
            "within-task slope of log input on log turns falls toward 1 as compaction tightens. This is an accounting counterfactual on fixed trajectories: a compacting harness would also change what the agent does next, "
            "and the agent already saw the full history in the recorded runs.")
    runs = "As mech_attribution3: skills native-20261008-https (rep 0) + test-reps rep 1/2; seed native-seed native-20261003-trace-final + rep-1, rep-2; DeepSeek-V4.1-Flash; GEPA, SkillOpt, SkillAdam x three benchmarks; per-request records via graph.episode."
    agg = ("Per request: F_r = fixed columns (system, tool schemas, task, skill), H_r = X_r - F_r, increments d_r = H_r - H_{r-1} (H_0 = 0). Window k: H'_r = max(0, H_r - H_{r-k}); cap C: H'_r = min(H_r, C); X' = sum_r (F_r + H'_r). "
           "The skill text sits in F and is never compacted. Task set: Table 1's common tasks with a graph episode in all six runs (n shown; Table 1 n in the JSON). Per task the mean over the three runs of each side; "
           "rho = geometric mean over tasks of skill / seed with the same policy on both sides; rho_turns the same on turn counts R (graph requests); slope = within-task (task fixed effects) slope of log X' on log R pooled over the "
           "six episodes of each task. Window inf and cap inf are the recorded run (rho = Table 1 rho_trj on these tasks, up to the graph-episode input sum: SkillOpt TB2 1.423 vs 1.429, GEPA TB2 1.290 on 17 of 18 tasks vs 1.338). "
           "Caution on log rho / log rho_turns: it is unstable where rho_turns is near 1 (SkillOpt tau3); the robust reading is the gap rho - rho_turns and the slope.")
    logger.write_record("compaction", what, runs, agg, "\n\n".join(sections), ".venv/bin/python -m src.analysis.mech.compaction compute\npython3 -m src.analysis.mech.compaction draw")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["compute", "draw"])
    step = ap.parse_args().step
    if step == "compute":
        compute()
        return
    cells = json.loads((dataloader.CACHE / "compaction.json").read_text())["cells"]
    draw(cells)
    record(cells)


if __name__ == "__main__":
    main()
