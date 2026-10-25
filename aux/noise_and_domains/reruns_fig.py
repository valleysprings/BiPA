"""The rerun figure: every test run of the seed and of each returned skill, success and input tokens against the seed, both models.

DeepSeek-V4.1-Flash: appb.compute_reruns (the reruns of the Table 1 runs, checked against its frozen values). GPT-6-Luna: the Table 1 runs
(layout.run_dir, the tags of configs/experiments.yaml) and their test repetitions, read like Table 1's Luna cells (rep_groups,
merged_rows_of with luna_row). Per-run success = 100 * solved / verified in that run. Per-run input-token ratio = geometric mean over tasks of X_run,t over the
geometric mean of the seed's runs on t, over tasks with a verified outcome and X > 0 whose seed has X > 0 in at least one run. Bars: the mean of
a skill's runs, arithmetic for success, geometric for the token ratio; the grey line and band are the seed's mean and range.
Run:  python -m aux.noise_and_domains.reruns_fig   -> CACHE/analysis/appb/reruns_models.json, fig/mech/mech_reruns3.{pdf,svg},
      data/mech/mech_reruns3_numbers.json
"""
import json, math

import numpy as np

from aux.noise_and_domains import appb
from src import layout
from src.analysis.common import accumulate, geo, luna_row, merged_rows_of, rep_groups
from src.utils import logger
from src.utils.dataloader import SEED_TEXT, final_skill, gmean, out_dir, vid
from src.utils.fig import save_figure

LUNA = "gpt-6-luna"
OPT = (("GEPA", "gepa"), ("SkillOpt", "skillopt"), ("SkillAdam", "skilladam"))
# Table 1's Luna cells (Q, rho_X), recomputed below from the same runs
LUNA_TABLE1 = {"tau3-bench": {"seed": (69.3, 1.00), "GEPA": (65.3, 1.04), "SkillOpt": (63.1, 1.01), "SkillAdam": (64.0, 0.98)},
               "automationbench": {"seed": (37.2, 1.00), "GEPA": (41.4, 1.07), "SkillOpt": (42.8, 1.25), "SkillAdam": (38.6, 0.93)}}

PRINT_PT = 455.24   # the printed width of the figure, in pt
MIN_PT = 6.5
NOTE = "input-token row: bars and grey line are geometric means"
OC = {"SkillOpt": "#0072B2", "SkillAdam": "#D55E00", "GEPA": "#009E73", "seed": "0.45"}
OM = {"SkillOpt": "o", "SkillAdam": "s", "GEPA": "^", "seed": "o"}
BENCH = [("terminal-bench-2-1", "TB2"), ("tau3-bench", "τ³-bench"), ("automationbench", "AutoBench")]
MODELS = [("deepseek-flash", "DeepSeek-V4.1-Flash"), (LUNA, "GPT-6-Luna")]
GREY = "0.5"


def deepseek():
    out = appb.compute_reruns()
    return {bench: {"n_tasks": out[bench]["n_tasks"],
                    "skills": {s: {"succ": out[bench]["skills"][s]["succ"], "rho": out[bench]["skills"][s]["rho_pool"]}
                               for s in ("seed", "GEPA", "SkillOpt", "SkillAdam")}}
            for bench, _ in appb.BENCH}


def reps(run, text):
    groups = rep_groups([run, layout.repetition_dir(run)], vid(text))
    assert sorted(groups) == [0, 1, 2], (run, sorted(groups))
    return {k: merged_rows_of(fs, luna_row) for k, fs in sorted(groups.items())}


def luna():
    res = {}
    for bench, _ in appb.BENCH:
        seed_rows = reps(layout.run_dir("seed", bench, LUNA), SEED_TEXT)
        pooled = {}
        for rows in seed_rows.values():
            for (t, _), r in rows.items():
                if r["success"] is not None and r["X"] and r["X"] > 0:
                    pooled.setdefault(t, []).append(r["X"])
        pooled = {t: gmean(v) for t, v in pooled.items()}
        ref = accumulate(list(seed_rows.values()))
        cell = {"n_tasks": len(next(iter(seed_rows.values()))), "skills": {}}
        arms = [("seed", seed_rows)]
        for name, method in OPT:
            run = layout.run_dir(method, bench, LUNA)
            text = final_skill(method, run)
            if text == SEED_TEXT:
                cell["skills"][name] = {"returned_seed": True}
            else:
                arms.append((name, reps(run, text)))
        for name, rows in arms:
            succ, rho, nver, nrat = [], [], [], []
            for k in sorted(rows):
                rr = rows[k]
                ver = [r for r in rr.values() if r["success"] is not None]
                succ.append(round(100 * sum(r["success"] is True for r in ver) / len(ver), 4))
                logs = [math.log(r["X"] / pooled[t]) for (t, _), r in rr.items() if r["success"] is not None and r["X"] and r["X"] > 0 and t in pooled]
                rho.append(round(float(np.exp(np.mean(logs))), 4))
                nver.append(len(ver)); nrat.append(len(logs))
            _, g = geo(accumulate(list(rows.values())), ref, turns=True)
            q = sum(succ) / len(succ)
            cell["skills"][name] = {"succ": succ, "rho": rho, "n_verified": nver, "n_ratio": nrat, "Q": round(q, 4), "rho_X": round(g["rho_trj"], 4)}
            if bench in LUNA_TABLE1:
                assert (round(q, 1), round(g["rho_trj"], 2)) == LUNA_TABLE1[bench][name], (bench, name, q, g["rho_trj"])
        res[bench] = cell
    return res


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    return plt


def panel(ax, cell, row):
    """row 0: success; row 1: input-token ratio."""
    from matplotlib.ticker import MaxNLocator
    sk = cell["skills"]
    names = ["seed"] + [s for s in ("GEPA", "SkillOpt", "SkillAdam") if not sk[s].get("returned_seed")]
    key = "succ" if row == 0 else "rho"
    seed = sk["seed"][key]
    ax.axhspan(min(seed), max(seed), color=GREY, alpha=0.16, lw=0, zorder=0)
    ax.axhline(np.mean(seed) if row == 0 else gmean(seed), color=GREY, lw=0.8, zorder=1)
    if row == 1:
        ax.axhline(1.0, color="k", lw=0.5, ls=(0, (3, 2)), zorder=1)
    allv = []
    for i, s in enumerate(names):
        vals = sk[s][key]
        allv += vals
        n = len(vals)
        xs = i + (np.arange(n) - (n - 1) / 2) * 0.2
        ax.scatter(xs, vals, s=20, marker=OM[s], color=OC[s], zorder=4, lw=0)
        m = np.mean(vals) if row == 0 else gmean(vals)
        ax.plot([i - 0.32, i + 0.32], [m, m], color=OC[s], lw=1.6, zorder=3, solid_capstyle="butt")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names if row == 1 else [""] * len(names), rotation=25 if row == 1 else 0, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0 if row == 0 else 2.5, pad=2)
    ax.set_xlim(-0.6, max(len(names), 4) - 0.4)
    ax.spines[["top", "right"]].set_visible(False)
    if row == 0:
        lo, hi = min(allv), max(allv)
        pad = max(1.5, 0.12 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    else:
        ax.set_yscale("log")
        lo, hi = min(allv + [1.0]) / 1.05, max(allv + [1.0]) * 1.05
        ax.set_ylim(lo, hi)
        ticks = [t for t in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7) if lo <= t <= hi]
        if len(ticks) > 5:
            ticks = [t for t in ticks if round(t * 10) % 2 == 0]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.1f}x" for t in ticks])
        ax.minorticks_off()
    if len(names) == 1 and row == 0:
        ax.text(0.97, 0.5, "all three optimizers\nreturned the seed:\nno skill rows", transform=ax.transAxes, ha="right", va="center", fontsize=7, color="0.25",
                linespacing=1.25, bbox=dict(fc="white", ec="none", alpha=0.9, pad=2), zorder=5)


def draw(D):
    plt = style()
    from matplotlib.text import Text
    fig, A = plt.subplots(5, 3, figsize=(6.3, 5.4), gridspec_kw=dict(height_ratios=[1, 1, 0.42, 1, 1], hspace=0.2, wspace=0.3, left=0.085, right=0.995,
                                                                     bottom=0.075, top=0.9))
    for ax in A[2]:
        ax.set_visible(False)
    axs = np.array([A[0], A[1], A[3], A[4]])
    for g, (model, mdisp) in enumerate(MODELS):
        for j, (bench, bdisp) in enumerate(BENCH):
            for row in (0, 1):
                ax = axs[2 * g + row, j]
                panel(ax, D[model][bench], row)
                if row == 0:
                    ax.set_title(bdisp, loc="left", fontweight="bold", pad=3)
        axs[2 * g, 0].set_ylabel("test success (%)")
        axs[2 * g + 1, 0].set_ylabel("input tokens / seed")
        top = axs[2 * g, 0].get_position().y1
        fig.text(0.085, top + 0.035, mdisp, fontsize=8.5, fontweight="bold", ha="left", va="bottom")
        fig.text(0.995, top + 0.035, NOTE, fontsize=7.5, color="0.25", ha="right", va="bottom")
        if g:
            y = top + 0.057
            fig.add_artist(plt.Line2D([0.02, 0.995], [y, y], color="0.75", lw=0.5, transform=fig.transFigure))
    handles = [plt.Line2D([], [], marker="o", ls="", color=OC["seed"], ms=4.5),
               plt.Rectangle((0, 0), 1, 1, fc=GREY, alpha=0.16, lw=0),
               plt.Line2D([], [], ls="-", color=GREY, lw=1.2)]
    handles += [plt.Line2D([], [], marker=OM[s], ls="", color=OC[s], ms=4.5) for s in ("GEPA", "SkillOpt", "SkillAdam")]
    handles += [plt.Line2D([], [], ls="-", color="0.2", lw=1.6)]
    labels = ["seed run", "seed: range of its 3 runs", "seed mean", "GEPA run", "SkillOpt run", "SkillAdam run", "mean of a skill's runs"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 1.0), frameon=False, ncol=7, handlelength=1.2, columnspacing=1.0,
               handletextpad=0.4, fontsize=7)
    fig.canvas.draw()
    sizes = [t.get_fontsize() * PRINT_PT / (6.3 * 72) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    assert min(sizes) >= MIN_PT, min(sizes)
    save_figure(fig, "reruns3")


def main():
    D = {"deepseek-flash": deepseek(), LUNA: luna()}
    (out_dir("appb") / "reruns_models.json").write_text(json.dumps(D, indent=1))
    draw(D)
    numbers = {}
    for model, _ in MODELS:
        for bench, _ in BENCH:
            for s, v in D[model][bench]["skills"].items():
                numbers[f"{model}|{bench}|{s}"] = "returned the seed" if v.get("returned_seed") else {
                    "success_runs": v["succ"], "success_bar_arithmetic_mean": float(np.mean(v["succ"])),
                    "token_ratio_runs": v["rho"], "token_bar_geometric_mean": gmean(v["rho"])}
    logger.DATA.mkdir(parents=True, exist_ok=True)
    (logger.DATA / "mech_reruns3_numbers.json").write_text(json.dumps(numbers, indent=1))
    for k, v in numbers.items():
        print(k, v)


if __name__ == "__main__":
    main()
