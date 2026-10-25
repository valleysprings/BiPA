"""mech_devsize: how the dev gain reported for identical candidates and the MDE80 of a single-run comparison change with the
size of the dev panel, under the task model of src.analysis.selection.

Run: python -m aux.selection_replay.devsize compute   -> CACHE/analysis/main/selection_devsize.json, data/mech/mech_devsize.md (reads selection.json)
     python -m aux.selection_replay.devsize draw      -> fig/mech/mech_devsize.{pdf,svg}
"""
import json
import sys

import numpy as np

from src.analysis.selection import (BENCHES, MODERATE, best_of, check_inputs, executions, fit_prior, mde80, outcomes, reseed, seed_files,
                                    task_probabilities)
from src.utils import dataloader, logger
from src.utils.fig import save_figure

DEVSIZE_DRAWS = 40_000
DEV_SIZES = (9, 15, 30, 60, 120)
DEVSIZE_RULES = (("SkillOpt", 4), ("GEPA", 7))
PRIORS = ("fitted", "moderate")


def devsize(dev, prior, index, j, baseline_quantile) -> dict:
    """Per dev size: E[announced gain] (points) of SkillOpt's best of 4 and GEPA's argmax over 7 with a random baseline and with the baseline
    at the run's own seed-draw mid-quantile, and the MDE80 of a single-run comparison (points)."""
    rows = {}
    for s, size in enumerate(DEV_SIZES):
        reseed(index, j, 100 + s)
        p = task_probabilities(dev, prior, size, DEVSIZE_DRAWS)
        base = outcomes(p)
        at_quantile = float(np.quantile(base, baseline_quantile, method="nearest"))
        row = {"baseline_at_quantile": at_quantile}
        for rule, k in DEVSIZE_RULES:
            row[rule] = {"E_gain_random_points": 100 * float(best_of(p, k, base).mean()) / size,
                         "E_gain_quantile_points": 100 * float(best_of(p, k, at_quantile).mean()) / size}
        critical, mde = mde80(p, step=0.5 if size > 30 else 0.25)
        row["MDE80_points"] = 100 * mde / size
        row["critical_net_tasks"] = critical
        rows[str(size)] = row
    return rows


def compute():
    """Per benchmark and prior, the dev-size rows from the seed-draw mid-quantile that selection.json recorded."""
    sel = json.loads((dataloader.CACHE / "selection.json").read_text())
    result = {}
    for index, (bench, _) in enumerate(BENCHES):
        dev_files, test_files = seed_files(bench)
        dev, test = executions(dev_files), executions(test_files)
        check_inputs(bench, dev, test)
        pooled = [(sum(v), len(v)) for v in list(dev.values()) + list(test.values())]
        a, b = fit_prior(pooled)
        mean = sum(k for k, _ in pooled) / sum(n for _, n in pooled)
        result[bench] = {"priors": {}, "devsize": {}}
        for j, name in enumerate(PRIORS):
            prior = (a, b) if name == "fitted" else (MODERATE * mean, MODERATE * (1 - mean))
            q = sel[bench]["priors"][name]["seed_draw_mid_quantile"]
            result[bench]["priors"][name] = {"seed_draw_mid_quantile": q}
            result[bench]["devsize"][name] = devsize(dev, prior, index, j, q)
    dataloader.write_json("selection_devsize", result)
    record_devsize(result)


def record_devsize(result):
    """The audit record data/mech/mech_devsize.md."""
    rows = []
    for bench, _ in BENCHES:
        for prior in PRIORS:
            q = result[bench]["priors"][prior]["seed_draw_mid_quantile"]
            for size in DEV_SIZES:
                r = result[bench]["devsize"][prior][str(size)]
                rows.append([bench, prior, size, f"{q:.2f} -> {r['baseline_at_quantile']:.0f}/{size}",
                             f"{r['SkillOpt']['E_gain_random_points']:.1f}", f"{r['SkillOpt']['E_gain_quantile_points']:.1f}",
                             f"{r['GEPA']['E_gain_random_points']:.1f}", f"{r['GEPA']['E_gain_quantile_points']:.1f}",
                             f"{r['MDE80_points']:.1f}", r["critical_net_tasks"]])
    logger.write_record(
        "devsize",
        "How the gain a success-only selector announces under no effect, and the smallest effect a single-run dev comparison detects, change "
        "with the size of the dev panel (SkillOpt's best of 4, GEPA's argmax over 7), in percentage points of the dev panel.",
        "The task model of mech_selection (the same seed executions and fitted / moderate priors).",
        f"Per dev size, {DEVSIZE_DRAWS} draws: the dev tasks are a random subset of the benchmark's dev tasks (posteriors) when the size is at most "
        "its dev panel, else all of them plus tasks drawn from the prior. Random baseline: the seed's score drawn like the candidates. Quantile "
        "baseline: the baseline fixed at the same mid-quantile of the seed-score distribution as the run's own seed draw at the real dev size "
        "(the quantile and the resulting score are listed). MDE80: the smallest true gain detected with probability 0.8 by a one-sided 0.05 "
        "comparison of two single runs. Random seed [20261012, benchmark index, prior index, 100 + size index].",
        logger.md_table(["bench", "prior", "dev size", "seed-draw quantile -> baseline", "SkillOpt E[gain] random (pts)", "SkillOpt E[gain] at quantile (pts)",
                    "GEPA E[gain] random (pts)", "GEPA E[gain] at quantile (pts)", "MDE80 (pts)", "critical net difference (tasks)"], rows),
        "python -m aux.selection_replay.devsize compute\npython -m aux.selection_replay.devsize draw")


def style():
    """matplotlib with TrueType fonts in the pdf (the ACL check rejects Type 3)."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    return plt


def draw_devsize(result):
    """mech_devsize, one column (3.25 in): the dev gain SkillOpt's rule (best of 4, left) and GEPA's rule (best of 7,
    right) report from a random seed draw when every candidate is identical to the seed, against the number of dev tasks (log axis), one
    coloured line per benchmark, fitted task model only (the moderate prior is given in the text).  Sizes beyond a benchmark's dev panel
    add tasks drawn from the prior (exchangeable tasks).  The dev size of the Table 1 runs (30) is marked."""
    plt = style()
    benches = (("tau3-bench", r"$\tau^3$-bench", "#0072B2", "o"), ("automationbench", "AutoBench", "#009E73", "^"),
               ("terminal-bench-2-1", "TB2", "#D55E00", "s"))
    rules = (("SkillOpt", "SkillOpt's rule (best of 4)"), ("GEPA", "GEPA's rule (best of 7)"))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "font.size": 7,
                         "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.6), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.83, bottom=0.22, top=0.88, wspace=0.12)
    for ax, (rule, title) in zip(axes, rules):
        ax.axvline(30, color="0.8", lw=0.8, zorder=0)
        for bench, label, color, marker in benches:
            ys = [result[bench]["devsize"]["fitted"][str(s)][rule]["E_gain_random_points"] for s in DEV_SIZES]
            ax.plot(DEV_SIZES, ys, color=color, lw=1.4, marker=marker, markersize=3.2, clip_on=False)
            if ax is axes[-1]:
                ax.text(DEV_SIZES[-1] * 1.18, ys[-1], label, color="0.15", fontsize=6.5, va="center", ha="left")
        ax.set_title(title, fontsize=7, pad=3)
        ax.set_xscale("log")
        ax.set_xticks(DEV_SIZES)
        ax.set_xticklabels([str(s) for s in DEV_SIZES])
        ax.minorticks_off()
        ax.tick_params(labelsize=6.5, length=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 14)
        ax.set_yticks([0, 5, 10])
        ax.grid(axis="y", color="0.92", lw=0.5)
        ax.set_axisbelow(True)
    # TB2 and AutoBench end close together at 120: nudge the right-hand labels apart
    texts = sorted(axes[-1].texts, key=lambda t: t.get_position()[1])
    for lo, hi in zip(texts, texts[1:]):
        if hi.get_position()[1] - lo.get_position()[1] < 1.4:
            hi.set_y(lo.get_position()[1] + 1.4)
    axes[0].set_ylabel("gain reported for\nidentical candidates (pts)", fontsize=7)
    fig.text(0.48, 0.02, "dev tasks (log scale; line: 30, the recorded runs)", ha="center", va="bottom", fontsize=7)
    save_figure(fig, "devsize")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m aux.selection_replay.devsize compute|draw")
    if sys.argv[1] == "compute":
        compute()
    else:
        draw_devsize(json.loads((dataloader.CACHE / "selection_devsize.json").read_text()))
