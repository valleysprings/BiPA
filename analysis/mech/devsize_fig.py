"""Paper 1 (mechanism paper), Figure 'Larger dev sets help slowly': the dev-size replay of analysis.mech.selection, redrawn with plain labels.

Role in the argument: Finding 1 says a larger dev set is no quick fix for selection noise.  The numbers are the dev-size replay that
selection.py computes (devsize(); 40,000 simulated dev draws per size on the seed's fitted task model, no new episodes) and stores in
assets/cache/analysis/mech/selection.json under <bench>["devsize"][<prior>][<size>]:
  (a) E_gain_random_points  the dev gain (points) that SkillOpt's rule (keep the best of 4 candidates if it beats the baseline) and GEPA's
                            rule (the best of 7 children against the seed) announce when every candidate is identical to the seed
  (b) MDE80_points          the smallest true gain (points) that a comparison of two single dev runs detects with 80% power
selection.py is maintained by another session; this module only reads its cache and draws, so the original figure code stays untouched.

Command (from bipa):  python3 -m analysis.mech.devsize_fig  -> fig/mech/mech_devsize.{pdf,svg}
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analysis.mech import common as MC

SIZES = (9, 15, 30, 60, 120)
BENCHES = (("terminal-bench-2-1", "TB2", "s"), ("tau3-bench", r"$\tau^3$-bench", "o"), ("automationbench", "AutoBench", "^"))
RULES = (("SkillOpt", "SkillOpt's rule (best of 4)", "0.15"), ("GEPA", "GEPA's rule (best of 7)", "0.6"))
PRIORS = (("fitted", "-"), ("moderate", "--"))


def main():
    """Two stacked axes against the number of dev tasks (log axis): (a) the gain announced for unchanged candidates, (b) the smallest detectable gain."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.6})
    result = json.loads((MC.CACHE / "selection.json").read_text())
    fig, axs = plt.subplots(2, 1, figsize=(3.25, 2.75), sharex=True)
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.13, top=0.985, hspace=0.12)
    for bench, _, marker in BENCHES:
        for prior, ls in PRIORS:
            rows = [result[bench]["devsize"][prior][str(s)] for s in SIZES]
            for rule, _, shade in RULES:
                axs[0].plot(SIZES, [r[rule]["E_gain_random_points"] for r in rows], color=shade, ls=ls, lw=1.0, marker=marker, markersize=3.5)
            # a missing point: no gain up to the dev-set size is detected with probability 0.8
            axs[1].plot(SIZES, [r["MDE80_points"] for r in rows], color="0.15", ls=ls, lw=1.0, marker=marker, markersize=3.5)
    for ax in axs:
        ax.axvline(30, color="0.75", lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(SIZES)
        ax.set_xticklabels([str(s) for s in SIZES])
        ax.minorticks_off()
        ax.tick_params(labelsize=7, length=2)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("(a) gain announced for\nunchanged candidates (pts)", fontsize=7)
    axs[1].set_ylabel("(b) smallest gain one dev\ncomparison detects (pts)", fontsize=7)
    axs[1].set_xlabel("dev tasks (30 in our runs)", fontsize=7)
    rule_handles = [Line2D([], [], color=shade, lw=1.2, label=label) for _, label, shade in RULES]
    prior_handles = [Line2D([], [], color="0.3", lw=1.0, ls="-", label="fitted task model"),
                     Line2D([], [], color="0.3", lw=1.0, ls="--", label="flatter task model")]
    axs[0].legend(handles=rule_handles + prior_handles, loc="upper right", fontsize=6.5, frameon=False, handlelength=1.8, borderaxespad=0.1)
    bench_handles = [Line2D([], [], color="0.3", lw=0, marker=m, markersize=4, label=d) for _, d, m in BENCHES]
    axs[1].legend(handles=bench_handles, loc="upper right", fontsize=6.5, frameon=False, ncol=3, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.1)
    MC.save_figure(fig, "devsize")


if __name__ == "__main__":
    main()
