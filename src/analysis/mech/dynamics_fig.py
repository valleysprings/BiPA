"""mech_dynamics: appendix figure for app:dynamics (RQ4), H2 "skills just squeeze more thinking".

  (a) the seed's own test-time curve: success of the seed's episodes in six quantile bins of turns R, for both models on tau3-bench
      and AutoBench (TB2's 18 tasks are too few for this statistic). Success falls as turns grow (DeepSeek tau3: 87.8% in the lowest
      bin to 27.0% in the highest; Luna tau3: 87.0% to 16.2%). The association is across tasks (harder tasks take more turns):
      it describes where the added turns sit, not a causal dose.
  (b) white-squeeze rate of each learned skill: of the tasks the skill pushed to more turns than the seed (dR > 0), the share whose
      success did not rise (dQ <= 0) — 77 to 100% across all twelve optimizer cells (both models x both benchmarks x GEPA/SkillOpt/
      SkillAdam), with the task counts on the bars.

Data: assets/cache/analysis/mech/dynamics.json (written by analysis.mech.dynamics; numbers as recorded in mech_dynamics.md, C1 and C2).
  python3 -m src.analysis.mech.dynamics_fig  -> fig/mech/mech_dynamics.{pdf,svg}
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.mech import common as MC

MODELS = [("deepseek-flash", "DeepSeek"), ("gpt-6-luna", "Luna")]
BENCH = [("tau3-bench", r"$\tau^3$"), ("automationbench", "AutoB")]
ARMS = [("gepa", "GEPA"), ("skillopt", "SkillOpt"), ("skilladam", "SkillAdam")]
# bench color in (a); optimizer color in (b), as in mech_paired
BC = {"tau3-bench": "#0072B2", "automationbench": "#E69F00"}
OC = {"SkillOpt": "#0072B2", "SkillAdam": "#D55E00", "GEPA": "#009E73"}
LS = {"deepseek-flash": "-", "gpt-6-luna": "--"}
MK = {"deepseek-flash": "o", "gpt-6-luna": "s"}


def load():
    return json.loads((MC.CACHE / "dynamics.json").read_text())["cells"]


def draw_curve(ax, cells):
    """(a) Seed success per turns bin; color = benchmark, line style / marker = model; the legend carries the turn ranges."""
    for model, mshort in MODELS:
        for bench, bshort in BENCH:
            bins = cells[f"{model}|{bench}"]["curve"]["R"]
            xs = list(range(len(bins)))
            ys = [b["success"] for b in bins]
            label = f"{mshort} {bshort}, {bins[0]['lo']:.0f}\u2013{bins[-1]['hi']:.0f} turns"
            ax.plot(xs, ys, LS[model], marker=MK[model], ms=3, lw=1.0, color=BC[bench],
                    markerfacecolor="white" if model == "gpt-6-luna" else BC[bench], markeredgewidth=0.8, label=label)
    ax.set_xticks(range(6))
    ax.set_xticklabels([str(i + 1) for i in range(6)])
    ax.set_xlabel("turns bin of the seed's episodes (low $\u2192$ high)", fontsize=7, labelpad=1)
    ax.set_ylabel("seed success (%)", fontsize=7, labelpad=1)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=6.5, length=2, pad=1)
    ax.legend(frameon=False, fontsize=6.3, loc="lower left", handlelength=1.8, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_squeeze(ax, cells):
    """(b) White-squeeze bars: groups = model x benchmark, bars = optimizer, task counts on top."""
    width = 0.26
    for g, (model, mshort) in enumerate(MODELS):
        for b, (bench, bshort) in enumerate(BENCH):
            gx = 2 * g + b
            cell = cells[f"{model}|{bench}"]
            for j, (arm, label) in enumerate(ARMS):
                h = cell["arms"][arm]["h2"]
                v = 100 * h["wasted_of_up"]
                x = gx + (j - 1) * width
                ax.bar(x, v, width=width * 0.9, color=OC[label], edgecolor="white", linewidth=0.4)
                ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.3)
                ax.text(x, v / 2, f"{h['n_wasted']}/{h['n_up']}", ha="center", va="center", fontsize=5.6, color="white", rotation=90)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{mshort}\n{bshort}" for mshort in ("DeepSeek", "Luna") for bshort in (r"$\tau^3$", "AutoB")], fontsize=6.5)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("white-squeeze (%)", fontsize=7, labelpad=1)
    ax.tick_params(axis="y", labelsize=6.5, length=2, pad=1)
    ax.tick_params(axis="x", length=0, pad=2)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=OC[label], label=label) for _, label in ARMS]
    ax.legend(handles=handles, frameon=False, fontsize=6.3, loc="lower left", ncol=3, handlelength=0.9, handletextpad=0.3,
              columnspacing=0.6, borderaxespad=0.1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.linewidth": 0.6})
    cells = load()
    fig = plt.figure(figsize=(6.75, 2.1))
    ax_a = fig.add_axes([0.065, 0.15, 0.42, 0.70])
    ax_b = fig.add_axes([0.585, 0.15, 0.40, 0.70])
    draw_curve(ax_a, cells)
    draw_squeeze(ax_b, cells)
    fig.text(0.065, 0.985, "(a) The seed's own test-time curve", fontsize=7.5, fontweight="bold", va="top")
    fig.text(0.585, 0.985, "(b) Extra turns buy no success", fontsize=7.5, fontweight="bold", va="top")
    MC.save_figure(fig, "dynamics")


if __name__ == "__main__":
    main()
