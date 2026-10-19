"""mech_transfer: appendix figure for RQ4 generality (app:ood / app:transfer).
(a) Cross-benchmark transfer (single-run scan, rep0): each DeepSeek skill trained on one execution benchmark, evaluated unchanged
on the other's test panel (hollow), against the skills optimized on the target benchmark itself (filled) and the target seed (dashed).
(b) Cross-model transfer (avg@3): the DeepSeek tau3-bench skills evaluated unchanged under GPT-6 Luna (hollow), against Luna's
no-skill agent, Luna's seed (dashed) and Luna's natively optimized skills (filled).
Numbers are quoted verbatim from the finalized records paper/assets/data/mech/mech_ood.md (single-run scan) and
paper/assets/data/mech/mech_transfer.md (avg@3); this script reads no run artifacts.
  .venv/bin/python -m src.analysis.mech.transfer_fig"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils import fig
METHODS = ["GEPA", "SkillOpt", "SkillAdam"]
OC = {"SkillOpt": "#0072B2", "SkillAdam": "#D55E00", "GEPA": "#009E73"}
MARK = {"SkillOpt": "o", "SkillAdam": "s", "GEPA": "^"}

# paper/assets/data/mech/mech_ood.md (rep0 single-run scan; seed rows are the Table 1 avg@3 test caches)
OOD = [  # (target bench, seed Q, local Q per method, transferred Q per method, transfer source)
    ("AutoBench", 45.0, {"GEPA": 44.2, "SkillOpt": 41.7, "SkillAdam": 44.7},
     {"GEPA": 40.0, "SkillOpt": 42.5, "SkillAdam": 41.7}, "TB2"),
    ("TB2", 75.9, {"GEPA": 74.1, "SkillOpt": 64.8, "SkillAdam": 68.5},
     {"GEPA": 77.8, "SkillOpt": 64.7, "SkillAdam": 72.2}, "AutoBench"),
]
# paper/assets/data/mech/mech_transfer.md (avg@3, GPT-6 Luna on tau3-bench test)
LUNA_NOSKILL = 64.9
LUNA_SEED = 69.3
LUNA_NATIVE = {"GEPA": 65.3, "SkillOpt": 63.1, "SkillAdam": 64.0}      # Luna optimizes from the same seed (Table 1 lower panel)
LUNA_TRANSFER = {"GEPA": 63.1, "SkillOpt": 64.0, "SkillAdam": 65.8}    # DeepSeek-trained, evaluated unchanged


def draw():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "STIXGeneral"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.4, 2.05), gridspec_kw=dict(width_ratios=[7, 4.4], wspace=0.12), sharey=True)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.15, top=0.70)

    # (a) cross-benchmark, single-run scan
    for c, (target, seed, local, trans, src) in enumerate(OOD):
        base = c * 4
        axl.plot([base - 0.55, base + 2.55], [seed] * 2, color="0.15", lw=1.0, ls=(0, (4, 2)), zorder=1)
        axl.text(base + 2.55, seed + 0.8, f"seed {seed:.1f}", fontsize=7, color="0.15", ha="right", va="bottom")
        for m, name in enumerate(METHODS):
            axl.scatter(base + m - 0.14, local[name], s=26, marker=MARK[name], color=OC[name], lw=0, zorder=3)
            axl.scatter(base + m + 0.14, trans[name], s=26, marker=MARK[name], facecolor="white", edgecolor=OC[name], lw=1.0, zorder=3)
        axl.text(base + 1, 1.045, f"target: {target}", transform=axl.get_xaxis_transform(), fontsize=7, fontweight="bold",
                 ha="center", va="bottom", clip_on=False)
    axl.set_xlim(-0.7, 6.7)
    axl.set_ylim(35, 82)
    axl.set_xticks([c * 4 + m for c in range(2) for m in range(3)])
    axl.set_xticklabels(METHODS * 2)
    axl.tick_params(axis="x", length=0, pad=2)
    axl.set_ylabel("test success (%)")
    axl.spines[["top", "right"]].set_visible(False)
    axl.set_title("(a) across benchmarks (single-run scan)", fontsize=7, loc="left", pad=20)

    # (b) cross-model, avg@3
    axr.axhline(LUNA_SEED, color="0.15", lw=1.0, ls=(0, (4, 2)), zorder=1)
    axr.text(-0.5, LUNA_SEED + 0.8, f"seed {LUNA_SEED:.1f}", fontsize=7, color="0.15", ha="left", va="bottom")
    axr.scatter(0, LUNA_NOSKILL, s=26, marker="o", facecolor="white", edgecolor="0.5", lw=1.0, zorder=3)
    for m, name in enumerate(METHODS):
        x = 1.2 + m
        axr.scatter(x - 0.13, LUNA_NATIVE[name], s=26, marker=MARK[name], color=OC[name], lw=0, zorder=3)
        axr.scatter(x + 0.13, LUNA_TRANSFER[name], s=26, marker=MARK[name], facecolor="white", edgecolor=OC[name], lw=1.0, zorder=3)
    axr.set_xlim(-0.55, 3.75)
    axr.set_xticks([0, 1.2, 2.2, 3.2])
    axr.set_xticklabels(["no skill"] + METHODS)
    axr.tick_params(axis="x", length=0, pad=2)
    axr.tick_params(axis="y", length=0)
    axr.spines[["top", "right", "left"]].set_visible(False)
    axr.text(1.6, 1.045, r"target: GPT-6 Luna, $\tau^3$-bench", transform=axr.get_xaxis_transform(), fontsize=7, fontweight="bold",
             ha="center", va="bottom", clip_on=False)
    axr.set_title(r"(b) across models (avg@3)", fontsize=7, loc="left", pad=20)

    handles = [plt.Line2D([], [], ls="", marker=MARK[n], color=OC[n], ms=4.5, label=n) for n in METHODS]
    handles += [plt.Line2D([], [], ls="", marker="o", mfc="0.4", mec="0.4", ms=4.5, label="optimized in-domain"),
                plt.Line2D([], [], ls="", marker="o", mfc="white", mec="0.4", ms=4.5, mew=1.0, label="transferred in, unchanged"),
                plt.Line2D([], [], ls="", marker="o", mfc="white", mec="0.5", ms=4.5, mew=1.0, label="no skill"),
                plt.Line2D([], [], color="0.15", lw=1.0, ls=(0, (4, 2)), label="seed")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.04), ncol=4, frameon=False, fontsize=7,
               handlelength=1.2, columnspacing=0.9, handletextpad=0.3, labelspacing=0.35)
    return fig


def main():
    fig = draw()
    fig.save_figure(fig, "transfer")
    for target, seed, local, trans, src in OOD:
        for name in METHODS:
            print(f"{src}->{target} {name}: local {local[name]:.1f} | transferred {trans[name]:.1f} | seed {seed:.1f} (single-run)")
    for name in METHODS:
        print(f"DeepSeek->Luna {name}: native {LUNA_NATIVE[name]:.1f} | transferred {LUNA_TRANSFER[name]:.1f} "
              f"| seed {LUNA_SEED:.1f} | no skill {LUNA_NOSKILL:.1f} (avg@3)")


if __name__ == "__main__":
    main()
