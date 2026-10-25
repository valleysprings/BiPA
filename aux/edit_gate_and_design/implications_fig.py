"""The design-principles figure (mech_implications4), drawn from the caches of design selrules, designev and batching.

(a) edits kept under four acceptance rules: an unchanged candidate and one that solves `plot_gain` more dev tasks (mean and range over the benchmarks);
(b) share of an edit's (and a learned skill's) token change explained by skill length, turns and tool calls;
(c) log rho as recorded and recomputed at the seed's calls per turn;
(d) change in log input tokens against the parent, rejected and kept edits.
The drawn numbers go to OUT/data/mech/mech_implications4_numbers.json.
Run:  python -m aux.edit_gate_and_design.implications_fig
"""
import json

import numpy as np

from aux.figures.layoutcheck import check
from src.utils.dataloader import CACHE
from src.utils.fig import save_figure
from src.utils.logger.paperout import DATA

BENCH = [("tau3-bench", r"$\tau^3$-bench", "τ³-bench"), ("automationbench", "AutoBench", "AutoBench"), ("terminal-bench-2-1", "TB2", "TB2")]
SEL_BENCHES = ["terminal-bench-2-1", "tau3-bench", "automationbench"]
RULES = ["better", "critical", "confirm", "mean3"]
FS = 7
BLUE, GREY, INK = "#0072B2", "#A9A9A9", "#222222"
REJ, KEPT = "#A9A9A9", "#4A4A4A"
TITLES = ["(a) Two wins keep fewer unchanged candidates, most gains",
          "(b) Turns, not skill length, explain token change",
          "(c) At the seed's calls per turn, AutoBench's log ρ halves",
          "(d) Kept and rejected edits add tokens alike"]
TITLE_DX = [0.065, 0.13, 0.065, 0.065]
TITLE_DY = 0.07


def load(name):
    path = CACHE / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"{path} missing: run python -m aux.edit_gate_and_design.design {name}")
    return json.loads(path.read_text())


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=FS, length=2)


def legend_above(ax, handles):
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(handles), fontsize=FS, frameon=False,
              handlelength=0.9, handleheight=0.8, handletextpad=0.35, columnspacing=1.0, borderaxespad=0.15, borderpad=0.0)


def panel_a(ax, data, nums, Patch):
    gain = str(data["plot_gain"])
    labels = ["(1) one win\n(current)", "(2) one win\nby the critical\nmargin", "(3) two wins\nin a row", "(4) 3-run\ntotal"]
    width = 0.38
    for i, key in enumerate(RULES):
        null = [100 * data[b]["null"][key] for b in SEL_BENCHES]
        power = [100 * data[b]["power"][gain][key] for b in SEL_BENCHES]
        for offset, values, color, tag in ((-width / 2, null, GREY, "copy"), (width / 2, power, BLUE, "gain")):
            mean = float(np.mean(values))
            nums[f"a_{key}_{tag}"] = dict(mean=mean, lo=min(values), hi=max(values))
            ax.bar(i + offset, mean, width=width, color=color, edgecolor="none")
            ax.plot([i + offset] * 2, [min(values), max(values)], color="black", lw=0.9)
    ax.set_xticks(range(len(RULES)))
    ax.set_xticklabels(labels, fontsize=FS, linespacing=0.95)
    ax.set_xlim(-0.6, len(RULES) - 0.4)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("edits kept (%)", fontsize=FS, labelpad=1)
    clean(ax)
    ax.tick_params(axis="x", length=0, pad=2)
    legend_above(ax, [Patch(facecolor=GREY, label="unchanged candidate"), Patch(facecolor=BLUE, label=f"solves {gain} more dev tasks")])


def panel_b(ax, data, nums):
    edge, test = data["A"]["edge_level"], data["A"]["test_level"]
    labels = ["skill length", "turns", "tool calls", "skill length", "turns"]
    values = [edge["skill length"]["r2"], edge["turns"]["r2"], edge["actions"]["r2"], test["skill length"]["r2"], test["turns"]["r2"]]
    colors = [GREY, BLUE, GREY, GREY, BLUE]
    ypos = np.array([5.6, 4.6, 3.6, 1.4, 0.4])
    ax.barh(ypos, values, color=colors, height=0.68)
    ax.text(-0.02, 6.55, f"{edge['turns']['n']} edits", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=FS, fontweight="bold")
    ax.text(-0.02, 2.35, f"{test['turns']['n']} learned skills", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=FS,
            fontweight="bold")
    for y, v, lab, col in zip(ypos, values, ["e_len", "e_turns", "e_calls", "s_len", "s_turns"], colors):
        ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=FS, color=(BLUE if col == BLUE else "black"))
        nums[f"b_{lab}"] = round(v, 2)
    ax.set_ylim(-0.3, 6.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=FS)
    for t, col in zip(ax.get_yticklabels(), colors):
        t.set_color(BLUE if col == BLUE else "black")
    ax.set_xlim(0, 1.09)
    ax.spines["bottom"].set_bounds(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("token change explained (R²)", fontsize=FS, labelpad=1)
    clean(ax)
    ax.tick_params(axis="y", length=0)


def panel_c(ax, data, nums, Patch):
    groups = [(key, [c for c in data["cells"] if c["bench"] == b]) for b, key, _ in BENCH] + [("all", data["cells"])]
    x = np.arange(len(groups))
    width = 0.38
    measured = [float(np.mean([c["log_rho"] for c in cells])) for _, cells in groups]
    recomputed = [float(np.mean([c["log_rho_cf"] for c in cells])) for _, cells in groups]
    for (name, _), m, r in zip(groups, measured, recomputed):
        nums[f"c_{name}"] = dict(recorded=round(m, 3), recomputed=round(r, 3))
    ax.bar(x - width / 2, measured, width, color=GREY)
    ax.bar(x + width / 2, recomputed, width, color=BLUE)
    ax.axhline(0, color="0.3", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label in BENCH] + ["all"], fontsize=FS)
    ax.set_ylabel(r"$\log\rho$ (input tokens vs. seed)", fontsize=FS, labelpad=1)
    ax.set_ylim(0, 0.22)
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    clean(ax)
    ax.tick_params(axis="x", length=0, pad=2)
    legend_above(ax, [Patch(facecolor=GREY, label="as recorded"), Patch(facecolor=BLUE, label="recomputed at the seed's calls per turn")])


def panel_d(ax, data, nums):
    b = data["B"]
    rng = np.random.default_rng(0)
    for i, (name, g, color) in enumerate((("rejected", b["rejected"], REJ), ("kept", b["accepted"], KEPT))):
        values = 100 * np.array(g["values"])
        jitter = rng.uniform(-0.17, 0.17, len(values))
        ax.scatter(i + jitter, values, s=9, color=color, alpha=0.85, linewidths=0, zorder=2)
        ax.hlines(100 * g["mean"], i - 0.32, i + 0.32, color="black", lw=1.3, zorder=3)
        ax.text(i, 104, f"{100 * g['share_above']:.0f}% add tokens", ha="center", va="bottom", fontsize=FS, color=INK)
        nums[f"d_{name}"] = dict(n=g["n"], mean=round(g["mean"], 3), share_above=round(100 * g["share_above"], 1))
    ax.axhline(0, color="0.4", lw=0.6, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"rejected ({b['rejected']['n']} edits)", f"kept ({b['accepted']['n']} edits)"], fontsize=FS)
    for t in ax.get_xticklabels():
        t.set_color(INK)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-60, 118)
    ax.set_yticks([-50, 0, 50, 100])
    ax.spines["left"].set_bounds(-60, 100)
    ax.set_ylabel("change in log input tokens\nvs. parent (×100)", fontsize=FS, labelpad=2.5, linespacing=0.95)
    clean(ax)
    ax.tick_params(axis="x", length=0, pad=2)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                         "font.size": FS, "axes.linewidth": 0.6})
    selrules, designev, batching = load("selrules"), load("designev"), load("batching")
    figure, axes = plt.subplots(2, 2, figsize=(6.3, 3.4))
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.9, wspace=0.42, hspace=0.8)
    nums = {}
    panel_a(axes[0, 0], selrules, nums, Patch)
    panel_b(axes[0, 1], designev, nums)
    panel_c(axes[1, 0], batching, nums, Patch)
    panel_d(axes[1, 1], designev, nums)
    for ax, title, dx in zip(axes.flat, TITLES, TITLE_DX):
        box = ax.get_position()
        figure.text(box.x0 - dx, box.y1 + TITLE_DY, title, fontsize=FS + 0.5, fontweight="bold", ha="left", va="bottom")
    assert not check(figure), "layout problems above"
    save_figure(figure, "implications4")
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "mech_implications4_numbers.json").write_text(json.dumps(nums, indent=1))
    print(json.dumps(nums, indent=1))


if __name__ == "__main__":
    main()
