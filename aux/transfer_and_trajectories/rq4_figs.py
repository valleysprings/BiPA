"""Two appendix figures from explore.json and transfer_dq.json (OUT/transfer_and_trajectories):
  mech_transfer2: success difference to the seed (a) and input-token ratio (b), in-domain vs off-distribution
  mech_explore:   (a) tasks whose three runs follow more than one behavior; (b) share of converging tasks among gains /
                  unchanged / losses, for the seed over no skill and the learned skills over the seed
Run:  python -m aux.transfer_and_trajectories.rq4_figs
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import layout
from src.utils import fig

HERE = layout.OUT / "transfer_and_trajectories"
OC = {"GEPA": "#009E73", "SkillOpt": "#0072B2", "SkillAdam": "#D55E00"}
MK = {"GEPA": "^", "SkillOpt": "o", "SkillAdam": "s"}
NOSKILL, SEED, LEARNED = "#BBBBBB", "#555555", "#882255"
plt.rcParams.update({"font.family": "serif", "font.size": 7})


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=6.5, length=2, pad=1)


# ---------- transfer ----------
RHO = {("in-domain", "DeepSeek", "TB2"): {"GEPA": 1.34, "SkillOpt": 1.43, "SkillAdam": 1.37},
       ("in-domain", "DeepSeek", "tau3"): {"GEPA": 1.15, "SkillOpt": 1.22, "SkillAdam": 1.09},
       ("in-domain", "DeepSeek", "AutoB"): {"GEPA": 1.21, "SkillOpt": 1.25, "SkillAdam": 1.08},
       ("cross-benchmark", "DeepSeek", "TB2->AutoB"): {"GEPA": 1.32, "SkillOpt": 1.49, "SkillAdam": 1.24},
       ("cross-benchmark", "DeepSeek", "AutoB->TB2"): {"GEPA": 1.22, "SkillOpt": 1.01, "SkillAdam": 1.00},
       ("in-domain", "Luna", "tau3"): {"GEPA": 1.04, "SkillOpt": 1.01, "SkillAdam": 0.98},
       ("cross-model", "Luna", "tau3"): {"GEPA": 1.17, "SkillOpt": 1.38, "SkillAdam": 1.08}}
SLOTS = [("in-domain", "DeepSeek", "TB2", "TB2"), ("in-domain", "DeepSeek", "tau3", r"$\tau^3$"),
         ("in-domain", "DeepSeek", "AutoB", "AutoB"),
         ("cross-benchmark", "DeepSeek", "TB2->AutoB", "TB2$\\to$\nAutoB"), ("cross-benchmark", "DeepSeek", "AutoB->TB2", "AutoB$\\to$\nTB2"),
         ("in-domain", "Luna", "tau3", "Luna's\nown"), ("cross-model", "Luna", "tau3", "from\nDeepSeek")]
XPOS = [0, 1, 2, 3.5, 4.6, 6.1, 7.3]


def draw_transfer():
    rows = json.loads((HERE / "transfer_dq.json").read_text())
    by = {(r["group"], r["model"], r["bench"], r["method"]): r for r in rows}
    figure, (axq, axr) = plt.subplots(1, 2, figsize=(6.4, 2.15), gridspec_kw=dict(wspace=0.22))
    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.25, top=0.83)
    for ax in (axq, axr):
        for x0, x1 in ((-0.5, 2.5), (5.55, 6.65)):  # in-domain columns shaded
            ax.axvspan(x0, x1, color="#F2F2F2", zorder=0, lw=0)
    for x, (group, model, bench, _) in zip(XPOS, SLOTS):
        hollow = group != "in-domain"
        for j, m in enumerate(OC):
            r = by[(group, model, bench, m)]
            xx = x + (j - 1) * 0.22
            kw = dict(marker=MK[m], ms=4.2, color=OC[m], markerfacecolor="white" if hollow else OC[m], markeredgewidth=0.9, lw=0)
            axq.plot([xx, xx], [r["lo"], r["hi"]], color=OC[m], lw=0.9, zorder=2)
            axq.plot(xx, r["dQ"], zorder=3, **kw)
            axr.plot(xx, RHO[(group, model, bench)][m], zorder=3, **kw)
    axq.axhline(0, color="#333333", lw=0.7, ls="--", zorder=1)
    axr.axhline(1, color="#333333", lw=0.7, ls="--", zorder=1)
    axq.set_ylabel("success $-$ seed (points)", fontsize=7, labelpad=1)
    axr.set_ylabel("input tokens / seed", fontsize=7, labelpad=1)
    axq.set_ylim(-27, 11)
    axr.set_ylim(0.9, 1.55)
    for ax, title in ((axq, "(a) success against the seed"), (axr, "(b) token ratio against the seed")):
        style(ax)
        ax.set_xticks(XPOS)
        ax.set_xticklabels([s[3] for s in SLOTS], fontsize=6.2)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_xlim(-0.6, 7.9)
        ax.set_title(title, fontsize=7, loc="left", pad=11)
        for xc, label in ((1, "DeepSeek, own setting"), (4.05, "DeepSeek, transferred"), (6.7, r"Luna, $\tau^3$")):
            ax.text(xc, 1.005, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.2, color="#444444")
    handles = [plt.Line2D([], [], marker=MK[m], color=OC[m], lw=0, ms=4.2, label=m) for m in OC]
    handles += [plt.Line2D([], [], marker="o", color="#444444", lw=0, ms=4.2, label="own setting (shaded)"),
                plt.Line2D([], [], marker="o", color="#444444", markerfacecolor="white", lw=0, ms=4.2, label="transferred")]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, -0.01), ncol=5, frameon=False, fontsize=6.4,
                  handletextpad=0.3, columnspacing=1.0)
    fig.save_figure(figure, "transfer2")


# ---------- exploration ----------
CELLS = [("deepseek-flash|tau3-bench", "DeepSeek\n" r"$\tau^3$"), ("deepseek-flash|automationbench", "DeepSeek\nAutoB"),
         ("gpt-6-luna|tau3-bench", "Luna\n" r"$\tau^3$"), ("gpt-6-luna|automationbench", "Luna\nAutoB")]
ARMS = [("native-noseed", "No skill", NOSKILL), ("native-seed", "Seed", SEED),
        ("gepa", "GEPA", OC["GEPA"]), ("skillopt", "SkillOpt", OC["SkillOpt"]), ("skilladam", "SkillAdam", OC["SkillAdam"])]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def pooled(cells, keys):
    out = {"gain": [0, 0], "tie": [0, 0], "loss": [0, 0]}  # [converged, total]
    for cell in cells.values():
        for k in keys:
            c = cell["contrasts"][k]
            ties = c["n"] - c["gain"] - c["loss"]
            out["gain"][0] += c["gain_less"]; out["gain"][1] += c["gain"]
            out["loss"][0] += c["loss_less"]; out["loss"][1] += c["loss"]
            out["tie"][0] += c["tie_less"]; out["tie"][1] += ties
    return out


def draw_explore():
    cells = json.loads((HERE / "explore.json").read_text())["cells"]
    figure, (axa, axb) = plt.subplots(1, 2, figsize=(6.4, 2.1), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.25))
    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.27, top=0.86)
    w = 0.16
    for g, (key, _) in enumerate(CELLS):
        for j, (arm, label, color) in enumerate(ARMS):
            v = 100 * (1 - cells[key]["arms"][arm]["A_share"][0])
            axa.bar(g + (j - 2) * w, v, width=w * 0.88, color=color, edgecolor="white", linewidth=0.4)
    axa.set_xticks(range(4))
    axa.set_xticklabels([c[1] for c in CELLS], fontsize=6.4)
    axa.tick_params(axis="x", length=0, pad=2)
    axa.set_ylabel("tasks with $>$1 behavior (%)", fontsize=7, labelpad=1)
    axa.set_ylim(0, 22)
    axa.set_title("(a) behavior attractors among a task's three runs", fontsize=7, loc="left")
    style(axa)
    contrasts = [("Seed over no skill", ["seed_vs_noskill"], SEED),
                 ("learned skills over seed", ["gepa_vs_seed", "skillopt_vs_seed", "skilladam_vs_seed"], LEARNED)]
    groups = [("gain", "success\nrose"), ("tie", "success\nunchanged"), ("loss", "success\nfell")]
    for j, (label, keys, color) in enumerate(contrasts):
        p = pooled(cells, keys)
        for g, (k, _) in enumerate(groups):
            conv, n = p[k]
            v, lo, hi = wilson(conv, n)
            x = g + (j - 0.5) * 0.34
            axb.bar(x, v, width=0.3, color=color, edgecolor="white", linewidth=0.4, label=label if g == 0 else None)
            axb.plot([x, x], [lo, hi], color="#222222", lw=0.8)
            axb.text(x, 3, f"{conv}/{n}", ha="center", va="bottom", fontsize=5.6, color="white", rotation=90)
    axb.axhline(50, color="#333333", lw=0.6, ls=":")
    axb.set_xticks(range(3))
    axb.set_xticklabels([g[1] for g in groups], fontsize=6.4)
    axb.tick_params(axis="x", length=0, pad=2)
    axb.set_ylim(0, 100)
    axb.set_ylabel("runs converged (%)", fontsize=7, labelpad=1)
    axb.set_title("(b) runs converged, by change in success", fontsize=7, loc="left")
    style(axb)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, label=l) for _, l, c in ARMS]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor=LEARNED, label="learned over seed (pooled)")]
    handles[1].set_label("Seed (in b: seed over no skill)")
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, -0.01), ncol=6, frameon=False, fontsize=6.2,
                  handlelength=0.9, handletextpad=0.3, columnspacing=0.8)
    fig.save_figure(figure, "explore")


if __name__ == "__main__":
    draw_transfer()
    # mech_explore.pdf (panel a only) and mech_gains.pdf come from gains_fig.py
    print("ok")
