"""Figure bip_writes: the call edges of the graph by turn, and the serialized writes of the learned AutoBench skills.

(a) One AutoBench task (the teaser pair: finance-vendor-payment-approval, DeepSeek-V4.1-Flash, run 0): every turn is a column, every tool call that turn issues is a square,
    red = write (changes the simulated world), blue = read, gray = other.  The seed groups its writes (three in one turn), SkillOpt's skill issues them one per turn.
(b) All AutoBench test episodes, both models: the share of writes that share a turn with another write, and the share of reads that share a turn with another read,
    for the seed and for each other agent on the same tasks (dumbbell: hollow = seed, filled = the row's agent; 95% paired-bootstrap interval).

Data: OUT/calls_and_writes/calls_rw_data.pkl (per episode: the class of each executed call in each turn) and calls_rw.json (shares and intervals, the 'main' variant of mech_calls_rw.md).
Run:  python -m aux.calls_and_writes.fig_writes
"""
import json
import pickle

import numpy as np

from aux.token_attribution.bipcommon import *  # noqa: F401,F403
from src import layout

CR = layout.OUT / "calls_and_writes"
DATA = None
RW = None
TASK = "finance-vendor-payment-approval"
COL = dict(W="#C0392B", R="#0072B2", O="#BDBDBD")
ROWS = ["No skill", "GEPA", "SkillOpt", "SkillAdam"]
ROW_LABEL = {"No skill": "no skill", "GEPA": "GEPA", "SkillOpt": "SkillOpt", "SkillAdam": "SkillAdam"}


def kind(c):
    return c if c in ("W", "R") else "O"


def case(row):
    e = DATA[("deepseek-flash", row)][0][TASK]
    turns = [[kind(c) for c in t] for t in e["turns"]]
    return dict(turns=turns, R=e["R"], N=e["N"], X=e["X"], success=e["success"])


def cell(model, row):
    return next(c for c in RW if c["model"] == model and c["row"] == row and c["variant"] == "main")


def draw_strip(ax, ep, y0, ymax):
    """One episode: a column per turn, one square per executed call, writes red, reads blue, other gray; returns (turns with >=2 writes, turns with exactly 1 write)."""
    multi, single = [], []
    for t, calls in enumerate(ep["turns"], start=1):
        order = sorted(calls, key=lambda k: {"R": 0, "O": 1, "W": 2}[k])
        for i, k in enumerate(order):
            ax.add_patch(plt_rect(t - 0.42, y0 + i * 0.9 + 0.05, 0.84, 0.8, COL[k]))
        nw = sum(1 for k in calls if k == "W")
        if nw >= 2:
            multi.append(t)
        elif nw == 1:
            single.append(t)
        if not calls:
            ax.plot([t], [y0 + 0.45], marker="_", ms=4, color=MUTED, mew=0.8)
    return multi, single


def plt_rect(x, y, w, h, c):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, fc=c, ec="white", lw=0.4, zorder=3)


def load():
    """Read the inputs this figure draws from."""
    global DATA, RW
    DATA = pickle.loads((CR / "calls_rw_data.pkl").read_bytes())["data"]
    RW = json.loads((CR / "calls_rw.json").read_text())["cells"]


def main():
    load()
    plt = setup_mpl()
    from matplotlib.patches import Patch
    seed, so = case("Seed skill"), case("SkillOpt")
    fig = plt.figure(figsize=(7.1, 2.85))
    # ---------------- (a)
    axa = fig.add_axes([0.065, 0.19, 0.405, 0.62])
    nmax = max(len(seed["turns"]), len(so["turns"]))
    hmax = max(len(t) for ep in (seed, so) for t in ep["turns"])
    gap = 1.7
    y_so = 0.0
    y_seed = hmax * 0.9 + 0.5 + gap
    m1, s1 = draw_strip(axa, seed, y_seed, hmax)
    m2, s2 = draw_strip(axa, so, y_so, hmax)
    axa.set_xlim(0.2, nmax + 0.8)
    axa.set_ylim(-0.3, y_seed + hmax * 0.9 + 1.7)
    for sp in ("top", "right", "left"):
        axa.spines[sp].set_visible(False)
    axa.set_yticks([])
    axa.set_xticks([1, 5, 10, 15, 20])
    axa.set_xlabel("turn", fontsize=7.5, labelpad=1)
    axa.tick_params(axis="x", labelsize=7)
    for ep, y, name in ((seed, y_seed, "seed skill"), (so, y_so, "SkillOpt skill")):
        nw = sum(1 for t in ep["turns"] for k in t if k == "W")
        nwt = sum(1 for t in ep["turns"] if "W" in t)
        axa.text(0.2, y + hmax * 0.9 + 0.62, name, fontsize=7.5, fontweight="bold", ha="left", va="bottom", color=INK)
        axa.text(nmax + 0.8, y + hmax * 0.9 + 0.62, f"{ep['R']} turns, {ep['N']} calls, {nw} writes in {nwt} turns", fontsize=7, ha="right", va="bottom", color=INK)
    fig.text(0.065, 0.925, "(a) One AutoBench task, the calls of every turn", fontsize=8, fontweight="bold", ha="left", va="center", color=INK)
    # legend under the title of (a), right side of the axis label row
    from matplotlib.lines import Line2D as L2
    handles = [Patch(fc=COL["W"], label="write"), Patch(fc=COL["R"], label="read"), Patch(fc=COL["O"], label="other call"),
               L2([0], [0], marker="_", ls="", color=MUTED, ms=5, mew=0.8, label="turn without a call")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.055, -0.005), ncol=4, frameon=False, fontsize=6.8, handlelength=0.9, handleheight=0.8, columnspacing=0.9, borderaxespad=0.0, handletextpad=0.4)

    # ---------------- (b): dumbbells, writes and reads
    labels_y = {}
    y = 0.0
    slots = []
    for model in ("deepseek-flash", "gpt-6-luna"):
        slots.append(("hdr", model, y))
        y += 0.85
        for r in ROWS:
            slots.append(("row", (model, r), y))
            y += 1.0
        y += 0.2
    axw = fig.add_axes([0.665, 0.22, 0.14, 0.59])
    axr = fig.add_axes([0.835, 0.22, 0.14, 0.59], sharey=axw)
    for ax, what, title in ((axw, "Wpar", "(b) Writes sharing a turn\nwith another write"), (axr, "Rpar", "Reads sharing a turn\nwith another read")):
        for kindslot, key, yy in slots:
            if kindslot == "hdr":
                continue
            model, r = key
            c = cell(model, r)
            s0, s1_ = c["point"][f"seed.{what}"] * 100, c["point"][f"skill.{what}"] * 100
            l0, h0 = [v * 100 for v in c["ci"][f"seed.{what}"]]
            l1, h1 = [v * 100 for v in c["ci"][f"skill.{what}"]]
            ax.plot([s0, s1_], [yy, yy], color="#9A9A9A", lw=0.9, zorder=2)
            ax.plot([l0, h0], [yy, yy], color="#6E6E6E", lw=2.2, alpha=0.35, zorder=3, solid_capstyle="butt")
            ax.plot([l1, h1], [yy, yy], color=COL["W"] if what == "Wpar" else COL["R"], lw=2.2, alpha=0.35, zorder=3, solid_capstyle="butt")
            ax.plot([s0], [yy], marker="o", ms=4.4, mfc="white", mec=INK, mew=0.8, zorder=4)
            ax.plot([s1_], [yy], marker="o", ms=4.4, mfc=COL["W"] if what == "Wpar" else COL["R"], mec="white", mew=0.5, zorder=5)
        ax.set_xlim(-3, 103)
        ax.set_ylim(y - 0.4, -1.25)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=7)
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.text(0.0, 1.01, title, transform=ax.transAxes, fontsize=7.6, fontweight="bold", ha="left", va="bottom", color=INK, linespacing=1.0)
    axw.set_xlabel("% of the agent's writes", fontsize=7)
    axr.set_xlabel("% of the agent's reads", fontsize=7)
    axw.set_yticks([yy for k, key, yy in slots if k == "row"])
    labs = []
    for k, key, yy in slots:
        if k == "row":
            c = cell(*key)
            labs.append(ROW_LABEL[key[1]] + (f" ({c['n']})" if c["n"] != 120 else ""))
    axw.set_yticklabels(labs, fontsize=7)
    plt.setp(axr.get_yticklabels(), visible=False)
    axr.tick_params(axis="y", length=0)
    for k, key, yy in slots:
        if k == "hdr":
            axw.text(-0.02, yy, MODEL_NAME[key], transform=axw.get_yaxis_transform(), ha="right", va="center", fontsize=7.3, fontweight="bold", color=INK, clip_on=False)
    # key to the dots
    from matplotlib.lines import Line2D
    kh = [Line2D([0], [0], marker="o", ls="", mfc="white", mec=INK, ms=4.4, label="seed, same tasks"),
          Line2D([0], [0], marker="o", ls="", mfc="#7F7F7F", mec="white", ms=4.4, label="the row's agent")]
    fig.legend(handles=kh, loc="lower center", bbox_to_anchor=(0.83, 0.03), ncol=2, frameon=False, fontsize=6.8, handletextpad=0.2, columnspacing=1.2)
    fig.text(0.83, 0.012, "number of tasks in brackets where fewer than 120 had usable runs", fontsize=5.8, color=MUTED, ha="center", va="center")
    save(fig, "bip_writes")
    plt.close(fig)
    # numbers for the record
    out = dict(case=dict(seed=dict(turns=["".join(t) or "-" for t in seed["turns"]], R=seed["R"], N=seed["N"], X=seed["X"]),
                         skillopt=dict(turns=["".join(t) or "-" for t in so["turns"]], R=so["R"], N=so["N"], X=so["X"])),
               agg={f"{m}|{r}": dict(n=cell(m, r)["n"], **{k: cell(m, r)["point"][k] for k in ("seed.Wpar", "skill.Wpar", "seed.Rpar", "skill.Rpar", "seed.m", "skill.m")},
                                     ci_Wpar_seed=cell(m, r)["ci"]["seed.Wpar"], ci_Wpar_skill=cell(m, r)["ci"]["skill.Wpar"],
                                     ci_Rpar_seed=cell(m, r)["ci"]["seed.Rpar"], ci_Rpar_skill=cell(m, r)["ci"]["skill.Rpar"])
                    for m in ("deepseek-flash", "gpt-6-luna") for r in ROWS})
    (OUT / "bip_writes_numbers.json").write_text(json.dumps(out, indent=1))
    for k, v in out["agg"].items():
        print(k, v["n"], "W seed %.3f skill %.3f | R seed %.3f skill %.3f" % (v["seed.Wpar"], v["skill.Wpar"], v["seed.Rpar"], v["skill.Rpar"]))
    print(out["case"])


if __name__ == "__main__":
    main()
