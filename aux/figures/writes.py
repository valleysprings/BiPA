"""The serialized-writes figure (AutoBench): how the serialize rules put writes one per turn while reads stay grouped.

(a) The teaser task, DeepSeek, run 0, for the seed and the three learned skills: one column per turn, one square per executed call
(write, read, other). (b) The share of writes (reads) that share a turn with another write (read), over all test episodes with complete
logs (the 'main' variant of calls_rw.json, a ratio of totals); filled circle DeepSeek, hollow diamond Luna, cross the episode of (a).
Reads calls_rw_data.pkl and calls_rw.json (python -m aux.calls_and_writes.calls_rw extract, then stats); writes
<cache>/analysis/main/bipwrites3.json and fig/mech/mech_bipwrites3.
"""
import json
import pickle

from src import layout
from src.utils import dataloader as C
from src.utils.fig import save_figure

DATA_DIR = layout.OUT / "calls_and_writes"
TASK = "finance-vendor-payment-approval"
INK, MUTED, GRID = "#222222", "#5A5A5A", "#E2E5EA"
COL = dict(W="#C0392B", R="#7FA7CC", O="#D3D3D3")
WCOL, RCOL = "#C0392B", "#3F7FB5"
AG = ["Seed skill", "GEPA", "SkillOpt", "SkillAdam"]
AG_LABEL = {"Seed skill": "Seed", "GEPA": "GEPA", "SkillOpt": "SkillOpt", "SkillAdam": "SkillAdam", "No skill": "No skill"}
AG_COL = {"Seed skill": INK, "GEPA": "#009E73", "SkillOpt": "#0072B2", "SkillAdam": "#D55E00", "No skill": MUTED}


def load():
    paths = DATA_DIR / "calls_rw_data.pkl", DATA_DIR / "calls_rw.json"
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing {missing}: run python -m aux.calls_and_writes.calls_rw extract, then stats")
    return pickle.loads(paths[0].read_bytes())["data"], json.loads(paths[1].read_text())["cells"]


def episode(data, row, model="deepseek-flash", rep=0):
    turns = [[c if c in ("W", "R") else "O" for c in t] for t in data[(model, row)][rep][TASK]["turns"]]
    return dict(turns=turns, nturn=len(turns), N=sum(len(t) for t in turns), W=sum(t.count("W") for t in turns),
                R=sum(t.count("R") for t in turns), Wp=sum(t.count("W") for t in turns if t.count("W") >= 2),
                Rp=sum(t.count("R") for t in turns if t.count("R") >= 2))


def share(cells, model, row, what):
    """Percent of all writes (reads) sharing a turn; the seed row is the seed side of the no-skill pair (all 120 tasks)."""
    side, row = ("seed", "No skill") if row == "Seed skill" else ("skill", row)
    c = next(c for c in cells if c["model"] == model and c["row"] == row and c["variant"] == "main")
    assert side == "skill" or c["n"] == 120
    return 100 * c["point"][f"{side}.{what}"], c["n"]


def draw(eps, cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 7.5, "mathtext.fontset": "stix", "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    W_IN, H_IN = 6.3, 3.2
    fig = plt.figure(figsize=(W_IN, H_IN))

    def ax_in(x, y, w, h):
        return fig.add_axes([x / W_IN, y / H_IN, w / W_IN, h / H_IN])

    hmax = max(len(t) for e in eps.values() for t in e["turns"])
    nmax = max(e["nturn"] for e in eps.values())
    SQ = 0.9
    BAND = hmax * SQ
    HEAD = 1.25
    STEP = BAND + HEAD + 0.55
    NROW = len(AG) + 1
    ybase = {a: (NROW - 1 - i) * STEP for i, a in enumerate(AG)}
    yc = {a: ybase[a] + BAND / 2 for a in AG}
    yc["No skill"] = BAND / 2 + 0.25
    ytop = (NROW - 1) * STEP + BAND + HEAD + 0.1
    ylo = -0.2
    TITLE_Y1, TITLE_Y2 = H_IN - 0.13, H_IN - 0.28

    # (a)
    ax_l, ax_w = 0.62, 3.3
    plot_y, plot_h = 0.36, 2.42
    axa = ax_in(ax_l, plot_y, ax_w, plot_h)
    for a in AG:
        e, y0 = eps[a], ybase[a]
        for t, calls in enumerate(e["turns"], start=1):
            for i, k in enumerate(sorted(calls, key="WRO".index)):
                axa.add_patch(Rectangle((t - 0.42, y0 + i * SQ + 0.05), 0.84, SQ - 0.1, fc=COL[k], ec="none", zorder=3))
            if not calls:
                axa.plot([t - 0.3, t + 0.3], [y0 + 0.45, y0 + 0.45], color=MUTED, lw=0.7, solid_capstyle="butt")
        axa.text(-0.25, yc[a], AG_LABEL[a], ha="right", va="center", fontsize=7.5, fontweight="bold", color=AG_COL[a], clip_on=False)
        hy = y0 + BAND + 0.2
        axa.text(0.55, hy, f"{e['nturn']} turns, {e['N']} calls", ha="left", va="bottom", fontsize=7, color=INK)
        axa.text(nmax + 0.5, hy, f"writes sharing a turn: {e['Wp']} of {e['W']}    reads: {e['Rp']} of {e['R']}",
                 ha="right", va="bottom", fontsize=7, color=INK)
        axa.axhline(y0 - 0.02, color=GRID, lw=0.5, zorder=1)
    axa.set_xlim(0.45, nmax + 0.55)
    axa.set_ylim(ylo, ytop)
    for sp in ("top", "right", "left", "bottom"):
        axa.spines[sp].set_visible(False)
    axa.set_yticks([])
    axa.set_xticks([1, 5, 10, 15, 20])
    axa.tick_params(axis="x", labelsize=7, length=2, pad=1)
    axa.set_xlabel("turn", fontsize=7, labelpad=1)
    lh = [Rectangle((0, 0), 1, 1, fc=COL["W"], ec="none"), Rectangle((0, 0), 1, 1, fc=COL["R"], ec="none"),
          Rectangle((0, 0), 1, 1, fc=COL["O"], ec="none"), Line2D([0], [0], color=MUTED, lw=0.7)]
    axa.legend(lh, ["write (changes state)", "read", "other call", "turn without a call"], loc="center left",
               bbox_to_anchor=(0.0, (yc["No skill"] - ylo) / (ytop - ylo)), ncol=2, frameon=False, fontsize=6.8,
               handlelength=1.0, handleheight=0.8, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0, labelspacing=0.3)
    fig.text(0.04 / W_IN, TITLE_Y1 / H_IN, "(a) One task, DeepSeek: learned skills issue writes one per turn",
             fontsize=8, fontweight="bold", ha="left", va="center", color=INK)

    # (b)
    bx0, bw, bgap = 4.36, 0.78, 0.22
    rows_b = AG + ["No skill"]
    for j, (what, title, col) in enumerate((("Wpar", "writes sharing a turn", WCOL), ("Rpar", "reads sharing a turn", RCOL))):
        ax = ax_in(bx0 + j * (bw + bgap), plot_y, bw, plot_h)
        for r in rows_b:
            y = yc[r]
            ax.axhline(y, color=GRID, lw=0.5, zorder=1)
            ds, _ = share(cells, "deepseek-flash", r, what)
            lu, n_lu = share(cells, "gpt-6-luna", r, what)
            ax.plot([ds], [y + 0.1], marker="o", ms=4.6, mfc=col, mec="white", mew=0.4, ls="", zorder=4, clip_on=False)
            ax.plot([lu], [y - 0.75], marker="D", ms=3.8, mfc="white", mec=col, mew=0.9, ls="", zorder=4, clip_on=False)
            if r in eps:
                e = eps[r]
                v = 100 * (e["Wp"] / e["W"] if what == "Wpar" else e["Rp"] / e["R"])
                ax.plot([v], [y + 0.95], marker="x", ms=4.4, mec=INK, mew=0.9, ls="", zorder=5, clip_on=False)
            if n_lu != 120 and what == "Wpar":
                ax.text(103, y - 0.75, f"{n_lu} tasks", fontsize=6.5, color=MUTED, ha="right", va="center")
        ax.axhline((yc["No skill"] + ybase["SkillAdam"]) / 2 - 0.15, color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
        ax.set_xlim(-4, 104)
        ax.set_ylim(ylo, ytop)
        ax.set_yticks([])
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100%"])
        ax.tick_params(axis="x", labelsize=7, length=2, pad=1)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.text(0.5, (ytop - 0.15 - ylo) / (ytop - ylo), title, transform=ax.transAxes, fontsize=7, ha="center", va="top", color=INK)
        if j == 0:
            ax.text(-0.06, yc["No skill"], "No skill", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.5,
                    fontweight="bold", color=AG_COL["No skill"], clip_on=False)
    tbx = ax_l + ax_w + 0.06
    fig.text(tbx / W_IN, TITLE_Y1 / H_IN, "(b) All episodes, both models: learned skills",
             fontsize=8, fontweight="bold", ha="left", va="center", color=INK)
    fig.text(tbx / W_IN, TITLE_Y2 / H_IN, "cut writes sharing a turn; reads stay grouped",
             fontsize=8, fontweight="bold", ha="left", va="center", color=INK)
    kh = [Line2D([0], [0], marker="o", ls="", ms=4.6, mfc=MUTED, mec="white", label="DeepSeek"),
          Line2D([0], [0], marker="D", ls="", ms=3.8, mfc="white", mec=MUTED, mew=0.9, label="Luna"),
          Line2D([0], [0], marker="x", ls="", ms=4.4, mec=INK, mew=0.9, label="episode in (a)")]
    fig.legend(handles=kh, loc="lower center", bbox_to_anchor=((bx0 + bw + bgap / 2) / W_IN, 0.0), ncol=3, frameon=False, fontsize=6.8,
               handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
    save_figure(fig, "bipwrites3")
    plt.close(fig)


def main():
    data, cells = load()
    eps = {a: episode(data, a) for a in AG}
    draw(eps, cells)
    out = dict(episodes={AG_LABEL[a]: {k: v for k, v in e.items() if k != "turns"} | dict(turns=["".join(t) or "-" for t in e["turns"]])
                         for a, e in eps.items()},
               shares={f"{m}|{r}|{w}": share(cells, m, r, w) for m in ("deepseek-flash", "gpt-6-luna") for r in AG + ["No skill"]
                       for w in ("Wpar", "Rpar")})
    C.write_json("bipwrites3", out)
    for k, v in out["shares"].items():
        print(k, "%.1f" % v[0], v[1])


if __name__ == "__main__":
    main()
