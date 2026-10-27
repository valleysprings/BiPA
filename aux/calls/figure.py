"""The serialized-writes figure (mech_bipwrites3, AutoBench): the serialize rules put writes one per turn while reads stay
grouped. (a) The teaser's task in DeepSeek's first test run, for the seed and the three learned skills: one column per
turn, one square per executed call (write, read, other). (b) The share of writes (reads) that share a turn with another
write (read) over every complete test episode, the main variant of mech_calls_rw; filled circle DeepSeek, hollow diamond
Luna, cross the episode of (a). Drawn from the calls_rw.json and episode.json that aux.calls.calls writes."""
import json

from src.analysis import settings as S
from src.utils import fig as F

ROWS = ("native-seed", "gepa", "skillopt", "skilladam")
COLOR = dict(W=F.CALL["write"], R=F.CALL["read_light"], O=F.FAINT)
SHARE = (("Wpar", "writes sharing a turn", F.CALL["write"]), ("Rpar", "reads sharing a turn", F.CALL["read_line"]))


def share(cells, model, method, what):
    """Percent of all writes (reads) sharing a turn, and the tasks; the seed's is the seed side of the No skill pair."""
    side, row = ("seed", "No skill") if method == "native-seed" else ("skill", S.ARM_NAME[method])
    c = next(c for c in cells if (c["model"], c["row"], c["variant"]) == (model, row, "main"))
    return 100 * c["point"][f"{side}.{what}"], c["n"]


def episode(turns):
    turns = [[] if t == "-" else list(t) for t in turns]
    count = lambda k, least=1: sum(t.count(k) for t in turns if t.count(k) >= least)
    return dict(turns=turns, nturn=len(turns), N=sum(map(len, turns)), W=count("W"), R=count("R"), Wp=count("W", 2), Rp=count("R", 2))


def draw(eps, cells):
    plt = F.pyplot(F.TEXT, title=F.TITLE, label=F.SMALL, tick=F.SMALL, legend=6.8)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    width, height = F.TEXT_W, 3.2
    fig = plt.figure(figsize=(width, height))
    ax_in = lambda x, y, w, h: fig.add_axes([x / width, y / height, w / width, h / height])
    label = {m: (S.ARM_NAME[m], F.ARM[m]) for m in ROWS} | {"native-noseed": ("No skill", F.MUTED)}
    band = max(len(t) for e in eps.values() for t in e["turns"]) * 0.9
    nmax, step, ylo = max(e["nturn"] for e in eps.values()), band + 1.25 + 0.55, -0.2
    ybase = {m: (len(ROWS) - i) * step for i, m in enumerate(ROWS)}
    yc = {m: ybase[m] + band / 2 for m in ROWS} | {"native-noseed": band / 2 + 0.25}
    ytop = len(ROWS) * step + band + 1.35
    # (a) one episode
    axa = ax_in(0.62, 0.36, 3.3, 2.42)
    for m in ROWS:
        e, y0 = eps[m], ybase[m]
        for t, calls in enumerate(e["turns"], start=1):
            for i, k in enumerate(sorted(calls, key="WRO".index)):
                axa.add_patch(Rectangle((t - 0.42, y0 + i * 0.9 + 0.05), 0.84, 0.8, fc=COLOR[k], ec="none", zorder=3))
            if not calls:
                axa.plot([t - 0.3, t + 0.3], [y0 + 0.45] * 2, color=F.MUTED, lw=0.7, solid_capstyle="butt")
        axa.text(-0.25, yc[m], label[m][0], ha="right", va="center", fontsize=F.TEXT, fontweight="bold", color=label[m][1], clip_on=False)
        axa.text(0.55, y0 + band + 0.2, f"{e['nturn']} turns, {e['N']} calls", ha="left", va="bottom", fontsize=F.SMALL)
        axa.text(nmax + 0.5, y0 + band + 0.2, f"sharing a turn: {e['Wp']} of {e['W']} writes, {e['Rp']} of {e['R']} reads",
                 ha="right", va="bottom", fontsize=F.SMALL)
        axa.axhline(y0 - 0.02, color=F.GRID, lw=0.5, zorder=1)
    axa.set(xlim=(0.45, nmax + 0.55), ylim=(ylo, ytop), yticks=[], xticks=[1, 5, 10, 15, 20])
    axa.spines[:].set_visible(False)
    axa.tick_params(axis="x", length=2, pad=1)
    axa.set_xlabel("turn", labelpad=1)
    axa.legend([Rectangle((0, 0), 1, 1, fc=COLOR[k], ec="none") for k in "WRO"] + [Line2D([0], [0], color=F.MUTED, lw=0.7)],
               ["write (changes state)", "read", "other call", "turn without a call"], loc="center left",
               bbox_to_anchor=(0.0, (yc["native-noseed"] - ylo) / (ytop - ylo)), ncol=2, frameon=False, handlelength=1.0,
               handleheight=0.8, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0, labelspacing=0.3)
    title = lambda x, y, text: fig.text(x / width, y / height, text, fontsize=F.TITLE, fontweight="bold", ha="left", va="center")
    title(0.04, height - 0.13, "(a) One task, DeepSeek: learned skills issue writes one per turn")
    # (b) every episode, both models
    for j, (what, name, color) in enumerate(SHARE):
        ax = ax_in(4.36 + j, 0.36, 0.78, 2.42)
        for m in ROWS + ("native-noseed",):
            y = yc[m]
            ax.axhline(y, color=F.GRID, lw=0.5, zorder=1)
            for model, dy, style in ((S.DEEPSEEK, 0.1, dict(ms=4.6, mfc=color, mec="white", mew=0.4)),
                                     (S.LUNA, -0.75, dict(ms=3.8, mfc="white", mec=color, mew=0.9))):
                value, _ = share(cells, model, m, what)
                ax.plot([value], [y + dy], marker=F.MODEL_MARKER[model]["marker"], ls="", zorder=4, clip_on=False, **style)
            if m in eps:
                e = eps[m]
                ax.plot([100 * (e["Wp"] / e["W"] if what == "Wpar" else e["Rp"] / e["R"])], [y + 0.95], marker="x", ms=4.4, mec=F.INK,
                        mew=0.9, ls="", zorder=5, clip_on=False)
        ax.axhline((yc["native-noseed"] + ybase["skilladam"]) / 2 - 0.15, color=F.MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
        ax.set(xlim=(-4, 104), ylim=(ylo, ytop), yticks=[], xticks=[0, 50, 100], xticklabels=["0", "50", "100%"])
        ax.tick_params(axis="x", length=2, pad=1)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.text(0.5, (ytop - 0.15 - ylo) / (ytop - ylo), name, transform=ax.transAxes, fontsize=F.SMALL, ha="center", va="top")
        if j == 0:
            ax.text(-0.06, yc["native-noseed"], "No skill", transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=F.TEXT, fontweight="bold", color=F.MUTED, clip_on=False)
    title(3.98, height - 0.13, "(b) All episodes, both models: learned skills")
    title(3.98, height - 0.28, "cut writes sharing a turn; reads stay grouped")
    keys = [Line2D([0], [0], marker="o", ls="", ms=4.6, mfc=F.MUTED, mec="white", label="DeepSeek"),
            Line2D([0], [0], marker="D", ls="", ms=3.8, mfc="white", mec=F.MUTED, mew=0.9, label="Luna"),
            Line2D([0], [0], marker="x", ls="", ms=4.4, mec=F.INK, mew=0.9, label="that row's episode in (a)")]
    fig.legend(handles=keys, loc="lower center", bbox_to_anchor=(5.1 / width, 0.0), ncol=3, frameon=False, handletextpad=0.2,
               columnspacing=0.8, borderaxespad=0.1)
    F.save_figure(fig, "bipwrites3")
    plt.close(fig)


def main():
    folder = S.cache("calls")
    turns = json.loads((folder / "episode.json").read_text())
    draw({m: episode(turns[S.ARM_NAME[m]]) for m in ROWS}, json.loads((folder / "calls_rw.json").read_text())["cells"])


if __name__ == "__main__":
    main()
