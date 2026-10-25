"""The per-skill cumulative-tokens figure: each learned skill's gap to the seed, per context component, in the input tokens a task has read so far.

Per task, the cumulative tokens read from each component up to turn r (an ended episode keeps its total), averaged over the task's runs,
then over tasks. Column 1: the seed over all its tasks (stacked, absolute); columns 2-4: GEPA, SkillOpt, SkillAdam minus the seed on the
tasks run under both, per component, stacked above and below zero, with the net gap as a black line. Turns run to the longest episode of
the cell, so the right end of every curve is the full-episode value; beside each panel, the four components at the end of the episode.
Rows: DeepSeek tau3 / AutoBench / TB2, Luna tau3 / AutoBench (on Luna TB2 every optimizer returned the seed).
Reads bipdata.pkl (python -m aux.token_attribution.bipdata); writes <cache>/analysis/main/bipcum3.json and fig/mech/mech_bipcum3.
"""
import pickle
from collections import defaultdict

import numpy as np

from src import layout
from src.utils import dataloader as C
from src.utils.fig import save_figure

DATA = layout.OUT / "token_attribution/bipdata.pkl"
INK, MUTED, GRID = "#222222", "#5A5A5A", "#E2E5EA"
BENCH_NAME = {"tau3-bench": "τ³-bench", "automationbench": "AutoBench", "terminal-bench-2-1": "TB2"}
STACK = ["prompt", "skill", "tools", "messages"]
LABEL = dict(prompt="system prompt, tools, task", skill="skill text", tools="tool outputs", messages="agent and user messages")
COLOR = dict(prompt="#BBBBBB", skill="#D55E00", tools="#009E73", messages="#56B4E9")
OPTS = [("gepa", "GEPA", "#2A9D4B"), ("skillopt", "SkillOpt", "#1F6FB5"), ("skilladam", "SkillAdam", "#E08A1E")]
COLS = [("deepseek-flash", "tau3-bench"), ("deepseek-flash", "automationbench"), ("deepseek-flash", "terminal-bench-2-1"),
        ("gpt-6-luna", "tau3-bench"), ("gpt-6-luna", "automationbench")]


def cum_by_task(eps, rmax):
    """{task: rmax x 4 run mean of the cumulative tokens per component up to turn r}; an ended episode keeps its total."""
    per = defaultdict(list)
    for e in eps:
        c = np.cumsum(e["M"][:rmax], axis=0)
        if c.shape[0] < rmax:
            c = np.vstack([c, np.repeat(c[-1:], rmax - c.shape[0], axis=0)])
        per[e["task"]].append(c)
    return {t: np.mean(v, axis=0) for t, v in per.items()}


def cell(episodes, model, bench):
    seed = episodes[(model, bench, "seed")]
    rmax = int(max([e["R"] for e in seed] + [e["R"] for o, _, _ in OPTS for e in episodes.get((model, bench, o), [])]))
    cs = cum_by_task(seed, rmax)
    out = dict(rmax=rmax, seed=np.mean(list(cs.values()), axis=0), seed_tasks=len(cs), opts={})
    for opt, _, _ in OPTS:
        if episodes.get((model, bench, opt)):
            cl = cum_by_task(episodes[(model, bench, opt)], rmax)
            tasks = sorted(set(cs) & set(cl))
            out["opts"][opt] = dict(d=np.mean([cl[t] - cs[t] for t in tasks], axis=0), n=len(tasks),
                                    seed_end=np.mean([cs[t][-1] for t in tasks], axis=0), learned_end=np.mean([cl[t][-1] for t in tasks], axis=0))
    return out


def kfmt(v, sign=True):
    """v in thousands of tokens -> '707k', '+175k', '−21k', '0'; 1M or more -> '1.10M'."""
    if abs(v) >= 999.5:
        body = "%.2fM" % (abs(v) / 1000.0)
    else:
        r = int(np.floor(abs(v) + 0.5))
        if r == 0:
            return "0"
        body = "%dk" % r
    return ("−" if v < 0 else "+" if sign else "") + body


def pfmt(v):
    v = int(np.floor(abs(v) + 0.5)) * (1 if v >= 0 else -1)
    return "+%d" % v if v > 0 else ("−%d" % -v if v < 0 else "0")


def tick_fmt(ax):
    """y ticks in thousands of tokens as k, or as M when the top tick reaches 1M."""
    from matplotlib.ticker import FixedFormatter, FixedLocator
    lo, hi = ax.get_ylim()
    ticks = [t for t in ax.get_yticks() if lo <= t <= hi]
    use_m = max(abs(t) for t in ticks) >= 1000
    labels = ["0" if t == 0 else ("−" if t < 0 else "") + ("%gM" % (abs(t) / 1000.0) if use_m else "%gk" % abs(t)) for t in ticks]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(labels))


def style(ax, last_row):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.5, pad=1.5)
    if last_row:
        ax.set_xlabel("turn", fontsize=7, labelpad=1)


def side_numbers(axn, vals, sign):
    """The four component values beside a panel, top to bottom in stack order, each with a colour key."""
    from matplotlib.patches import Rectangle
    axn.set_xlim(0, 1)
    axn.set_ylim(0, 1)
    axn.axis("off")
    for k, j in enumerate(range(3, -1, -1)):
        y = 0.80 - k * 0.2
        axn.add_patch(Rectangle((0.0, y - 0.055), 0.12, 0.11, color=COLOR[STACK[j]], lw=0, transform=axn.transAxes))
        axn.text(0.98, y, kfmt(vals[j], sign), ha="right", va="center", fontsize=6.5, color=INK, transform=axn.transAxes)


def draw(cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 7.5, "mathtext.fontset": "stix", "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    fig = plt.figure(figsize=(6.3, 5.5))
    outer = GridSpec(6, 5, figure=fig, height_ratios=[1, 1, 1, 0.16, 1, 1], width_ratios=[1, 0.06, 1, 1, 1],
                     left=0.105, right=0.995, top=0.94, bottom=0.085, hspace=0.36, wspace=0.12)
    grow, gcol = [0, 1, 2, 4, 5], [0, 2, 3, 4]
    for ri, ((model, bench), c) in enumerate(zip(COLS, cells)):
        r = c["rmax"]
        x = np.arange(1, r + 1)
        lo, hi = 0.0, 0.0
        for o in c["opts"].values():
            d = o["d"] / 1000.0
            hi = max(hi, np.clip(d, 0, None).sum(1).max(), d.sum(1).max())
            lo = min(lo, np.clip(d, None, 0).sum(1).min(), d.sum(1).min())
        span = hi - lo
        for k in range(4):
            sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[grow[ri], gcol[k]], width_ratios=[1, 0.40], wspace=0.04)
            ax, axn = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
            if k == 0:
                bottom = np.zeros(r)
                for j, g in enumerate(STACK):
                    v = c["seed"][:, j] / 1000.0
                    ax.fill_between(x, bottom, bottom + v, step="mid", color=COLOR[g], lw=0)
                    bottom = bottom + v
                ax.set_ylim(0, bottom.max() * 1.25)
                ax.text(0.03, 0.97, kfmt(bottom[-1], False), transform=ax.transAxes, fontsize=6.8, va="top", ha="left", color=INK, fontweight="bold")
                side_numbers(axn, c["seed"][-1] / 1000.0, False)
                ax.set_ylabel(BENCH_NAME[bench], fontsize=7.5, fontweight="bold", color=INK, labelpad=2)
            else:
                opt, _, ocol = OPTS[k - 1]
                o = c["opts"][opt]
                d = o["d"] / 1000.0
                pos0, neg0 = np.zeros(r), np.zeros(r)
                for j, g in enumerate(STACK):
                    p, q = np.clip(d[:, j], 0, None), np.clip(d[:, j], None, 0)
                    ax.fill_between(x, pos0, pos0 + p, step="mid", color=COLOR[g], lw=0)
                    ax.fill_between(x, neg0, neg0 + q, step="mid", color=COLOR[g], lw=0)
                    pos0 += p
                    neg0 += q
                ax.axhline(0, color=INK, lw=0.5)
                ax.step(x, d.sum(1), where="mid", color=INK, lw=1.0)
                ax.set_ylim(lo - 0.05 * span, hi + 0.22 * span)
                ext = (o["learned_end"] - o["seed_end"]) / 1000.0
                pct = ext.sum() / (o["seed_end"].sum() / 1000.0) * 100
                ax.text(0.03, 0.97, "%s, %s%%" % (kfmt(ext.sum()), pfmt(pct)), transform=ax.transAxes, fontsize=6.8, va="top", ha="left",
                        color=ocol, fontweight="bold")
                side_numbers(axn, ext, True)
            ax.set_xlim(0.5, r + 0.5)
            style(ax, ri in (2, 4))
            tick_fmt(ax)
            if k > 1:
                ax.set_yticklabels([])
            if ri == 0:
                ax.set_title("Seed" if k == 0 else OPTS[k - 1][1] + " − seed", fontsize=8, fontweight="bold", loc="left", pad=3,
                             color=INK if k == 0 else OPTS[k - 1][2])
    fig.text(0.065, 0.985, "y axes: input tokens read so far per task", fontsize=7, color=MUTED, ha="left", va="top")
    fig.text(0.012, 0.785, "DeepSeek-V4.1-Flash", rotation=90, va="center", fontsize=8.5, fontweight="bold", color=INK)
    fig.text(0.012, 0.215, "GPT-6-Luna", rotation=90, va="center", fontsize=8.5, fontweight="bold", color=INK)
    handles = [Patch(color=COLOR[g], label=LABEL[g]) for g in STACK] + [Line2D([0], [0], color=INK, lw=1.0, label="total gap to the seed")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.55, 0.0), handlelength=1.4, columnspacing=1.2)
    save_figure(fig, "bipcum3")


def numbers(cells):
    res = {}
    for (model, bench), c in zip(COLS, cells):
        res[f"{model}|{bench}|seed"] = dict(turns=c["rmax"], tasks=c["seed_tasks"], total_k=round(float(c["seed"][-1].sum() / 1000), 1),
                                            by_component_k={g: round(float(c["seed"][-1, j] / 1000), 1) for j, g in enumerate(STACK)})
        for opt, o in c["opts"].items():
            ext = (o["learned_end"] - o["seed_end"]) / 1000.0
            res[f"{model}|{bench}|{opt}"] = dict(turns=c["rmax"], tasks=o["n"], seed_total_k=round(float(o["seed_end"].sum() / 1000), 1),
                                                 learned_total_k=round(float(o["learned_end"].sum() / 1000), 1), extra_k=round(float(ext.sum()), 1),
                                                 extra_pct=round(float(ext.sum() / (o["seed_end"].sum() / 1000.0) * 100), 1),
                                                 extra_k_by_component={g: round(float(ext[j]), 1) for j, g in enumerate(STACK)})
    return res


def main():
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}: run python -m aux.token_attribution.attribution_luna, then python -m aux.token_attribution.bipdata")
    episodes = pickle.loads(DATA.read_bytes())["episodes"]
    cells = [cell(episodes, m, b) for m, b in COLS]
    res = numbers(cells)
    C.write_json("bipcum3", res)
    for k, v in res.items():
        print(k, v)
    draw(cells)


if __name__ == "__main__":
    main()
