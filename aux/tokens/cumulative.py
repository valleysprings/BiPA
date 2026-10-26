"""How the extra input tokens build up within a task: the input tokens a task has read up to each turn, by context
component (rest of the prompt, skill text, tool outputs, agent and user messages), for the seed and each learned skill.

Per task the cumulative tokens of each component up to turn r (an ended episode keeps its total), averaged over the task's
test runs, then over tasks; a learned skill is compared with the seed on the tasks run under both, and its extra tokens
are the difference of the full-episode totals (relative: over the seed's total, a ratio of mean totals, so long tasks
weigh more than in rho). Tokens of a turn are split between components by the episode graph (src.graph.episode).

mech_bipcum3 has a row per model and benchmark on which every optimizer learned a skill, the curves running to the
longest episode of the row; the record also gives the learned skills of any other benchmark. mech_bipcum_main2 shows the
seed against SkillOpt under DeepSeek, its curves cut at the ninety-fifth percentile of the episode lengths of the seed and
the learned skills."""
from collections import defaultdict

import numpy as np

from src.analysis import settings as S
from src.analysis.attribution import side
from src.utils import fig as style
from src.utils.logger import md_table, record

GROUPS = ["prompt", "skill", "tools", "messages"]
LABEL = dict(prompt="system prompt, tools, task", skill="skill text", tools="tool outputs", messages="agent and user messages")
FEATURED = "skillopt"  # the learned skill of the main-text figure
MODEL_Y = {S.DEEPSEEK: 0.785, S.LUNA: 0.215}  # where mech_bipcum3 names each model's block of rows


def group(comp):
    return {"system": "prompt", "tool_schemas": "prompt", "task": "prompt", "skill": "skill", "history": "messages"}.get(
        comp, "tools" if comp.startswith("tool:") else "other")


def matrices(model, bench, method):
    """[(task, turns x components tokens)] of every graph episode of a row's test runs."""
    out = []
    for rep in side(model, bench, method)[0].values():
        for task, (e, _) in rep.items():
            M = np.zeros((e.R, len(GROUPS)))
            for j, comp in enumerate(e.comps):
                if (g := group(comp)) in GROUPS:
                    M[:, GROUPS.index(g)] += e.Xrg[:, j]
            out.append((task, M))
    return out


def per_task(eps, rmax):
    """({task: run mean of the cumulative tokens up to rmax turns}, {task: run mean of the full-episode totals})."""
    cum, tot = defaultdict(list), defaultdict(list)
    for task, M in eps:
        c = np.cumsum(M, axis=0)
        tot[task].append(c[-1])
        c = c[:rmax]
        cum[task].append(np.vstack([c, np.repeat(c[-1:], rmax - len(c), axis=0)]))
    return {t: np.mean(v, axis=0) for t, v in cum.items()}, {t: np.mean(v, axis=0) for t, v in tot.items()}


def cell(model, bench, cut=None):
    """The seed's curves and totals, and per learned skill its gap to the seed on the tasks both ran."""
    opts = [o for m, b, o in S.learned([model], [bench])]
    eps = {o: matrices(model, bench, o) for o in ["native-seed"] + opts}
    lengths = [M.shape[0] for v in eps.values() for _, M in v]
    rmax = int(np.ceil(np.percentile(lengths, cut))) if cut else max(lengths)
    seed_cum, seed_tot = per_task(eps["native-seed"], rmax)
    out = dict(model=model, bench=bench, rmax=rmax, seed=np.mean(list(seed_cum.values()), axis=0),
               seed_end=np.mean(list(seed_tot.values()), axis=0), seed_tasks=len(seed_tot), opts={})
    for o in opts:
        own_cum, own_tot = per_task(eps[o], rmax)
        tasks = sorted(set(seed_tot) & set(own_tot))
        mean = lambda d: np.mean([d[t] for t in tasks], axis=0)
        out["opts"][o] = dict(n=len(tasks), d=mean(own_cum) - mean(seed_cum), curve=mean(own_cum), seed_curve=mean(seed_cum),
                              seed_end=mean(seed_tot), learned_end=mean(own_tot))
    return out


def kfmt(v, sign=True):
    """v in thousands of tokens -> '707k', '+175k', '−21k', '0'; 1M or more -> '1.10M'."""
    if abs(v) >= 999.5:
        body = "%.2fM" % (abs(v) / 1000)
    elif (r := int(np.floor(abs(v) + 0.5))) == 0:
        return "0"
    else:
        body = "%dk" % r
    return ("−" if v < 0 else "+" if sign else "") + body


def pfmt(v):
    v = int(np.floor(abs(v) + 0.5)) * (1 if v >= 0 else -1)
    return "+%d" % v if v > 0 else ("−%d" % -v if v < 0 else "0")


def ticks_k(ax, ticks=None):
    """y ticks in thousands of tokens as k, or as M when the top tick reaches 1M."""
    from matplotlib.ticker import FixedFormatter, FixedLocator
    lo, hi = ax.get_ylim()
    ticks = [t for t in (ax.get_yticks() if ticks is None else ticks) if lo <= t <= hi]
    use_m = max(abs(t) for t in ticks) >= 1000
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(["0" if t == 0 else ("−" if t < 0 else "") + ("%gM" % (abs(t) / 1000) if use_m else "%gk" % abs(t))
                                                 for t in ticks]))


def draw_bipcum3(cells):
    """Column 1: the seed (stacked, absolute); columns 2-4: each optimizer minus the seed per component, stacked above and
    below zero, with the net gap as a black line; beside each panel the four components at the end of the episode."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    plt = style.pyplot(style.TEXT)
    models = list(dict.fromkeys(c["model"] for c in cells))
    heights = [r for m in models for r in [1] * sum(c["model"] == m for c in cells) + [0.16]][:-1]
    rows = [k for k, h in enumerate(heights) if h == 1]
    fig = plt.figure(figsize=(style.TEXT_W, 5.5))
    outer = GridSpec(len(heights), 5, figure=fig, height_ratios=heights, width_ratios=[1, 0.06, 1, 1, 1],
                     left=0.105, right=0.995, top=0.94, bottom=0.085, hspace=0.36, wspace=0.12)
    last = {max(i for i, c in enumerate(cells) if c["model"] == m) for m in models}
    for ri, c in enumerate(cells):
        r, x = c["rmax"], np.arange(1, c["rmax"] + 1)
        gaps = [o["d"] / 1000 for o in c["opts"].values()]
        hi = max(max(np.clip(d, 0, None).sum(1).max(), d.sum(1).max()) for d in gaps + [np.zeros((1, 4))])
        lo = min(min(np.clip(d, None, 0).sum(1).min(), d.sum(1).min()) for d in gaps + [np.zeros((1, 4))])
        for k, opt in enumerate([None] + list(S.OPTIMIZERS)):
            sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[rows[ri], [0, 2, 3, 4][k]], width_ratios=[1, 0.40], wspace=0.04)
            ax, axn = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
            if opt is None:
                stack, bottom = c["seed"] / 1000, np.zeros(r)
                for j, g in enumerate(GROUPS):
                    ax.fill_between(x, bottom, bottom + stack[:, j], step="mid", color=style.COMPONENT[g], lw=0)
                    bottom = bottom + stack[:, j]
                ax.set_ylim(0, bottom.max() * 1.25)
                ax.text(0.03, 0.97, kfmt(bottom[-1], False), transform=ax.transAxes, fontsize=6.8, va="top", ha="left", color=style.INK, fontweight="bold")
                ends, sign = c["seed"][-1] / 1000, False
                ax.set_ylabel(style.BENCH_LABEL[c["bench"]], fontsize=7.5, fontweight="bold", color=style.INK, labelpad=2)
            else:
                o, d = c["opts"][opt], c["opts"][opt]["d"] / 1000
                pos, neg = np.zeros(r), np.zeros(r)
                for j, g in enumerate(GROUPS):
                    p, q = np.clip(d[:, j], 0, None), np.clip(d[:, j], None, 0)
                    ax.fill_between(x, pos, pos + p, step="mid", color=style.COMPONENT[g], lw=0)
                    ax.fill_between(x, neg, neg + q, step="mid", color=style.COMPONENT[g], lw=0)
                    pos, neg = pos + p, neg + q
                ax.axhline(0, color=style.INK, lw=0.5)
                ax.step(x, d.sum(1), where="mid", color=style.INK, lw=1.0)
                ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.22 * (hi - lo))
                ends, sign = (o["learned_end"] - o["seed_end"]) / 1000, True
                ax.text(0.03, 0.97, "%s, %s%%" % (kfmt(ends.sum()), pfmt(100 * ends.sum() / (o["seed_end"].sum() / 1000))), transform=ax.transAxes,
                        fontsize=6.8, va="top", ha="left", color=style.ARM[opt], fontweight="bold")
            axn.set(xlim=(0, 1), ylim=(0, 1))
            axn.axis("off")
            for n, j in enumerate(range(3, -1, -1)):
                axn.add_patch(Rectangle((0.0, 0.745 - n * 0.2), 0.12, 0.11, color=style.COMPONENT[GROUPS[j]], lw=0, transform=axn.transAxes))
                axn.text(0.98, 0.80 - n * 0.2, kfmt(ends[j], sign), ha="right", va="center", fontsize=6.5, color=style.INK, transform=axn.transAxes)
            ax.set_xlim(0.5, r + 0.5)
            style.despine(ax)
            ax.grid(axis="y", color=style.GRID, lw=0.4)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=6.5, pad=1.5)
            if ri in last:
                ax.set_xlabel("turn", fontsize=7, labelpad=1)
            ticks_k(ax)
            if k > 1:
                ax.set_yticklabels([])
            if ri == 0:
                ax.set_title("Seed" if opt is None else S.ARM_NAME[opt] + " − seed", fontsize=8, fontweight="bold", loc="left", pad=3,
                             color=style.INK if opt is None else style.ARM[opt])
    fig.text(0.065, 0.985, "y axes: input tokens read so far per task", fontsize=7, color=style.MUTED, ha="left", va="top")
    for m in models:
        fig.text(0.012, MODEL_Y[m], S.MODEL_NAME[m], rotation=90, va="center", fontsize=8.5, fontweight="bold", color=style.INK)
    handles = [Patch(color=style.COMPONENT[g], label=LABEL[g]) for g in GROUPS] + [Line2D([0], [0], color=style.INK, lw=1.0, label="total gap to the seed")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.55, 0.0), handlelength=1.4, columnspacing=1.2)
    style.save_figure(fig, "bipcum3")


def draw_main(cells):
    """Top row: the seed; bottom row: the featured skill, with the seed on the same tasks dashed."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    plt, fs, colour = style.pyplot(style.TEXT), 7.0, style.ARM[FEATURED]
    fig, axes = plt.subplots(2, len(cells), figsize=(style.COL_W, 2.65), sharex="col")
    for ci, c in enumerate(cells):
        o = c["opts"][FEATURED]
        x = np.arange(1, c["rmax"] + 1)
        ymax = max(c["seed"].sum(1).max(), o["curve"].sum(1).max(), o["seed_curve"].sum(1).max()) * 1.05 / 1000
        for ri, curve in enumerate((c["seed"], o["curve"])):
            ax, bottom = axes[ri, ci], np.zeros(c["rmax"])
            for j, g in enumerate(GROUPS):
                ax.fill_between(x, bottom, bottom + curve[:, j] / 1000, step="mid", color=style.COMPONENT[g], lw=0, alpha=0.95)
                bottom = bottom + curve[:, j] / 1000
            if ri == 1:
                ax.step(x, o["seed_curve"].sum(1) / 1000, where="mid", color=style.INK, lw=0.8, ls=(0, (3, 2)))
                ax.text(0.05, 0.95, "%+.0f%%" % extra_pct(o), transform=ax.transAxes, fontsize=fs, va="top", ha="left", color=colour, fontweight="bold")
            ax.set_xlim(0.5, c["rmax"] + 0.5)
            ax.set_ylim(0, ymax)
            ticks_k(ax, MaxNLocator(nbins=5, steps=[1, 2, 5, 10]).tick_values(0, ymax))
            style.despine(ax)
            ax.grid(axis="y", color=style.GRID, lw=0.4)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=6.5, pad=1.2, length=2)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
            if ri == 0:
                ax.set_title(style.BENCH_LABEL[c["bench"]], fontsize=fs + 0.5, fontweight="bold", loc="left", pad=2)
            else:
                ax.set_xlabel("turn", fontsize=fs, labelpad=1)
            if ci == 0:
                ax.set_ylabel(["Seed", S.ARM_NAME[FEATURED]][ri], fontsize=fs + 0.5, fontweight="bold", color=[style.INK, colour][ri], labelpad=1)
    fig.text(0.006, 0.585, "input tokens read so far per task", rotation=90, va="center", ha="left", fontsize=fs, color=style.MUTED)
    handles = [Patch(color=style.COMPONENT[g], label=LABEL[g]) for g in GROUPS]
    handles.append(Line2D([0], [0], color=style.INK, lw=0.8, ls=(0, (3, 2)), label="seed, same tasks"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=6.5, handlelength=1.2, columnspacing=0.8,
               handletextpad=0.4, labelspacing=0.25, bbox_to_anchor=(0.53, 0.0))
    fig.tight_layout(rect=(0.04, 0.145, 1, 1), w_pad=0.3, h_pad=0.4)
    style.save_figure(fig, "bipcum_main2")


def extra_pct(o):
    return 100 * (o["learned_end"].sum() - o["seed_end"].sum()) / o["seed_end"].sum()


def main():
    k = lambda v: f"{v / 1000:.1f}"
    pairs = S.learned_pairs()
    cells = [cell(m, b) for m, b in pairs]
    full = [c for c in cells if len(c["opts"]) == len(S.OPTIMIZERS)]
    main_cells = [cell(S.DEEPSEEK, b, cut=95) for m, b in pairs if m == S.DEEPSEEK and (m, b, FEATURED) in S.learned([m])]
    name = lambda c, *opt: " ".join([S.MODEL_SHORT[c["model"]], S.BENCH_SHORT[c["bench"]]] + [S.ARM_NAME[o] for o in opt])
    seed_rows = [[name(c), c["rmax"], c["seed_tasks"], k(c["seed_end"].sum())] + [k(v) for v in c["seed_end"]] for c in cells]
    gap_rows = [[name(c, opt), o["n"], k(o["seed_end"].sum()), k(o["learned_end"].sum()),
                 f"{(o['learned_end'] - o['seed_end']).sum() / 1000:+.1f}", f"{extra_pct(o):+.1f}%"]
                + [f"{(o['learned_end'][j] - o['seed_end'][j]) / 1000:+.1f}" for j in range(len(GROUPS))] for c in cells for opt, o in c["opts"].items()]
    main_rows = [[name(c, FEATURED), o["n"], c["rmax"], k(o["seed_end"].sum()), k(o["learned_end"].sum()), f"{extra_pct(o):+.1f}%"]
                 + [f"{(o['learned_end'][j] - o['seed_end'][j]) / 1000:+.1f}" for j in range(len(GROUPS))] for c in main_cells for o in [c["opts"][FEATURED]]]
    left = [name(c) for c in cells if c not in full]
    comps = [LABEL[g] for g in GROUPS]
    numbers = "\n\n".join([
        "Seed, input tokens per task in thousands (full episodes):",
        md_table(["seed", "turns drawn", "tasks", "total", *comps], seed_rows),
        "Learned skill against the seed over the same tasks, thousands of tokens (extra = learned minus seed):",
        md_table(["skill", "tasks", "seed total", "learned total", "extra", "relative", *(f"extra: {c}" for c in comps)], gap_rows),
        f"Main-text figure, {S.MODEL_NAME[S.DEEPSEEK]}, seed against {S.ARM_NAME[FEATURED]}:",
        md_table(["skill", "tasks", "turns drawn", "seed total", f"{S.ARM_NAME[FEATURED]} total", "relative", *(f"extra: {c}" for c in comps)], main_rows)]
        + ([f"Not drawn in mech_bipcum3 (not every optimizer learned a skill there): {', '.join(left)}."] if left else []))
    record(__spec__.name, "bipcum", numbers, S.runs_text([(m, b, o) for m, b in pairs for o in ["native-seed"] + [x for _, _, x in S.learned([m], [b])]]))
    draw_bipcum3(full)
    draw_main(main_cells)


if __name__ == "__main__":
    main()
