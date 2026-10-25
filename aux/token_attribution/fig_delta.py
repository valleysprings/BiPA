"""Figure bip_delta: what each learned skill changes, turn by turn and component by component, against the seed.

Same data as bip_view (bipdata.pkl, per-turn tokens by component group, the paper's own test episodes).  For every model, benchmark and optimizer the figure shows
  delta(r, g) = mean tokens read at turn r from group g by the optimizer's episodes  -  the same for the seed's episodes,
means taken over the episodes still running at turn r.  Groups: system prompt, tools and task | skill text | tool outputs | agent and user messages.
Signed stacked areas, black line = total change.  Turns are shown up to the last turn at which at least 30% of the seed's episodes AND 30% of the optimizer's episodes are still running.
Rows: GEPA, SkillOpt, SkillAdam.  Columns: DeepSeek tau3 / AutoBench / TB2, Luna tau3 / AutoBench (Luna TB2: every optimizer returned the seed).

Run:  python -m aux.token_attribution.fig_delta
"""
import json
import pickle

import numpy as np

from aux.token_attribution.bipcommon import *  # noqa: F401,F403

D = None
STACK = ["prompt", "skill", "tools", "messages"]
LABEL = dict(prompt="system prompt, tools, task", skill="skill text", tools="tool outputs", messages="agent and user messages")
COLOR = dict(prompt="#BBBBBB", skill="#D55E00", tools="#009E73", messages="#56B4E9")
OPTS = [("gepa", "GEPA", "#2A9D4B"), ("skillopt", "SkillOpt", "#1F6FB5"), ("skilladam", "SkillAdam", "#E08A1E")]
COLS = [("deepseek-flash", "tau3-bench"), ("deepseek-flash", "automationbench"), ("deepseek-flash", "terminal-bench-2-1"),
        ("gpt-6-luna", "tau3-bench"), ("gpt-6-luna", "automationbench")]
KEEP = 0.30


def profile(eps, rmax):
    S = np.zeros((rmax, 4))
    n = np.zeros(rmax)
    for e in eps:
        k = min(e["R"], rmax)
        S[:k] += e["M"][:k]
        n[:k] += 1
    return S / np.maximum(n, 1)[:, None], n


def last_turn(eps):
    rmax = max(e["R"] for e in eps)
    _, n = profile(eps, rmax)
    return max(r for r in range(1, rmax + 1) if n[r - 1] / len(eps) >= KEEP)


def delta(model, bench, opt):
    seed, L = D[(model, bench, "seed")], D[(model, bench, opt)]
    r = min(last_turn(seed), last_turn(L))
    ms, _ = profile(seed, r)
    ml, _ = profile(L, r)
    return r, ml - ms, len(seed), len(L)


def load():
    """Read the inputs this figure draws from."""
    global D
    D = pickle.loads((OUT / "bipdata.pkl").read_bytes())["episodes"]


def main():
    load()
    plt = setup_mpl()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(7.1, 5.0))
    gs = GridSpec(3, 6, figure=fig, width_ratios=[1, 1, 1, 0.3, 1, 1], left=0.095, right=0.99, top=0.865, bottom=0.115, hspace=0.16, wspace=0.34)
    gcol = [0, 1, 2, 4, 5]
    res = {}
    ylim = {}
    data = {}
    for ci, (model, bench) in enumerate(COLS):
        lim = 0.0
        for oi, (opt, _, _) in enumerate(OPTS):
            r, d, ns, nl = delta(model, bench, opt)
            data[(ci, oi)] = (r, d)
            pos = np.clip(d, 0, None).sum(1)
            neg = np.clip(d, None, 0).sum(1)
            lim = max(lim, pos.max(), -neg.min())
        ylim[ci] = lim * 1.12 / 1000.0
    for ci, (model, bench) in enumerate(COLS):
        for oi, (opt, oname, ocol) in enumerate(OPTS):
            ax = fig.add_subplot(gs[oi, gcol[ci]])
            r, d = data[(ci, oi)]
            x = np.arange(1, r + 1)
            pos0 = np.zeros(r)
            neg0 = np.zeros(r)
            for j, g in enumerate(["prompt", "skill", "tools", "messages"]):
                v = d[:, j] / 1000.0
                p, q = np.clip(v, 0, None), np.clip(v, None, 0)
                ax.fill_between(x, pos0, pos0 + p, step="mid", color=COLOR[g], lw=0, alpha=0.95)
                ax.fill_between(x, neg0, neg0 + q, step="mid", color=COLOR[g], lw=0, alpha=0.95)
                pos0 += p
                neg0 += q
            tot = d.sum(1) / 1000.0
            ax.step(x, tot, where="mid", color=INK, lw=0.9)
            ax.axhline(0, color=INK, lw=0.5)
            ax.set_xlim(0.5, r + 0.5)
            ax.set_ylim(-ylim[ci], ylim[ci])
            ax.set_facecolor("white")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(axis="y", color=GRID, lw=0.4)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=6.5, pad=1.5)
            if oi < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("turn", fontsize=7, labelpad=1)
            if ci in (0, 3):
                ax.set_ylabel(oname, fontsize=8, fontweight="bold", color=ocol, labelpad=2)
            if oi == 0:
                ax.set_title(BENCH_NAME[bench], fontsize=8, fontweight="bold", loc="left", pad=3)
            res["%s|%s|%s" % (model, bench, opt)] = dict(
                last_turn=int(r), mean_delta_k={g: round(float(d[:, j].mean() / 1000), 3) for j, g in enumerate(STACK)},
                mean_total_delta_k=round(float(d.sum(1).mean() / 1000), 3), total_delta_k_last_turn=round(float(d.sum(1)[-1] / 1000), 3))
    fig.text(0.095 + 0.0, 0.935, "DeepSeek-V4.1-Flash", fontsize=8.5, fontweight="bold", ha="left", color=INK)
    fig.text(0.74, 0.935, "GPT-6-Luna", fontsize=8.5, fontweight="bold", ha="left", color=INK)
    fig.text(0.012, 0.5, "change in tokens read by one turn, thousands (learned skill minus seed)", rotation=90, va="center", fontsize=7, color=MUTED)
    handles = [Patch(color=COLOR[g], label=LABEL[g]) for g in STACK] + [Line2D([0], [0], color=INK, lw=0.9, label="total change")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 0.0), handlelength=1.4, columnspacing=1.4)
    save(fig, "bip_delta")
    json.dump(res, open(OUT / "bip_delta_numbers.json", "w"), indent=1)
    for k, v in res.items():
        print(k, v["last_turn"], v["mean_delta_k"], v["mean_total_delta_k"])


if __name__ == "__main__":
    main()
