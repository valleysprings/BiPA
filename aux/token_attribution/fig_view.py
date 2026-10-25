"""Figure bip_view: the episode graph in aggregate, turn by context component, for every benchmark and both models.

For each model and benchmark, every recorded test episode is a graph between its turns and its context components.  Here the graph is averaged over episodes:
at turn r, the mean number of tokens the turn reads from each component, over the episodes that are still running at turn r.  Components are grouped as
  rest of the prompt (system prompt, tool schemas, task) | skill text | tool outputs (all tools) | conversation messages.
Stacked areas: the seed.  Dashed line: the learned skills of the three optimizers, pooled (total tokens read at that turn).  Strip below: share of episodes still running.
Turns are shown up to the last turn at which at least 30% of the seed's episodes are still running (beyond that the mean is over a few long tasks); in the record the cut is 30%.
Luna TB2: all three optimizers returned the seed, so there is no learned line.

Data: bipdata.pkl (episode graphs of the paper's own test runs; DeepSeek from the parse cache, Luna from attribution_luna's graph cache).  No model calls.
Run:  python -m aux.token_attribution.fig_view
"""
import json
import pickle

import numpy as np

from aux.token_attribution.bipcommon import *  # noqa: F401,F403

D = None
GROUPS = ["prompt", "skill", "tools", "messages"]          # column order of the per-turn matrices
STACK = ["prompt", "skill", "tools", "messages"]            # bottom to top
LABEL = dict(prompt="system prompt, tools, task", skill="skill text", tools="tool outputs", messages="agent and user messages")
COLOR = dict(prompt="#BBBBBB", skill="#D55E00", tools="#009E73", messages="#56B4E9")
BENCHES = [("tau3-bench", "tau3"), ("automationbench", "AutoB"), ("terminal-bench-2-1", "TB2")]
KEEP = 0.30   # last turn shown: the seed still has >= 30% of its episodes running
OPT = ["gepa", "skillopt", "skilladam"]


def profile(eps, rmax):
    """Mean tokens per group at turn r (r = 1..rmax) over episodes with R >= r, the number of such episodes, and the pooled prefix share of all input tokens."""
    S = np.zeros((rmax, 4))
    n = np.zeros(rmax)
    for e in eps:
        k = min(e["R"], rmax)
        S[:k] += e["M"][:k]
        n[:k] += 1
    mean = S / np.maximum(n, 1)[:, None]
    tot = sum(e["M"].sum() for e in eps)
    pref = sum(e["M"][:, 0].sum() + e["M"][:, 1].sum() for e in eps)
    return mean, n, pref / tot


def median_prefix(eps):
    """Median over episodes (R >= 2) of the prompt-and-skill share of the episode's input tokens and of the turn elasticity 1 + (1 - share) R / (R - 1)."""
    sf, er = [], []
    for e in eps:
        if e["R"] < 2:
            continue
        M = e["M"]
        s = e["R"] * (M[:, 0] + M[:, 1]).mean() / M.sum()
        if s >= 1:
            continue
        sf.append(s)
        er.append(1 + (1 - s) * e["R"] / (e["R"] - 1))
    return float(np.median(sf)), float(np.median(er))


def collect(model, bench):
    seed = D[(model, bench, "seed")]
    learned = [e for m in OPT for e in D.get((model, bench, m), [])]
    return seed, learned


def load():
    """Read the inputs this figure draws from."""
    global D
    D = pickle.loads((OUT / "bipdata.pkl").read_bytes())["episodes"]


def main():
    load()
    plt = setup_mpl()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(7.1, 4.5))
    left, right = 0.085, 0.99
    colw = (right - left) / 3
    pw = colw - 0.062
    models = ["deepseek-flash", "gpt-6-luna"]
    # row geometry (figure fractions): main axes + survival strip per model
    geom = {"deepseek-flash": (0.615, 0.265, 0.575, 0.045), "gpt-6-luna": (0.135, 0.265, 0.095, 0.045)}
    summary = {}
    # one y range per benchmark, shared by both models
    ymax = {}
    for bench, short in BENCHES:
        top = 0
        for model in models:
            seed, learned = collect(model, bench)
            Rs = np.array([e["R"] for e in seed])
            alive = np.array([(Rs >= r).mean() for r in range(1, Rs.max() + 1)])
            rmax = int(np.max(np.nonzero(alive >= KEEP)[0]) + 1)
            for eps in (seed, learned):
                if eps:
                    top = max(top, profile(eps, rmax)[0].sum(1).max() / 1000)
        ymax[short] = float(np.ceil(top * 1.04 / 10) * 10)
    for mi, model in enumerate(models):
        main_y, main_h, strip_y, strip_h = None, None, None, None
        top = 0.85 if mi == 0 else 0.385
        main_h, strip_h = 0.26, 0.05
        main_y = top - main_h
        strip_y = main_y - 0.012 - strip_h
        for bi, (bench, short) in enumerate(BENCHES):
            x0 = left + bi * colw + 0.03
            seed, learned = collect(model, bench)
            Rs = np.array([e["R"] for e in seed])
            alive = np.array([(Rs >= r).mean() for r in range(1, Rs.max() + 1)])
            rmax = int(np.max(np.nonzero(alive >= KEEP)[0]) + 1)
            ms, ns, pref_s = profile(seed, rmax)
            ax = fig.add_axes([x0, main_y, pw, main_h])
            ax2 = fig.add_axes([x0, strip_y, pw, strip_h], sharex=ax)
            xs = np.arange(1, rmax + 1)
            bottom = np.zeros(rmax)
            for g in STACK:
                v = ms[:, GROUPS.index(g)] / 1000
                ax.fill_between(xs, bottom, bottom + v, color=COLOR[g], lw=0, step="mid", zorder=2)
                bottom = bottom + v
            ax.step(xs, bottom, where="mid", color=INK, lw=0.7, zorder=4)
            info = dict(model=model, bench=bench, n_seed=len(seed), n_tasks_seed=len({e["task"] for e in seed}), rmax=rmax, prefix_share_seed=float(pref_s),
                        mean_R_seed=float(Rs.mean()), n_at_rmax_seed=int(ns[-1]))
            ax2.step(xs, [(Rs >= r).mean() * 100 for r in xs], where="mid", color=INK, lw=0.8)
            if False:  # per-turn means over the episodes still running are not comparable between the seed and a learned skill (different tasks are still running)
                Rl = np.array([e["R"] for e in learned])
                ml, nl, pref_l = profile(learned, rmax)
                ax.step(xs, ml.sum(1) / 1000, where="mid", color=INK, lw=1.0, ls=(0, (3, 2)), zorder=5)
                ax2.step(xs, [(Rl >= r).mean() * 100 for r in xs], where="mid", color=INK, lw=0.8, ls=(0, (3, 2)))
                info.update(n_learned=len(learned), mean_R_learned=float(Rl.mean()), prefix_share_learned=float(pref_l),
                            skill_tokens_seed=float(ms[:, 1].mean()), skill_tokens_learned=float(ml[:, 1].mean()),
                            top_seed_last=float(ms[-1].sum()), top_learned_last=float(ml[-1].sum()))
            summary[f"{model}|{short}"] = info
            ax.set_xlim(0.5, rmax + 0.5)
            ax.set_ylim(0, ymax[short])
            ax2.set_ylim(0, 105)
            ax2.set_yticks([0, 100])
            ax2.set_yticklabels(["0", "100"], fontsize=5.6)
            ax.tick_params(axis="x", labelbottom=False, length=0)
            ax.tick_params(axis="y", labelsize=6.8)
            for a in (ax, ax2):
                for sp in ("top", "right"):
                    a.spines[sp].set_visible(False)
                a.grid(axis="y", color=GRID, lw=0.5, zorder=0)
                a.set_axisbelow(True)
            ax2.tick_params(axis="x", labelsize=6.8)
            ax2.set_xlabel("turn", fontsize=7, labelpad=1)
            ax2.text(0.01, 0.08, "% of episodes still running", transform=ax2.transAxes, fontsize=5.6, color=MUTED, ha="left", va="bottom")
            if bi == 0:
                ax.set_ylabel("tokens read by one turn, thousands", fontsize=6.4)
                ax.yaxis.set_label_coords(-0.2, 0.5)
            msf, mer = median_prefix(seed)
            info.update(median_prefix_share_seed=msf, median_turn_elasticity_seed=mer)
            ax.text(0.0, 1.235, BENCH_NAME[short], transform=ax.transAxes, fontsize=7.8, fontweight="bold", ha="left", va="bottom", color=INK)
            ax.text(0.0, 1.11, "prefix share of input:", transform=ax.transAxes, fontsize=6.0, ha="left", va="bottom", color=MUTED)
            ax.text(0.0, 1.015, f"median episode {100 * msf:.0f}%, all tokens {100 * pref_s:.0f}%", transform=ax.transAxes, fontsize=6.0, ha="left", va="bottom", color=MUTED)
            if False:
                ax.text(1.0, 1.235, "seed only", transform=ax.transAxes, fontsize=6.3, ha="right", va="bottom", color=MUTED)
        fig.text(0.004, main_y + main_h / 2, MODEL_NAME[model], rotation=90, fontsize=7.6, fontweight="bold", ha="left", va="center", color=INK)
    handles = [Patch(fc=COLOR[g], label=LABEL[g]) for g in STACK] + [
        Line2D([0], [0], color=INK, lw=0.8, label="total")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.535, 0.995), ncol=5, frameon=False, fontsize=6.8, handlelength=1.1, handleheight=0.8, columnspacing=1.0, handletextpad=0.4)
    save(fig, "bip_view")
    plt.close(fig)
    (OUT / "bip_view_numbers.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})


if __name__ == "__main__":
    main()
