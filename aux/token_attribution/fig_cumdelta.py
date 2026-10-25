"""Figure bip_cumdelta: what each learned skill adds to the input tokens read so far, turn by turn and component by component.

Per task, the cumulative tokens read from each component up to turn r (an episode that has ended keeps its total), averaged over the task's runs; then averaged over the tasks that have
runs under both the seed and the optimizer.  delta(r, g) = learned minus seed.  Because every task counts at every turn, the curve does not depend on how long the episodes run, and its right
end is the extra input tokens per task.  Turns shown: up to the 95th percentile of the episode length of the seed and the optimizer.
Rows: GEPA, SkillOpt, SkillAdam.  Columns: DeepSeek tau3 / AutoBench / TB2, Luna tau3 / AutoBench (Luna TB2: every optimizer returned the seed).

Run:  python -m aux.token_attribution.fig_cumdelta
"""
import json
import pickle
from collections import defaultdict

import numpy as np

from aux.token_attribution.bipcommon import *  # noqa: F401,F403

D = None
STACK = ["prompt", "skill", "tools", "messages"]
LABEL = dict(prompt="system prompt, tools, task", skill="skill text", tools="tool outputs", messages="agent and user messages")
COLOR = dict(prompt="#BBBBBB", skill="#D55E00", tools="#009E73", messages="#56B4E9")
OPTS = [("gepa", "GEPA", "#2A9D4B"), ("skillopt", "SkillOpt", "#1F6FB5"), ("skilladam", "SkillAdam", "#E08A1E")]
COLS = [("deepseek-flash", "tau3-bench"), ("deepseek-flash", "automationbench"), ("deepseek-flash", "terminal-bench-2-1"),
        ("gpt-6-luna", "tau3-bench"), ("gpt-6-luna", "automationbench")]


def cum_by_task(eps, rmax):
    """{task: (rmax x 4) mean over the task's runs of the cumulative tokens by component up to turn r, carried forward after the episode ends}."""
    per = defaultdict(list)
    for e in eps:
        C = np.cumsum(e["M"][:rmax], axis=0)
        if C.shape[0] < rmax:
            C = np.vstack([C, np.repeat(C[-1:], rmax - C.shape[0], axis=0)])
        per[e["task"]].append(C)
    return {t: np.mean(v, axis=0) for t, v in per.items()}


def delta(model, bench, opt):
    seed, L = D[(model, bench, "seed")], D[(model, bench, opt)]
    rmax = int(np.ceil(np.percentile([e["R"] for e in seed] + [e["R"] for e in L], 95)))
    cs, cl = cum_by_task(seed, rmax), cum_by_task(L, rmax)
    tasks = sorted(set(cs) & set(cl))
    d = np.mean([cl[t] - cs[t] for t in tasks], axis=0)
    # full totals (not cut at rmax), same pairing
    fs, fl = cum_by_task(seed, 10 ** 3), cum_by_task(L, 10 ** 3)
    tot = np.mean([fl[t][-1] - fs[t][-1] for t in tasks], axis=0)
    seed_tot = np.mean([fs[t][-1] for t in tasks], axis=0)
    return rmax, d, len(tasks), tot, seed_tot


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
    data, ylim, res = {}, {}, {}
    for ci, (model, bench) in enumerate(COLS):
        lim = 0.0
        for oi, (opt, _, _) in enumerate(OPTS):
            r, d, n, tot, seed_tot = delta(model, bench, opt)
            data[(ci, oi)] = (r, d, n, tot, seed_tot)
            lim = max(lim, np.clip(d, 0, None).sum(1).max(), -np.clip(d, None, 0).sum(1).min())
        ylim[ci] = lim * 1.12 / 1000.0
    for ci, (model, bench) in enumerate(COLS):
        for oi, (opt, oname, ocol) in enumerate(OPTS):
            ax = fig.add_subplot(gs[oi, gcol[ci]])
            r, d, n, tot, seed_tot = data[(ci, oi)]
            x = np.arange(1, r + 1)
            pos0 = np.zeros(r)
            neg0 = np.zeros(r)
            for j, g in enumerate(STACK):
                v = d[:, j] / 1000.0
                p, q = np.clip(v, 0, None), np.clip(v, None, 0)
                ax.fill_between(x, pos0, pos0 + p, step="mid", color=COLOR[g], lw=0, alpha=0.95)
                ax.fill_between(x, neg0, neg0 + q, step="mid", color=COLOR[g], lw=0, alpha=0.95)
                pos0 += p
                neg0 += q
            ax.step(x, d.sum(1) / 1000.0, where="mid", color=INK, lw=0.9)
            ax.axhline(0, color=INK, lw=0.5)
            ax.set_xlim(0.5, r + 0.5)
            ax.set_ylim(-ylim[ci] * 0.35, ylim[ci]) if False else ax.set_ylim(-ylim[ci], ylim[ci])
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
                turns_shown=int(r), tasks=int(n),
                extra_tokens_per_task_k_by_component={g: round(float(tot[j] / 1000), 2) for j, g in enumerate(STACK)},
                extra_tokens_per_task_k_total=round(float(tot.sum() / 1000), 2), seed_tokens_per_task_k=round(float(seed_tot.sum() / 1000), 1),
                at_last_shown_turn_k_total=round(float(d.sum(1)[-1] / 1000), 2))
    fig.text(0.095, 0.935, "DeepSeek-V4.1-Flash", fontsize=8.5, fontweight="bold", ha="left", color=INK)
    fig.text(0.74, 0.935, "GPT-6-Luna", fontsize=8.5, fontweight="bold", ha="left", color=INK)
    fig.text(0.012, 0.5, "extra input tokens read so far, thousands (learned skill minus seed, per task)", rotation=90, va="center", fontsize=7, color=MUTED)
    handles = [Patch(color=COLOR[g], label=LABEL[g]) for g in STACK] + [Line2D([0], [0], color=INK, lw=0.9, label="total")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 0.0), handlelength=1.4, columnspacing=1.4)
    save(fig, "bip_cumdelta")
    json.dump(res, open(OUT / "bip_cumdelta_numbers.json", "w"), indent=1)
    for k, v in res.items():
        print(k, v["turns_shown"], v["tasks"], v["extra_tokens_per_task_k_by_component"], v["extra_tokens_per_task_k_total"], "seed", v["seed_tokens_per_task_k"])


if __name__ == "__main__":
    main()
