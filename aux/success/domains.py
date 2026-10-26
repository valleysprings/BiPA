"""Test success per task domain of tau3-bench and AutomationBench, Table 1's DeepSeek-V4.1-Flash cells (Figure
mech_domains_wide).

Per domain, the share of the tasks with a verified outcome that a run solves, averaged over the three runs: Table 1's Q
restricted to the domain, so the all column is Table 1's Q. The standard error is the standard deviation over tasks of the
per-task mean success over the square root of the task count, a lower bound of the uncertainty (it leaves out run noise).
The figure names the largest gap between a learned skill and the seed in any domain, and the bound, rounded up to 0.1 point,
within which every other gap lies."""
import json, math

import numpy as np

from src.analysis import settings as S
from src.utils import fig as F, logger

# domains in the order the figure draws them, with their tick labels
DOMAINS = {"tau3-bench": {"airline": "airline", "retail": "retail", "telecom": "telecom", "banking_knowledge": "banking"},
           "automationbench": {d: d if d != "hr" else "HR" for d in ("finance", "hr", "marketing", "operations", "sales", "support")}}
ROW = {**S.ARM_NAME, "native-seed": "Seed skill"}  # the record's row labels


def by_domain(bench, method):
    """{domain or "all": {"q", "se", "tasks"}} of one cell."""
    reps = list(S.cell_rows(S.DEEPSEEK, bench, method).values())
    member = lambda t, key: key == "all" or S.domain_of(bench, t) == key  # noqa: E731
    per_task = {}
    for rows in reps:
        for t, r in rows.items():
            if r["success"] is not None:
                per_task.setdefault(t, []).append(float(r["success"] is True))
    out = {}
    for key in [*DOMAINS[bench], "all"]:
        rates = [np.mean([r["success"] is True for t, r in rows.items() if r["success"] is not None and member(t, key)]) for rows in reps]
        means = [np.mean(v) for t, v in per_task.items() if member(t, key)]
        out[key] = {"q": float(100 * np.mean(rates)), "se": float(100 * np.std(means, ddof=1) / math.sqrt(len(means))),
                    "tasks": sum(member(t, key) for t in reps[0])}
    return out


def gaps(res):
    """[(bench, domain, method, learned - seed)], largest first, and the bound of all but the largest."""
    out = sorted(((b, k, m, res[b][m][k]["q"] - res[b]["native-seed"][k]["q"]) for b in DOMAINS for k in DOMAINS[b] for m in S.OPTIMIZERS),
                 key=lambda g: -abs(g[3]))
    return out, math.ceil(10 * abs(out[1][3])) / 10


def numbers(res):
    out = []
    for bench, doms in DOMAINS.items():
        cells, keys = res[bench], [*doms, "all"]
        out += [f"### {bench}", "", logger.md_table(["Row"] + [f"{k} (n={cells['native-seed'][k]['tasks']})" for k in keys],
                                                    [[ROW[m]] + [f"{c[k]['q']:.1f}" for k in keys] for m, c in cells.items()]), "",
                "Standard error over tasks (points):", "",
                logger.md_table(["Row", *keys], [[ROW[m]] + [f"{c[k]['se']:.1f}" for k in keys] for m, c in cells.items()]), ""]
    g, bound = gaps(res)
    out += [f"Every learned skill lies within {bound:.1f} points of the seed in every domain except "
            + "; ".join(f"{S.ARM_NAME[m]} on {S.BENCH_SHORT[b]} {k} ({x:+.1f})" for b, k, m, x in g if abs(x) > bound) + "."]
    return "\n".join(out)


def title(bench, g, bound):
    out = [x for x in g if x[0] == bench and abs(x[3]) > bound]
    if not out:
        return f"{F.BENCH_LABEL[bench]}: all learned skills within {bound:.1f} points of the seed"
    head = ", ".join(f"{S.ARM_NAME[m]} on {DOMAINS[bench][k]}" for _, k, m, _ in out)
    side = "; ".join(f"{abs(x):.1f} points {'below' if x < 0 else 'above'} the seed" for *_, x in out)
    return f"{F.BENCH_LABEL[bench]}: only {head} lies further, {side}"


def draw(res):
    g, bound = gaps(res)
    plt = F.pyplot(7, title=7.5)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.7), gridspec_kw=dict(width_ratios=[4, 6], wspace=0.07), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.13, top=0.755)
    offs = [(i - 2) * 0.13 for i in range(len(S.ARMS))]
    face = {m: "white" if m == "native-noseed" else F.ARM[m] for m in S.ARMS}
    edge = {m: F.MID if m == "native-noseed" else F.ARM[m] for m in S.ARMS}
    for ax, (bench, doms) in zip(axes, DOMAINS.items()):
        for j, k in enumerate(doms):
            for m, o in zip(S.ARMS, offs):
                ax.scatter([j + o], [res[bench][m][k]["q"]], s=13, marker=F.MARKER[m], facecolor=face[m], edgecolor=edge[m], lw=0.8, zorder=3)
        for b, k, m, x in g:
            if b == bench and abs(x) > bound:
                ax.annotate(f"{x:+.1f}".replace("-", "−"), (list(doms).index(k) + offs[S.ARMS.index(m)], res[bench][m][k]["q"]),
                            xytext=(0, -5), textcoords="offset points", ha="center", va="top", fontsize=6.8, color=F.ARM[m])
        ax.set_xticks(range(len(doms)))
        ax.set_xticklabels(doms.values())
        ax.set_xlim(-0.5, len(doms) - 0.5)
        ax.set_ylim(0, 106)
        ax.set_yticks([0, 50, 100])
        ax.set_title(title(bench, g, bound), loc="left", pad=4)
        ax.grid(axis="y", color=F.GRID, lw=0.4)
        ax.set_axisbelow(True)
        F.despine(ax)
        ax.tick_params(axis="x", length=0, pad=2)
    axes[0].set_ylabel("test success (%)", labelpad=2)
    axes[1].tick_params(axis="y", length=0)
    axes[1].spines["left"].set_visible(False)
    handles = [plt.Line2D([], [], ls="", marker=F.MARKER[m], mfc=face[m], mec=edge[m], ms=4.2, mew=0.8) for m in S.ARMS]
    fig.legend(handles, [S.ARM_NAME[m] for m in S.ARMS], loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=len(S.ARMS),
               frameon=False, handlelength=1.0, columnspacing=1.4, handletextpad=0.3)
    F.save_figure(fig, "domains_wide", width=F.TEXT_W)


def main():
    res = {bench: {m: by_domain(bench, m) for m in S.ARMS} for bench in DOMAINS}
    (S.cache("success") / "domains.json").write_text(json.dumps(res, indent=1))
    logger.record(__spec__.name, "domains", numbers(res), S.runs_text([(S.DEEPSEEK, b, m) for b in DOMAINS for m in S.ARMS]))
    draw(res)


if __name__ == "__main__":
    main()
