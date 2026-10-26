"""Within-task turn slopes of input tokens, output tokens and tool calls (DeepSeek test episodes), the turn elasticity
and the prefix share of the same episodes, and the input-token figure.

Episodes: the first test run of the seed and of each learned skill, one episode per skill and task. Slope: the slope of
log y on log turns R with task fixed effects (src.analysis.episodes.fe_reg), with its standard error. Turn elasticity e_R
of each episode, from its turns and its history share s_H (src.analysis.episodes.metrics). Prefix share s_F = R F / X, the
part of an episode's input tokens that the fixed prefix (system prompt, tool schemas, task, skill text) makes up: its
median over the episodes, its token-weighted mean (sum of R F over sum of X), and the share of the input tokens that the
longest tenth of the episodes (by input tokens) reads.

compute     -> <cache>/main/slopes.json and the record mech_slopes
draw-input  -> fig/mech/mech_slopes_input2"""
import json, sys

import numpy as np

from src.analysis import settings as S
from src.analysis.attribution import graphs
from src.analysis.episodes import fe_reg, metrics
from src.utils import dataloader as C
from src.utils import fig as style
from src.utils.logger import md_table, record

MODEL = S.DEEPSEEK
QUANTITIES = (("input", lambda m, o: m["X"]), ("output", lambda m, o: o), ("calls", lambda m, o: m["N"]))


def output_tokens(episode):
    """Output tokens over the distinct requests of a recorded episode that the provider answered."""
    seen = {r["request_id"]: r for r in episode.get("requests", []) if r.get("input_tokens") is not None}
    return float(sum(r.get("output_tokens") or 0 for r in seen.values()))


def bench_episodes(bench):
    """{(skill, task): (metrics, output tokens)} of the first test run of the seed and of each learned skill; their runs."""
    eps, runs = {}, {}
    for method in ["native-seed"] + [m for _, _, m in S.learned([MODEL], [bench])]:
        files = {0: S.cell_files(MODEL, bench, method)[0]}
        name = "seed" if method == "native-seed" else S.ARM_NAME[method]
        runs[name] = S.run(MODEL, bench, method).name
        outs = {t: output_tokens(e) for t, e in S.episodes(files)[0].items()}
        for task, e in graphs(files)[0].items():
            if m := metrics(e):
                eps[(name, task)] = (m, outs.get(task))
    return eps, runs


def stats(values):
    v = np.array(values, float)
    return dict(n=len(v), min=float(v.min()), p5=float(np.percentile(v, 5)), p50=float(np.percentile(v, 50)),
                p95=float(np.percentile(v, 95)), max=float(v.max()), mean=float(v.mean()))


def prefix(ms):
    X = np.sort([m["X"] for m in ms])[::-1]
    return dict(n=len(ms), median=float(np.median([m["sF"] for m in ms])),
                weighted=sum(m["R"] * m["F"] for m in ms) / float(X.sum()), longest_tenth=float(X[:max(1, len(X) // 10)].sum() / X.sum()))


def compute():
    reg, pts, runs, elasticity, shares = {}, {}, {}, {}, {}
    for bench in C.BENCHES:
        eps, runs[bench] = bench_episodes(bench)
        reg[bench], pts[bench] = {}, {}
        for name, pick in QUANTITIES:
            items = [(task, m["R"], v) for (_, task), (m, o) in eps.items() if (v := pick(m, o)) and v > 0]
            reg[bench][name], (xs, ys) = fe_reg(items)
            pts[bench][name] = [xs.tolist(), ys.tolist()]
        elasticity[bench] = [1 + (1 - m["sF"]) * m["R"] / (m["R"] - 1) for m, _ in eps.values()]
        shares[bench] = prefix([m for m, _ in eps.values()])
    elasticity["all"] = [e for b in C.BENCHES for e in elasticity[b]]
    elasticity = {k: stats(v) for k, v in elasticity.items()}
    C.write_json("slopes", dict(reg=reg, pts=pts, runs=runs, elasticity=elasticity))
    head = ["Benchmark", "quantity", "slope", "SE", "episodes", "tasks", "R^2 (within)"]
    rows = [[style.BENCH_LABEL[b], q, f"{v['slope']:.3f}", f"{v['se']:.3f}", v["n"], v["n_tasks"], f"{v['r2']:.2f}"]
            for b in C.BENCHES for q, v in reg[b].items()]
    stat_rows = [[style.BENCH_LABEL.get(b, b)] + [f"{e[k]:.3f}" if k != "n" else e[k] for k in ("n", "min", "p5", "p50", "p95", "max", "mean")]
                 for b, e in elasticity.items()]
    share_rows = [[style.BENCH_LABEL[b], s["n"], f"{s['median']:.4f}", f"{s['weighted']:.4f}", f"{s['longest_tenth']:.4f}"] for b, s in shares.items()]
    numbers = "\n\n".join([md_table(head, rows), "Turn elasticity e_R per episode", md_table(["Benchmark", "episodes", "min", "p5", "p50", "p95", "max", "mean"], stat_rows),
                           "Prefix share s_F of the same episodes",
                           md_table(["Benchmark", "episodes", "median s_F", "token-weighted s_F", "input-token share of the longest tenth"], share_rows)])
    cells = [(MODEL, b, m) for b in C.BENCHES for m in ["native-seed"] + [m for _, _, m in S.learned([MODEL], [b])]]
    record(__spec__.name, "slopes", numbers, S.runs_text(cells))
    for bench, r in reg.items():
        print(bench, {name: (round(v["slope"], 3), v["n"]) for name, v in r.items()})


def draw_input():
    """The input-token figure: input tokens against turns, both over the task's mean, every episode, with the fitted and slope-1 lines."""
    plt = style.pyplot(7.0, title=7)
    c = json.loads((C.CACHE / "slopes.json").read_text())
    reg = {b: c["reg"][b]["input"] for b in C.BENCHES}
    pts = {b: (np.array(c["pts"][b]["input"][0]), np.array(c["pts"][b]["input"][1])) for b in C.BENCHES}
    all_x, all_y = (np.concatenate([p[i] for p in pts.values()]) for i in (0, 1))
    xlo, xhi = np.exp(all_x.min()) / 1.12, np.exp(all_x.max()) * 1.12
    lo, hi = np.exp(all_y.min()) / 1.25, np.exp(all_y.max()) * 1.25
    xx = np.array([xlo, xhi])
    fig, axs = plt.subplots(1, 3, figsize=(0.86 * style.COL_W, 1.75), sharey=True)
    fig.subplots_adjust(left=0.155, right=0.975, bottom=0.2, top=0.89, wspace=0.3)
    for ax, b in zip(axs, C.BENCHES):
        col = style.BENCH_COLOR[b]
        ax.scatter(np.exp(pts[b][0]), np.exp(pts[b][1]), s=1.4, color=col, alpha=0.35, lw=0)
        ax.plot(xx, xx, color=style.INK, ls="--", lw=0.6)
        ax.plot(xx, xx ** reg[b]["slope"], color=col, lw=1.2)
        ax.set(xscale="log", yscale="log", xlim=(xlo, xhi), ylim=(lo, hi), xticks=[0.3, 1, 3], xticklabels=["0.3", "1", "3"], yticks=[0.1, 0.3, 1, 3])
        ax.minorticks_off()
        ax.tick_params(pad=1.5)
        ax.set_title(style.BENCH_LABEL[b], fontweight="bold", pad=2)
        ax.text(0.05, 0.97, f"slope {reg[b]['slope']:.2f}", transform=ax.transAxes, ha="left", va="top", color=col)
        style.despine(ax)
    axs[0].set_yticklabels(["0.1", "0.3", "1", "3"])
    axs[0].set_ylabel("input tokens / task mean", labelpad=1.5)
    axs[1].set_xlabel("turns / task mean", labelpad=1.5)
    style.save_figure(fig, "slopes_input2")


if __name__ == "__main__":
    {"compute": compute, "draw-input": draw_input}[sys.argv[1]]()
