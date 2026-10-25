"""Within-task turn slopes of input tokens, output tokens and calls (DeepSeek test episodes), and Figure 3.

compute     -> <cache>/analysis/main/slopes.json
draw-input  -> fig/mech/mech_slopes_input2 (the input-token row)
"""
import json
import sys

import numpy as np

from src.analysis.episodes import BENCH, episodes, fe_reg, metrics, output_totals
from src.utils import dataloader as C
from src.utils.fig import save_figure

MODEL = "deepseek-flash"
QUANTITIES = (("input", lambda m, o: m["X"]), ("output", lambda m, o: o), ("calls", lambda m, o: m["N"]))


def bench_episodes(bench):
    """{(skill, task): (metrics, output tokens)}: the seed's first test run and each returned optimizer skill; run names."""
    seed_run = C.current_run("native-seed", MODEL, bench)
    sides = [("seed", seed_run, C.SEED_TEXT)]
    sides += [(disp, run, text) for disp, (run, text) in C.returned_skills(MODEL, bench, C.OPTIMIZERS).items()]
    eps, runs = {}, {}
    for disp, run, text in sides:
        path = C.eval_file(run, "test", C.vid(text))
        if path is None:
            continue
        runs[disp] = run.name
        outs = output_totals(path)
        for task, e in episodes(path).items():
            if m := metrics(e):
                eps[(disp, task)] = (m, outs.get(task))
    return eps, runs


def stats(values):
    v = np.array(values, float)
    return dict(n=len(v), min=float(v.min()), p5=float(np.percentile(v, 5)), p50=float(np.percentile(v, 50)),
                p95=float(np.percentile(v, 95)), max=float(v.max()), mean=float(v.mean()))


def compute():
    reg, pts, runs, elasticity = {}, {}, {}, {}
    for bench, _, _ in BENCH:
        eps, runs[bench] = bench_episodes(bench)
        reg[bench], pts[bench] = {}, {}
        for name, pick in QUANTITIES:
            items = [(task, m["R"], v) for (_, task), (m, o) in eps.items() if (v := pick(m, o)) and v > 0]
            reg[bench][name], (xs, ys) = fe_reg(items)
            pts[bench][name] = [xs.tolist(), ys.tolist()]
        elasticity[bench] = [1 + (1 - m["sF"]) * m["R"] / (m["R"] - 1) for m, _ in eps.values()]
    elasticity["all"] = [e for b, _, _ in BENCH for e in elasticity[b]]
    C.write_json("slopes", dict(reg=reg, pts=pts, runs=runs, elasticity={k: stats(v) for k, v in elasticity.items()}))
    for bench, r in reg.items():
        print(bench, {name: (round(v["slope"], 3), v["n"]) for name, v in r.items()})


def draw_input():
    """Figure 3: input tokens against turns, both over the task's mean, every episode, with the fitted and slope-1 lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = 7.0
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                         "font.size": fs, "axes.labelsize": fs, "axes.titlesize": fs, "xtick.labelsize": fs, "ytick.labelsize": fs,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2, "ytick.major.size": 2})
    benches = [("tau3-bench", "τ³-bench", "#0072B2"), ("automationbench", "AutoBench", "#D55E00"), ("terminal-bench-2-1", "TB2", "#009E73")]
    c = json.loads((C.CACHE / "slopes.json").read_text())
    reg = {b: c["reg"][b]["input"] for b, _, _ in benches}
    pts = {b: (np.array(c["pts"][b]["input"][0]), np.array(c["pts"][b]["input"][1])) for b, _, _ in benches}
    all_x = np.concatenate([p[0] for p in pts.values()])
    all_y = np.concatenate([p[1] for p in pts.values()])
    xlo, xhi = np.exp(all_x.min()) / 1.12, np.exp(all_x.max()) * 1.12
    lo, hi = np.exp(all_y.min()) / 1.25, np.exp(all_y.max()) * 1.25
    xx = np.array([xlo, xhi])
    fig, axs = plt.subplots(1, 3, figsize=(0.86 * 219.08 / 72, 1.75), sharey=True)
    fig.subplots_adjust(left=0.155, right=0.975, bottom=0.2, top=0.89, wspace=0.3)
    for ax, (b, name, col) in zip(axs, benches):
        ax.scatter(np.exp(pts[b][0]), np.exp(pts[b][1]), s=1.4, color=col, alpha=0.35, lw=0)
        ax.plot(xx, xx, color="k", ls="--", lw=0.6)
        ax.plot(xx, xx ** reg[b]["slope"], color=col, lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(lo, hi)
        ax.set_xticks([0.3, 1, 3])
        ax.set_xticklabels(["0.3", "1", "3"])
        ax.set_yticks([0.1, 0.3, 1, 3])
        ax.minorticks_off()
        ax.tick_params(pad=1.5)
        ax.set_title(name, fontweight="bold", pad=2)
        ax.text(0.05, 0.97, f"slope {reg[b]['slope']:.2f}", transform=ax.transAxes, ha="left", va="top", color=col)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_yticklabels(["0.1", "0.3", "1", "3"])
    axs[0].set_ylabel("input tokens / task mean", labelpad=1.5)
    axs[1].set_xlabel("turns / task mean", labelpad=1.5)
    save_figure(fig, "slopes_input2")
    print({name: round(reg[b]["slope"], 3) for b, name, _ in benches})


if __name__ == "__main__":
    {"compute": compute, "draw-input": draw_input}[sys.argv[1]]()
