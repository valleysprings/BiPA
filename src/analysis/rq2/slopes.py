"""mech_slopes: the Appendix-C within-task slope figure of paper 1 from baseline episodes only.
Data (needs .venv): per benchmark the seed's run-0 test episode and one episode per task of each baseline skill (GEPA, SkillOpt, SkillAdam; common.returned_skills restricted to
the three baselines), exactly the pooling of mechanism/fig1.py compute, with the within-task fixed-effects regression fig1.fe_reg of log input tokens / log output tokens / log calls on log requests.
Also the per-episode turn elasticity e_R = 1 + s_H * R / (R - 1) (s_H = 1 - s_F, the history share of the input tokens, R = requests of the episode, both from slopes_checks.metrics).
  .venv/bin/python -m src.analysis.rq2.slopes compute      -> assets/cache/analysis/rq/slopes.json
  .venv/bin/python -m src.analysis.rq2.slopes draw   -> paper/assets/fig/mech/mech_slopes.pdf/.svg and data/mech/mech_slopes.md
  .venv/bin/python -m src.analysis.rq2.slopes draw-input   -> fig/mech/mech_slopes_input.pdf/.svg and data/mech/mech_slopes_input.md
    (the first row of mech_slopes as a single-column figure for the methodology section; merged from slopes_input_fig.py)"""
import collections
import json
import sys
import numpy as np
from src.utils import fig, logger, dataloader
from src.utils.fig import save_figure
MODEL = "deepseek-flash"
EXPECTED = {"tau3-bench": dict(input=1.349, output=1.025, calls=0.840, n=300),
            "automationbench": dict(input=1.556, output=0.881, calls=0.853, n=480),
            "terminal-bench-2-1": dict(input=1.584, output=0.498, calls=0.936, n=71)}


def compute():
    """Pool seed and baseline episodes per benchmark, regress, and compute the e_R distribution; write mech/slopes.json."""
    from src.utils import dataloader as C
    from src.analysis.appendix.slopes import metrics, episodes
    from src.analysis.appendix.appc import output_totals, fe_reg, BENCH
    reg, pts, raw, runs, elasticity = {}, {}, [], {}, {}
    all_e = []
    for bench, _, _ in BENCH:
        episodes_by_key = collections.OrderedDict()
        outputs_by_key = {}
        seed_run = C.current_run("native-seed", MODEL, bench)
        seed_file = C.eval_file(seed_run, "test", C.vid(C.SEED_TEXT))
        raw.append(str(seed_file))
        runs[bench] = {"seed": seed_run.name}
        seed_eps = episodes(seed_file)
        seed_out = output_totals(seed_file)
        for task, es in seed_eps.items():
            m = metrics(es)
            if m:
                episodes_by_key[("seed", task)] = m
                outputs_by_key[("seed", task)] = seed_out.get(task)
        for disp, (run, text) in C.returned_skills(MODEL, bench, dataloader.OPTIMIZERS).items():
            path = C.eval_file(run, "test", C.vid(text))
            if path is None:
                continue
            raw.append(str(path))
            runs[bench][disp] = run.name
            eps = episodes(path)
            outs = output_totals(path)
            for task, ec in eps.items():
                m = metrics(ec)
                if m:
                    episodes_by_key[(disp, task)] = m
                    outputs_by_key[(disp, task)] = outs.get(task)
        records = [(key[1], m, outputs_by_key[key]) for key, m in episodes_by_key.items()]
        reg[bench], pts[bench] = {}, {}
        for qn, pick in (("input", lambda m, o: m["X"]), ("output", lambda m, o: o), ("calls", lambda m, o: m["N"])):
            items = []
            for task, m, o in records:
                value = pick(m, o)
                if value and value > 0:
                    items.append((task, m["R"], value))
            r, p = fe_reg(items)
            reg[bench][qn] = r
            pts[bench][qn] = [p[0].tolist(), p[1].tolist()]
        # per-episode turn elasticity of the input tokens, e_R = 1 + s_H R / (R - 1)
        e_bench = []
        for key, m in episodes_by_key.items():
            s_h = 1 - m["sF"]
            e_bench.append(1 + s_h * m["R"] / (m["R"] - 1))
        elasticity[bench] = e_bench
        all_e += e_bench
    elasticity_stats = {}
    for name, values in list(elasticity.items()) + [("all", all_e)]:
        v = np.array(values, float)
        elasticity_stats[name] = dict(n=len(v), min=float(v.min()), p5=float(np.percentile(v, 5)), p50=float(np.percentile(v, 50)),
                                      p95=float(np.percentile(v, 95)), max=float(v.max()), mean=float(v.mean()))
    out = dict(reg=reg, pts=pts, raw=sorted(set(raw)), runs=runs, elasticity=elasticity_stats)
    dataloader.write_json("slopes", out)
    for bench, r in reg.items():
        print(bench, {qn: (round(v["slope"], 3), v["n"]) for qn, v in r.items()})
    print(json.dumps(elasticity_stats, indent=1))


def draw():
    """Draw mech_slopes with appc.draw_slopes (appc.save redirected to the mech figure writer) and write the record."""
    from src.analysis.appendix import appc
    plt = appc.style()
    # Liberation Serif is TrueType; the CFF Nimbus Roman cannot be embedded as Type 42
    plt.rcParams["font.serif"] = ["Liberation Serif"]
    c = json.loads((dataloader.CACHE / "slopes.json").read_text())
    appc.save = lambda fig, name: save_figure(fig, "slopes")
    appc.draw_slopes(plt, c)
    from src.analysis.appendix.appc import BENCH
    names = {b: n for b, n, _ in BENCH}
    rows = []
    for bench, disp in names.items():
        for qn in ("input", "output", "calls"):
            v = c["reg"][bench][qn]
            rows.append([disp, qn, f"{v['slope']:.3f}", f"{v['se']:.3f}", v["n"], v["n_tasks"], f"{v['r2']:.2f}"])
    table = logger.md_table(["Benchmark", "quantity", "slope k", "SE", "episodes", "tasks", "R^2 (within)"], rows)
    e_rows = []
    for name in list(c["reg"]) + ["all"]:
        s = c["elasticity"][name]
        e_rows.append([names.get(name, name), s["n"], f"{s['min']:.3f}", f"{s['p5']:.3f}", f"{s['p50']:.3f}", f"{s['p95']:.3f}", f"{s['max']:.3f}", f"{s['mean']:.3f}"])
    table += "\n\nPer-episode turn elasticity e_R = 1 + s_H R / (R - 1) over the same episodes (s_H = 1 - s_F, history share of input tokens)\n\n"
    table += logger.md_table(["Benchmark", "episodes", "min", "p5", "p50", "p95", "max", "mean"], e_rows)
    run_rows = [[names[b], ", ".join(f"{k}: {v}" for k, v in r.items())] for b, r in c["runs"].items()]
    table += "\n\nRun directories\n\n" + logger.md_table(["Benchmark", "runs"], run_rows)
    logger.write_record(
        "slopes",
        "Within-task slope k of log input tokens / log output tokens / log executed tool calls on log requests (turns), per benchmark; the dashed line is k = 1. Axes show the ratio to the task's geometric mean.",
        "DeepSeek-V4.1-Flash in OpenCode, test split: the seed's run-0 test episode and one episode per task of each baseline skill (GEPA, SkillOpt, SkillAdam; the https reruns native-20261008-https). No BACH episodes.",
        "Fixed-effects regression (per-task means removed) of log y on log requests, seed and the three baselines' episodes pooled, as mechanism/fig1.py fe_reg. Output row drops episodes with zero or missing output tokens.",
        table,
        ".venv/bin/python -m src.analysis.rq2.slopes compute\n.venv/bin/python -m src.analysis.rq2.slopes draw",
        "Numbers: assets/cache/analysis/rq/slopes.json (the BACH paper's fig1/cache.json is not touched).")


def draw_input():
    """Draw mech_slopes_input: the first row of mech_slopes (input tokens vs. turns) as a single-column 3.25 x 1.35 in figure, from slopes.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "STIXGeneral"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2})
    bench = [("tau3-bench", r"$\tau^3$-bench", "#0072B2"), ("automationbench", "AutoBench", "#D55E00"), ("terminal-bench-2-1", "TB2", "#009E73")]
    c = json.loads((dataloader.CACHE / "slopes.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(3.25, 1.35), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.25, top=0.88, wspace=0.3)
    rng = np.random.default_rng(0)
    xx = np.exp(np.array([-0.6, 0.6]))
    reg = {b: c["reg"][b]["input"] for b, _, _ in bench}
    pts = {b: (np.array(c["pts"][b]["input"][0]), np.array(c["pts"][b]["input"][1])) for b, _, _ in bench}
    # y range from the bulk of the episodes (0.5th to 99.5th percentile) and the fitted lines, so a few outlying episodes do not leave half the axes empty
    data_y = np.concatenate([pts[b][1] for b in pts])
    line_y = np.concatenate([np.log(xx) * max(reg[b]["slope"], 1.0) for b in pts])
    lo = np.exp(min(np.percentile(data_y, 0.5), line_y.min()))
    hi = np.exp(max(np.percentile(data_y, 99.5), line_y.max()))
    span = hi / lo
    lo, hi = lo / span ** 0.06, hi * span ** 0.06
    ticks = [t for t in (0.5, 1, 2, 4) if lo <= t <= hi]
    for ax, (b, disp, col) in zip(axs, bench):
        xs, ys = pts[b]
        sel = rng.choice(len(xs), min(len(xs), 250), replace=False)
        ax.scatter(np.exp(xs[sel]), np.exp(ys[sel]), s=2, color=col, alpha=0.3, lw=0, rasterized=True)
        ax.plot(xx, xx, color="k", ls="--", lw=0.6)
        ax.plot(xx, xx ** reg[b]["slope"], color=col, lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(np.exp(-0.7), np.exp(0.7))
        ax.set_ylim(lo, hi)
        ax.set_xticks([0.5, 1, 2])
        ax.set_xticklabels(["0.5", "1", "2"])
        ax.set_yticks(ticks)
        ax.minorticks_off()
        ax.set_title(disp, fontweight="bold", pad=2)
        ax.text(0.05, 0.95, f"slope {reg[b]['slope']:.2f}", transform=ax.transAxes, ha="left", va="top", color=col)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_yticklabels([f"{t:g}" for t in ticks])
    axs[0].set_ylabel("input tokens / task mean")
    axs[1].set_xlabel("turns / task mean")
    save_figure(fig, "slopes_input")
    rows = [[d, f"{reg[b]['slope']:.3f}", f"{reg[b]['se']:.3f}", reg[b]["n"], reg[b]["n_tasks"]] for b, d, _ in bench]
    logger.write_record("slopes_input", "First row of mech_slopes (input tokens vs. turns, both relative to the task's mean, log-log) as a 3.25 x 1.35 in single-column figure; dashed line k = 1.",
                   "As mech_slopes (seed run-0 and GEPA/SkillOpt/SkillAdam test episodes, DeepSeek-V4.1-Flash).", "Within-task fixed-effects slope, as mech_slopes; 250 points per panel are drawn at random (seed 0).",
                   logger.md_table(["Benchmark", "slope k", "SE", "episodes", "tasks"], rows), ".venv/bin/python -m src.analysis.rq2.slopes draw-input")
    print(rows)


if __name__ == "__main__":
    if sys.argv[1] == "compute":
        compute()
    elif sys.argv[1] == "draw-input":
        draw_input()
    else:
        draw()
