"""mech_slopes: the Appendix-C within-task slope figure of paper 1 from baseline episodes only.
Data (needs .venv): per benchmark the seed's run-0 test episode and one episode per task of each baseline skill (GEPA, SkillOpt, SkillAdam; common.returned_skills restricted to
the three baselines), exactly the pooling of mechanism/fig1.py compute, with the within-task fixed-effects regression fig1.fe_reg of log input tokens / log output tokens / log calls on log requests.
Also the per-episode turn elasticity e_R = 1 + s_H * R / (R - 1) (s_H = 1 - s_F, the history share of the input tokens, R = requests of the episode, both from slopes_checks.metrics).
  .venv/bin/python -m src.analysis.mech.slopes compute      -> assets/cache/analysis/mech/slopes.json
  python3 -m src.analysis.mech.slopes draw   -> paper/assets/fig/mech/mech_slopes.pdf/.svg and data/mech/mech_slopes.md"""
import collections
import json
import sys
import numpy as np
from src.analysis.mech import common as M

MODEL = "deepseek-flash"
EXPECTED = {"tau3-bench": dict(input=1.349, output=1.025, calls=0.840, n=300),
            "automationbench": dict(input=1.556, output=0.881, calls=0.853, n=480),
            "terminal-bench-2-1": dict(input=1.584, output=0.498, calls=0.936, n=71)}


def compute():
    """Pool seed and baseline episodes per benchmark, regress, and compute the e_R distribution; write mech/slopes.json."""
    from src.analysis import common as C
    from src.analysis.mechanism.slopes_checks import metrics, episodes
    from src.analysis.mechanism.fig1 import output_totals, fe_reg, BENCH
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
        for disp, (run, text) in C.returned_skills(MODEL, bench, M.BASELINES).items():
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
    M.write_json("slopes", out)
    for bench, r in reg.items():
        print(bench, {qn: (round(v["slope"], 3), v["n"]) for qn, v in r.items()})
    print(json.dumps(elasticity_stats, indent=1))


def draw():
    """Draw mech_slopes with appc.draw_slopes (appc.save redirected to the mech figure writer) and write the record."""
    from src.analysis.mechanism import appc
    plt = appc.style()
    # Liberation Serif is TrueType; the CFF Nimbus Roman cannot be embedded as Type 42
    plt.rcParams["font.serif"] = ["Liberation Serif"]
    c = json.loads((M.CACHE / "slopes.json").read_text())
    appc.save = lambda fig, name: M.save_figure(fig, "slopes")
    appc.draw_slopes(plt, c)
    from src.analysis.mechanism.fig1 import BENCH
    names = {b: n for b, n, _ in BENCH}
    rows = []
    for bench, disp in names.items():
        for qn in ("input", "output", "calls"):
            v = c["reg"][bench][qn]
            rows.append([disp, qn, f"{v['slope']:.3f}", f"{v['se']:.3f}", v["n"], v["n_tasks"], f"{v['r2']:.2f}"])
    table = M.md_table(["Benchmark", "quantity", "slope k", "SE", "episodes", "tasks", "R^2 (within)"], rows)
    e_rows = []
    for name in list(c["reg"]) + ["all"]:
        s = c["elasticity"][name]
        e_rows.append([names.get(name, name), s["n"], f"{s['min']:.3f}", f"{s['p5']:.3f}", f"{s['p50']:.3f}", f"{s['p95']:.3f}", f"{s['max']:.3f}", f"{s['mean']:.3f}"])
    table += "\n\nPer-episode turn elasticity e_R = 1 + s_H R / (R - 1) over the same episodes (s_H = 1 - s_F, history share of input tokens)\n\n"
    table += M.md_table(["Benchmark", "episodes", "min", "p5", "p50", "p95", "max", "mean"], e_rows)
    run_rows = [[names[b], ", ".join(f"{k}: {v}" for k, v in r.items())] for b, r in c["runs"].items()]
    table += "\n\nRun directories\n\n" + M.md_table(["Benchmark", "runs"], run_rows)
    M.write_record(
        "slopes",
        "Within-task slope k of log input tokens / log output tokens / log executed tool calls on log requests (turns), per benchmark; the dashed line is k = 1. Axes show the ratio to the task's geometric mean.",
        "DeepSeek-V4.1-Flash in OpenCode, test split: the seed's run-0 test episode and one episode per task of each baseline skill (GEPA, SkillOpt, SkillAdam; the https reruns native-20261008-https). No BACH episodes.",
        "Fixed-effects regression (per-task means removed) of log y on log requests, seed and the three baselines' episodes pooled, as mechanism/fig1.py fe_reg. Output row drops episodes with zero or missing output tokens.",
        table,
        ".venv/bin/python -m src.analysis.mech.slopes compute\npython3 -m src.analysis.mech.slopes draw",
        "Numbers: assets/cache/analysis/mech/slopes.json (the BACH paper's fig1/cache.json is not touched).")


if __name__ == "__main__":
    if sys.argv[1] == "compute":
        compute()
    else:
        draw()
