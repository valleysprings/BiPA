"""Draw mech_slopes (the appendix within-task slope figure) and write its record from slopes.json (written by the slopes compute step)."""
import json

from src.utils import dataloader, logger
from src.utils.fig import save_figure


def draw():
    """Draw mech_slopes with appc.draw_slopes (appc.save redirected to the mech figure writer) and write the record."""
    from aux.cost_appendix import appc_draw as appc
    plt = appc.style()
    # Liberation Serif is TrueType; the CFF Nimbus Roman cannot be embedded as Type 42
    plt.rcParams["font.serif"] = ["Liberation Serif"]
    c = json.loads((dataloader.CACHE / "slopes.json").read_text())
    appc.save = lambda fig, name: save_figure(fig, "slopes")
    appc.draw_slopes(plt, c)
    names = {b: n for b, n, _ in appc.BENCH}
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
        "DeepSeek-V4.1-Flash in OpenCode, test split: the seed's run-0 test episode and one episode per task of each baseline skill (GEPA, SkillOpt, SkillAdam; the https reruns native-20261008-https).",
        "Fixed-effects regression (per-task means removed) of log y on log requests, seed and the three baselines' episodes pooled, as appc.fe_reg. Output row drops episodes with zero or missing output tokens.",
        table,
        "python -m src.analysis.slopes compute\npython -m aux.cost_appendix.slopes_draw",
        "Numbers: assets/cache/analysis/main/slopes.json.")


if __name__ == "__main__":
    draw()
