"""mech_selrules: which acceptance rule selects against the noise of selection (paper 1, design principle "Select against the noise of selection").

Role in the argument.  An optimizer should accept an edit only when its dev gain clears what its own dev comparison can detect, or confirm it
on a second draw.  This module measures, per benchmark, how often four acceptance rules accept ONE candidate against its parent, when the
candidate is identical to the parent (false acceptance) and when it truly solves g more dev tasks in expectation (power), and then applies the
detectability gate to the logged accepted edges of the Table 1 runs.

Algorithm.  The task model is selection.py's: the seed's dev tasks with beta-binomial posterior success probabilities p (fitted prior).  A run of a
skill is the number of dev tasks solved when each task is Bernoulli(p); a true gain of g tasks is selection.raised(p, g) (probability mass added
on the least-solved tasks).  Per draw of the task probabilities, the parent and the candidate are run and compared under
  (a) better     candidate solves strictly more tasks than the parent on one run each (SkillOpt, SkillAdam gate)
  (b) detectable the single-run gain reaches MDE80 of the panel (selection.mde80, tasks)
  (c) confirm    better on one run, then both re-run on a fresh draw and better again
  (d) mean of 3  the sum of three runs each (same as the mean) is strictly higher for the candidate
For the logged edges: dQ (success points) x n_pair / 100 is the edge's dev gain in tasks; an accepted edge is refused by rule (b) when that
gain is below MDE80 of a comparison of n_pair tasks (the panel for full-panel edges, a random n_pair-task subset of the dev tasks for SkillAdam's
validation minibatches).  Refused and surviving accepted edges are compared on their change in mean log input tokens, dlogX.

  .venv/bin/python -m analysis.mech.selrules compute   -> assets/cache/analysis/mech/selrules.json, data/mech/mech_selrules.md
  python3 -m analysis.mech.selrules draw -> fig/mech/mech_selrules.{pdf,svg} (one column), cache mech_selrules.png
"""
import json
import sys

import numpy as np

from analysis.mech import common as M
from analysis.mech import edges
from analysis.mech import selection as S

RULES = [("better", "(a) better on one run"), ("detectable", "(b) gain >= MDE80"), ("confirm", "(c) better, then better again"),
         ("mean3", "(d) mean of 3 runs")]
TRUE_GAINS = (1, 2, 4)       # tasks; the panel shows PLOT_GAIN
PLOT_GAIN = 2
DRAWS = 100_000
MARKERS = {"better": "o", "detectable": "s", "confirm": "^", "mean3": "D"}
COLORS = {"terminal-bench-2-1": "#D55E00", "tau3-bench": "#0072B2", "automationbench": "#009E73"}


# ---- the simulation ----------------------------------------------------------------------------------------------------------------

def runs_total(p, repeats):
    """Tasks solved summed over `repeats` independent runs of a skill with task probabilities p (draws, tasks)."""
    total = np.zeros(p.shape[0], dtype=int)
    for _ in range(repeats):
        total += S.outcomes(p)
    return total


def acceptance(p, q, mde) -> dict:
    """{rule: acceptance rate} of one candidate with task probabilities q against a parent with p, on the same draws of the task model."""
    parent, child = S.outcomes(p), S.outcomes(q)
    net = child - parent
    # (c) a fresh draw of both after a first win
    net_again = S.outcomes(q) - S.outcomes(p)
    # (d) three runs each, summed
    net_mean3 = runs_total(q, 3) - runs_total(p, 3)
    return {"better": float((net > 0).mean()),
            "detectable": float((net >= mde).mean()),
            "confirm": float(((net > 0) & (net_again > 0)).mean()),
            "mean3": float((net_mean3 > 0).mean())}


def panel_threshold(dev, prior, size, index) -> float:
    """MDE80 (tasks) of a single-run comparison on `size` dev tasks (the panel, or a random subset of it)."""
    S.reseed(index, 31, size)
    p = S.task_probabilities(dev, prior, size, S.DRAWS)
    return S.mde80(p)[1]


def analyse(bench) -> dict:
    """Null acceptance and power per rule for one benchmark, plus the MDE80 thresholds of the panel sizes used by the logged edges."""
    dev_files, test_files = S.seed_files(bench)
    dev, test = S.executions(dev_files), S.executions(test_files)
    n_dev = len(dev)
    pooled = [(sum(v), len(v)) for v in list(dev.values()) + list(test.values())]
    prior = S.fit_prior(pooled)
    index = [x for x, _ in S.BENCHES].index(bench)
    S.reseed(index, 30)
    p = S.task_probabilities(dev, prior, n_dev, DRAWS)
    mde = panel_threshold(dev, prior, n_dev, index)
    S.reseed(index, 32)
    out = {"dev_tasks": n_dev, "mde80_tasks": mde, "null": acceptance(p, p, mde), "power": {}}
    for gain in TRUE_GAINS:
        out["power"][str(gain)] = acceptance(p, S.raised(p, gain), mde)
    # thresholds of the smaller comparisons logged (SkillAdam's validation minibatches)
    sizes = sorted({r["n_pair"] for r in logged_edges() if r["bench"] == bench})
    out["mde80_by_size"] = {str(s): (mde if s == n_dev else panel_threshold(dev, prior, s, index)) for s in sizes}
    return out


# ---- the logged edges --------------------------------------------------------------------------------------------------------------

def logged_edges() -> list:
    """The canonical dev edges (59) of the Table 1 runs."""
    return edges.load_dev_edges()


def weighted_mean(values, weights) -> float:
    """Mean of values with weights."""
    return float(np.average(values, weights=weights))


def edge_gate(result) -> dict:
    """Of the accepted dev edges, those a rule-(b) gate refuses (dev gain in tasks below MDE80 of the edge's comparison size) and their mean
    dlogX (x100 log points), unweighted and weighted by n_pair, against the surviving edges."""
    accepted = [r for r in logged_edges() if r["accepted"]]
    gain_tasks = np.array([r["dQ"] * r["n_pair"] / 100 for r in accepted])
    threshold = np.array([result[r["bench"]]["mde80_by_size"][str(r["n_pair"])] for r in accepted])
    refused = gain_tasks < threshold
    dlogx = 100 * np.array([r["dlogX"] for r in accepted])
    weight = np.array([r["n_pair"] for r in accepted], float)
    out = {"accepted": len(accepted), "refused": int(refused.sum()), "survive": int((~refused).sum()),
           "gain_tasks": gain_tasks.tolist(), "threshold_tasks": threshold.tolist(), "per_bench": {}}
    for name, mask in (("refused", refused), ("survive", ~refused)):
        out[name + "_dlogX_mean"] = float(dlogx[mask].mean()) if mask.any() else float("nan")
        out[name + "_dlogX_weighted"] = weighted_mean(dlogx[mask], weight[mask]) if mask.any() else float("nan")
    for bench, _ in S.BENCHES:
        in_bench = np.array([r["bench"] == bench for r in accepted])
        out["per_bench"][bench] = {"accepted": int(in_bench.sum()), "refused": int((refused & in_bench).sum())}
    return out


# ---- outputs -----------------------------------------------------------------------------------------------------------------------

def record(result):
    """The audit record data/mech/mech_selrules.md."""
    rows = []
    for bench, _ in S.BENCHES:
        r = result[bench]
        for key, label in RULES:
            rows.append([bench, label, f"{100 * r['null'][key]:.1f}"] + [f"{100 * r['power'][str(g)][key]:.1f}" for g in TRUE_GAINS])
    head = ["benchmark", "rule", "false acceptance %"] + [f"accepted at true gain {g} tasks %" for g in TRUE_GAINS]
    gate = result["edge_gate"]
    mde_rows = [[bench, result[bench]["dev_tasks"], f"{result[bench]['mde80_tasks']:.2f}", f"{100 * result[bench]['mde80_tasks'] / result[bench]['dev_tasks']:.1f}",
                 json.dumps(result[bench]["mde80_by_size"])] for bench, _ in S.BENCHES]
    bench_rows = [[bench, v["accepted"], v["refused"]] for bench, v in gate["per_bench"].items()]
    extra = "\n".join([
        "MDE80 of a single-run comparison (selection.mde80, fitted prior):", "",
        M.md_table(["benchmark", "dev tasks", "MDE80 (tasks)", "MDE80 (points)", "MDE80 by comparison size (tasks)"], mde_rows), "",
        f"Logged accepted edges refused by rule (b): {gate['refused']} of {gate['accepted']} (survive {gate['survive']}).", "",
        M.md_table(["benchmark", "accepted edges", "refused by (b)"], bench_rows), "",
        M.md_table(["group", "edges", "mean dlogX (x100)", "n_pair-weighted dlogX (x100)"],
                   [["refused", gate["refused"], f"{gate['refused_dlogX_mean']:.1f}", f"{gate['refused_dlogX_weighted']:.1f}"],
                    ["survive", gate["survive"], f"{gate['survive_dlogX_mean']:.1f}", f"{gate['survive_dlogX_weighted']:.1f}"]])])
    M.write_record(
        "selrules",
        "How often four acceptance rules accept one candidate against its parent: a candidate identical to the parent (false acceptance) and "
        f"one that truly solves g more dev tasks in expectation (g = {', '.join(map(str, TRUE_GAINS))}; the figure shows g = {PLOT_GAIN}). "
        "Also, how many of the 30 accepted logged dev edges a detectability gate (rule b) would have refused, and the token cost they carried.",
        "Seed skill dev and test executions of Table 1 (native-20261008-https), the beta-binomial task model of mech_selection; logged edges: the 59 "
        "canonical dev edges (edges_table.json, SkillAdam x terminal-bench-2-1 excluded), 30 accepted.",
        f"{S.DRAWS} draws of the task probabilities per benchmark; each rule compares one candidate with its parent on the same draw; rules: "
        + "; ".join(label for _, label in RULES) + ". Seeds fixed per benchmark. Edge gain in tasks = dQ x n_pair / 100 against MDE80 of an "
        "n_pair-task comparison.",
        M.md_table(head, rows),
        ".venv/bin/python -m analysis.mech.selrules compute\npython3 -m analysis.mech.selrules draw",
        extra)


def compute():
    """Simulate every benchmark, gate the logged edges, and write the cache and the audit record."""
    result = {bench: analyse(bench) for bench, _ in S.BENCHES}
    result["edge_gate"] = edge_gate(result)
    result["plot_gain"] = PLOT_GAIN
    M.write_json("selrules", result)
    record(result)


# ---- figure ------------------------------------------------------------------------------------------------------------------------

def legend_handles():
    """Marker (rule) and color (benchmark) legend handles."""
    from matplotlib.lines import Line2D
    rules = [Line2D([], [], color="0.3", lw=0, marker=MARKERS[k], markersize=4, label=label) for k, label in RULES]
    benches = [Line2D([], [], color=COLORS[b], lw=0, marker="o", markersize=4, label=d) for b, d in (
        ("terminal-bench-2-1", "TB2"), ("tau3-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench"))]
    return rules, benches


def panel(ax, data):
    """One axes: x = false-acceptance rate (%), y = acceptance at the true gain (%), a marker per rule, a color per benchmark; the diagonal
    marks a rule that accepts true and null candidates alike."""
    gain = str(data["plot_gain"])
    for bench in COLORS:
        for key, _ in RULES:
            x = 100 * data[bench]["null"][key]
            y = 100 * data[bench]["power"][gain][key]
            ax.plot([x], [y], marker=MARKERS[key], color=COLORS[bench], markersize=4.5, markeredgewidth=0, lw=0)
    ax.plot([0, 100], [0, 100], color="0.8", lw=0.6, zorder=0)
    ax.set_xlim(-2, 52)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("false acceptance (%)", fontsize=7)
    ax.set_ylabel(f"accepted at true gain {data['plot_gain']} tasks (%)", fontsize=7)
    ax.tick_params(labelsize=7, length=2)
    ax.spines[["top", "right"]].set_visible(False)


def panel_bars(ax, data):
    """One axes for the design-principle figure: per acceptance rule, how often it keeps a candidate identical to its parent (light bar) and
    one that truly solves PLOT_GAIN more dev tasks (dark bar); bar = mean over the three benchmarks, whisker = their range."""
    gain = str(data["plot_gain"])
    benches = list(COLORS)
    labels = ["one win\n(today)", "one large\nwin", "two wins\nin a row", "3-run\ntotal"]
    width = 0.38
    for i, (key, _) in enumerate(RULES):
        null = [100 * data[b]["null"][key] for b in benches]
        power = [100 * data[b]["power"][gain][key] for b in benches]
        for offset, values, color in ((-width / 2, null, "0.72"), (width / 2, power, "#0072B2")):
            mean = float(np.mean(values))
            ax.bar(i + offset, mean, width=width, color=color, edgecolor="none")
            ax.plot([i + offset] * 2, [min(values), max(values)], color="0.15", lw=0.7)
    ax.set_xticks(range(len(RULES)))
    ax.set_xticklabels(labels, fontsize=6, linespacing=0.95)
    ax.set_xlim(-0.6, len(RULES) - 0.4)
    # headroom above 100 for the legend
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("edits kept (%)", fontsize=7, labelpad=1)
    ax.tick_params(axis="y", labelsize=7, length=2)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_bounds(0, 100)


def style():
    """matplotlib with TrueType fonts in the pdf."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.size"] = 7
    import matplotlib.pyplot as plt
    return plt


def draw():
    """Standalone single-column version of the panel with its legend, saved as mech_selrules."""
    plt = style()
    data = json.load(open(M.CACHE / "selrules.json"))
    fig, ax = plt.subplots(figsize=(3.25, 2.9))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.40)
    panel(ax, data)
    rules, benches = legend_handles()
    fig.legend(handles=rules + benches, loc="lower center", ncol=2, fontsize=7, frameon=False, handletextpad=0.3, columnspacing=1.0,
               bbox_to_anchor=(0.5, 0.0))
    M.save_figure(fig, "selrules")


if __name__ == "__main__":
    {"compute": compute, "draw": draw}[sys.argv[1]]()
