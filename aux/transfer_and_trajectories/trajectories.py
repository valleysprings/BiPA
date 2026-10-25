"""Trajectory dynamics behind the H1 / H2 / H4 hypotheses, on the avg@3 test runs of both models.

H1 (skills lower the number of attractors): the three test runs of one task should look more alike under a learned skill than under
the seed or no skill. Signature of an episode = the sequence of executed tool-call names (turn order). Per task: mean pairwise
similarity over the 3 run pairs, with two similarity measures
  - jacc: Jaccard index of the tool-name sets (1.0 when both episodes make no call);
  - editsim: 1 - Levenshtein(a, b) / max(len(a), len(b)) on the name sequences (1.0 when both are empty);
and the number of distinct signatures among the 3 runs (behavioural modes, 1..3). Also h = X / R (mean input tokens per turn),
the per-turn content the hypothesis says should shrink.

H4 (skills are knowledge triggers, pass@k diversity): pass@1 = per-task success rate over the 3 runs averaged over tasks
(= Table 1's Q when every run covers every task), pass@3 = share of tasks solved in at least one of the 3 runs, gap = pass@3 - pass@1.

H2 (skills just squeeze more thinking; test-time scaling): the model's own test-time curve = seed-episode success rate binned by
turns R (and by input tokens X), quantile bins over the pooled seed episodes of one (model, benchmark). Against it, per learned
skill and task: dR = mean R (skill) - mean R (seed), dQ likewise; "wasted squeeze" = tasks with dR > 0 whose dQ <= 0, as a share of
the dR > 0 tasks and of all tasks. For tasks with dR > 0: the seed curve's success rate at the bin of the task's seed mean R
(origin) and at the bin of its skill mean R (destination) — whether the extra turns buy curve segments that still pay.

Runs: DeepSeek-V4.1-Flash = Table 1's own runs (seed native-seed + rep-1/rep-2; skills and no-skill native-* + test-reps rep 1/2,
picked by table1.rep_files exactly as Table 1). GPT-6 Luna = the avg@3 protocol of src.run repeat (rep 0 = the source run's first evaluation; rep 1/2 = assets/artifacts/execution/gpt-6-luna/test-reps/<bench>__<arm>__<tag>,
unknown-only supplement evaluations merged per slot; TB2's three optimizer arms returned the seed text and report the seed row).
Episodes come from the dynamics module's load_cell, which reads evaluation.json directly: requests deduplicated
by request_id, provider-failed requests (no usage) dropped, X = summed input tokens, N = executed tool calls, R = completed requests;
an outcome is verified when success is a boolean and outcome_valid is not False (src.run.known).

TB2 has 18 tasks and one permanently unknown task (torch-pipeline-parallelism); its cells are reported but the evidence is tau3 (75)
and AutoBench (120). No model calls.

Commands:  python -m aux.transfer_and_trajectories.trajectories        -> assets/cache/analysis/main/dynamics.json, data/mech/mech_dynamics.md
           python -m aux.transfer_and_trajectories.trajectories draw   -> fig/mech/mech_dynamics.{pdf,svg}: (a) the seed's own test-time curve
                                                                          and (b) the white-squeeze rate of each learned skill
"""
import argparse
import json

import numpy as np

from src.analysis.dynamics import editsim, load_cell, luna_seed_skill
from src.utils import fig, logger, dataloader

MODELS = (("deepseek-flash", "DeepSeek"), ("gpt-6-luna", "Luna"))
ARMS = [("native-noseed", "No skill"), ("native-seed", "Seed skill"),
        ("gepa", "GEPA"), ("skillopt", "SkillOpt"), ("skilladam", "SkillAdam")]
BENCH = [("tau3-bench", "tau3"), ("automationbench", "AutoB"), ("terminal-bench-2-1", "TB2")]
N_BINS = 6


# H1: trajectory convergence

def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def h1_cell(reps):
    """H1 numbers of one arm's 3 runs: per task mean pairwise similarity, modes, h; task sets need 3 episodes with R > 0."""
    tasks = [t for t in set.intersection(*(set(reps[k]) for k in sorted(reps)))
             if all(reps[k][t]["R"] > 0 for k in sorted(reps))] if len(reps) == 3 else []
    jac, edit, modes, hh, rr, xx = [], [], [], [], [], []
    for t in tasks:
        seqs = [reps[k][t]["seq"] for k in sorted(reps)]
        pairs = [(0, 1), (0, 2), (1, 2)]
        jac.append(float(np.mean([jaccard(seqs[i], seqs[j]) for i, j in pairs])))
        edit.append(float(np.mean([editsim(seqs[i], seqs[j]) for i, j in pairs])))
        modes.append(len({tuple(s) for s in seqs}))
        hh.append(float(np.mean([reps[k][t]["X"] / reps[k][t]["R"] for k in sorted(reps)])))
        rr.append(float(np.mean([reps[k][t]["R"] for k in sorted(reps)])))
        xx.append(float(np.mean([reps[k][t]["X"] for k in sorted(reps)])))
    if not tasks:
        return None
    return {"n": len(tasks), "jacc": float(np.mean(jac)), "editsim": float(np.mean(edit)),
            "modes": float(np.mean(modes)), "one_mode_share": float(np.mean([m == 1 for m in modes])),
            "h": float(np.mean(hh)), "R": float(np.mean(rr)), "X": float(np.mean(xx))}


# H4: pass@k

def h4_cell(reps):
    """pass@1 / pass@3 over the tasks with a verified outcome in all 3 runs."""
    if len(reps) != 3:
        return None
    tasks = [t for t in set.intersection(*(set(reps[k]) for k in sorted(reps)))
             if all(reps[k][t]["success"] is not None for k in sorted(reps))]
    if not tasks:
        return None
    wins = {t: sum(reps[k][t]["success"] is True for k in sorted(reps)) for t in tasks}
    return {"n": len(tasks), "pass1": float(100 * np.mean([wins[t] / 3 for t in tasks])),
            "pass3": float(100 * np.mean([wins[t] >= 1 for t in tasks]))}


# H2: seed test-time curve and the squeeze

def seed_curve(seed_reps):
    """The model's own test-time curve: seed episodes (verified outcomes) binned by R and by X, N_BINS quantile bins."""
    eps = [seed_reps[k][t] for k in sorted(seed_reps) for t in seed_reps[k] if seed_reps[k][t]["success"] is not None]
    out = {}
    for key in ("R", "X"):
        vals = np.array([e[key] for e in eps], dtype=float)
        edges = np.unique(np.quantile(vals, np.linspace(0, 1, N_BINS + 1)))
        bins = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            sel = [e for e in eps if (e[key] >= lo if i == 0 else e[key] > lo) and e[key] <= hi]
            if sel:
                bins.append({"lo": float(lo), "hi": float(hi), "n": len(sel),
                             "success": float(100 * np.mean([e["success"] for e in sel]))})
        out[key] = bins
    return out


def bin_of(bins, value):
    """Index of the curve bin holding value (last bin when above the top edge)."""
    for i, b in enumerate(bins):
        if value <= b["hi"]:
            return i
    return len(bins) - 1


def h2_cell(own_reps, seed_reps, curve_bins):
    """Per-task dR / dQ against the seed on tasks with 3 verified runs on both sides; wasted-squeeze shares; curve segments."""
    if len(own_reps) != 3 or len(seed_reps) != 3:
        return None
    tasks = [t for t in set.intersection(*(set(own_reps[k]) for k in range(3)), *(set(seed_reps[k]) for k in range(3)))
             if all(own_reps[k][t]["success"] is not None and seed_reps[k][t]["success"] is not None for k in range(3))]
    if not tasks:
        return None
    rows = []
    for t in tasks:
        ro = float(np.mean([own_reps[k][t]["R"] for k in range(3)]))
        rs = float(np.mean([seed_reps[k][t]["R"] for k in range(3)]))
        qo = float(np.mean([float(own_reps[k][t]["success"]) for k in range(3)]))
        qs = float(np.mean([float(seed_reps[k][t]["success"]) for k in range(3)]))
        rows.append({"task": t, "dR": ro - rs, "dQ": qo - qs, "R_own": ro, "R_seed": rs})
    dr = np.array([r["dR"] for r in rows])
    dq = np.array([r["dQ"] for r in rows])
    up = dr > 0
    wasted = up & (dq <= 0)
    origin, dest = [], []
    for r in rows:
        if r["dR"] > 0 and curve_bins:
            origin.append(curve_bins[bin_of(curve_bins, r["R_seed"])]["success"])
            dest.append(curve_bins[bin_of(curve_bins, r["R_own"])]["success"])
    return {"n": len(tasks), "dR": float(dr.mean()), "dQ": float(100 * dq.mean()),
            "share_dR_up": float(up.mean()), "n_up": int(up.sum()), "n_wasted": int(wasted.sum()),
            "wasted_of_up": float(wasted.sum() / up.sum()) if up.sum() else None,
            "wasted_of_all": float(wasted.mean()),
            "curve_at_origin": float(np.mean(origin)) if origin else None,
            "curve_at_dest": float(np.mean(dest)) if dest else None}


# driver

def compute():
    res = {}
    for model, mshort in MODELS:
        for bench, bshort in BENCH:
            arms = {}
            for arm, label in ARMS:
                if model == "gpt-6-luna" and arm not in ("native-noseed", "native-seed") and luna_seed_skill(arm, bench):
                    arms[arm] = {"label": label, "shared_seed": True}
                    continue
                reps = load_cell(model, arm, bench)
                arms[arm] = {"label": label, "shared_seed": False, "reps": reps,
                             "h1": h1_cell(reps), "h4": h4_cell(reps)}
            seed_reps = arms["native-seed"].get("reps")
            curve = seed_curve(seed_reps) if seed_reps else None
            for arm, label in ARMS:
                cell = arms[arm]
                if cell.get("shared_seed") or arm in ("native-noseed", "native-seed") or not cell.get("reps"):
                    cell["h2"] = None
                    continue
                cell["h2"] = h2_cell(cell["reps"], seed_reps, curve["R"] if curve else None)
            res[f"{model}|{bench}"] = {"bench": bench, "short": bshort, "model": model, "curve": curve, "arms": arms}
            print(model, bench, "loaded")
    return res


# record

def fmt(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def record(res):
    h1_rows, h1_delta, h4_rows, c1_rows, c2_rows = [], [], [], [], []
    for key, cell in res.items():
        model, bench = key.split("|")
        mshort = dict(MODELS)[model]
        for arm, label in ARMS:
            a = cell["arms"][arm]
            if a.get("shared_seed"):
                h1_rows.append([mshort, cell["short"], label, "= Seed row (returned the seed)"])
                h4_rows.append([mshort, cell["short"], label, "= Seed row"])
                continue
            h1, h4 = a["h1"], a["h4"]
            if h1:
                h1_rows.append([mshort, cell["short"], label, h1["n"], fmt(h1["jacc"], 3), fmt(h1["editsim"], 3),
                                fmt(h1["modes"]), f"{100 * h1['one_mode_share']:.0f}%", f"{h1['h']:.0f}", fmt(h1["R"], 1), f"{h1['X']:.0f}"])
            if h4:
                h4_rows.append([mshort, cell["short"], label, h4["n"], f"{h4['pass1']:.1f}", f"{h4['pass3']:.1f}", f"{h4['pass3'] - h4['pass1']:+.1f}"])
        seed_h1 = cell["arms"]["native-seed"]["h1"]
        for arm, label in ARMS[2:]:
            a = cell["arms"][arm]
            if a.get("shared_seed") or not a.get("h1") or not seed_h1:
                continue
            h1_delta.append([mshort, cell["short"], label,
                             f"{a['h1']['jacc'] - seed_h1['jacc']:+.3f}", f"{a['h1']['editsim'] - seed_h1['editsim']:+.3f}",
                             f"{a['h1']['modes'] - seed_h1['modes']:+.2f}", f"{a['h1']['h'] / seed_h1['h'] - 1:+.2f}"])
        if cell["curve"]:
            for i, b in enumerate(cell["curve"]["R"]):
                c1_rows.append([mshort, cell["short"], "turns R", i, f"{b['lo']:.0f}-{b['hi']:.0f}", b["n"], f"{b['success']:.1f}"])
            for i, b in enumerate(cell["curve"]["X"]):
                c1_rows.append([mshort, cell["short"], "input tokens X", i, f"{b['lo'] / 1000:.0f}k-{b['hi'] / 1000:.0f}k", b["n"], f"{b['success']:.1f}"])
        for arm, label in ARMS[2:]:
            a = cell["arms"][arm]
            if a.get("h2"):
                h = a["h2"]
                wasted = f"{100 * h['wasted_of_up']:.0f}% ({h['n_wasted']}/{h['n_up']})" if h["wasted_of_up"] is not None else "-"
                c2_rows.append([mshort, cell["short"], label, h["n"], f"{h['dR']:+.2f}", f"{100 * h['share_dR_up']:.0f}%",
                                f"{h['dQ']:+.1f}", wasted,
                                f"{100 * h['wasted_of_all']:.0f}%", fmt(h["curve_at_origin"], 1), fmt(h["curve_at_dest"], 1)])
    what = ("Trajectory dynamics for three mechanism hypotheses, on the avg@3 test runs of DeepSeek-V4.1-Flash (Table 1's runs) and GPT-6 Luna "
            "(the repeat-protocol runs). H1: convergence of a task's 3 runs — pairwise tool-name-sequence similarity (Jaccard on name sets; "
            "editsim = 1 - normalized Levenshtein) and behavioural modes (distinct call-name signatures among the 3 runs), plus h = X/R (input "
            "tokens per turn). H4: pass@1 (per-task success rate averaged over tasks) vs pass@3 (tasks solved in >= 1 of 3 runs). "
            "H2: the seed's own test-time curve (success binned by turns / input tokens) and per-task dR vs dQ of each learned skill against the "
            "seed; 'wasted squeeze' = tasks with more turns but no success gain.")
    runs = ("DeepSeek: seed = native-seed + rep-1/rep-2; no-skill and skills = their native runs + test-reps rep 1/2 (analysis.table1.rep_files, "
            "Table 1's selection). Luna: rep 0 = source run (native-20261011 native arms, native-20261012 optimizers), rep 1/2 = "
            "assets/artifacts/execution/gpt-6-luna/test-reps/<bench>__<arm>__<tag>, unknown-only supplements merged per slot (the src.run repeat "
            "protocol). Luna TB2's GEPA/SkillOpt/SkillAdam returned the seed text: their rows are the Seed row and are marked, not recomputed.")
    agg = ("Episode fields from evaluation.json: requests deduplicated by request_id, provider-failed requests (no usage) dropped; X = summed input "
           "tokens, N = executed tool calls, R = completed requests, signature = executed tool names in turn order. Verified outcome = boolean "
           "success with outcome_valid not False. H1 task set: 3 runs with R > 0. H4/H2 task set: verified outcome in all 3 runs (both sides for H2). "
           "Curve bins: 6 quantile bins over the pooled seed episodes of the (model, benchmark); origin/dest = the curve success at the bin of the "
           "task's seed mean R / skill mean R, averaged over the tasks with dR > 0. TB2 has 18 tasks and one permanently unknown task "
           "(torch-pipeline-parallelism); the evidence is tau3 (75) and AutoBench (120).")
    table = ("### A. H1: run-to-run trajectory similarity, behavioural modes, per-turn content h = X/R\n\n"
             + logger.md_table(["Model", "Bench", "Arm", "n tasks", "jacc", "editsim", "modes (of 3)", "1-mode %", "h = X/R", "R", "X"], h1_rows)
             + "\n\n### A2. H1: learned skill minus Seed (same benchmark)\n\n"
             + logger.md_table(["Model", "Bench", "Arm", "d jacc", "d editsim", "d modes", "d h (relative)"], h1_delta)
             + "\n\n### B. H4: pass@1 vs pass@3 over the 3 runs\n\n"
             + logger.md_table(["Model", "Bench", "Arm", "n tasks", "pass@1 %", "pass@3 %", "gap (pts)"], h4_rows)
             + "\n\n### C1. H2: the seed's own test-time curve (success % by quantile bin)\n\n"
             + logger.md_table(["Model", "Bench", "x", "bin", "range", "n episodes", "success %"], c1_rows)
             + "\n\n### C2. H2: learned skill vs seed, per task (dR in turns, dQ in points; wasted = dR > 0 and dQ <= 0)\n\n"
             + logger.md_table(["Model", "Bench", "Arm", "n tasks", "mean dR", "dR>0 %", "mean dQ", "wasted of dR>0", "wasted of all",
                            "seed curve % at origin bin", "seed curve % at dest bin"], c2_rows))
    dataloader.write_json("dynamics", {"cells": {k: {"curve": v["curve"],
                                             "arms": {a: {"label": x["label"], "shared_seed": x.get("shared_seed", False),
                                                          "h1": x.get("h1"), "h4": x.get("h4"), "h2": x.get("h2")}
                                                      for a, x in v["arms"].items()}} for k, v in res.items()}})
    logger.write_record("dynamics", what, runs, agg, table, "python -m aux.transfer_and_trajectories.trajectories")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", nargs="?", default="record", choices=["record", "draw"])
    if ap.parse_args().step == "draw":
        draw()
    else:
        record(compute())


# figure (reads the cache compute wrote)

FIG_BENCH = [("tau3-bench", r"$\tau^3$"), ("automationbench", "AutoB")]  # TB2's 18 tasks are too few for this statistic
# bench color in (a); optimizer color in (b), as in mech_paired
BC = {"tau3-bench": "#0072B2", "automationbench": "#E69F00"}
OC = {"SkillOpt": "#0072B2", "SkillAdam": "#D55E00", "GEPA": "#009E73"}
LS = {"deepseek-flash": "-", "gpt-6-luna": "--"}
MK = {"deepseek-flash": "o", "gpt-6-luna": "s"}


def load():
    return json.loads((dataloader.CACHE / "dynamics.json").read_text())["cells"]


def draw_curve(ax, cells):
    """(a) Seed success per turns bin; color = benchmark, line style / marker = model; the legend carries the turn ranges."""
    for model, mshort in MODELS:
        for bench, bshort in FIG_BENCH:
            bins = cells[f"{model}|{bench}"]["curve"]["R"]
            xs = list(range(len(bins)))
            ys = [b["success"] for b in bins]
            label = f"{mshort} {bshort}, {bins[0]['lo']:.0f}–{bins[-1]['hi']:.0f} turns"
            ax.plot(xs, ys, LS[model], marker=MK[model], ms=3, lw=1.0, color=BC[bench],
                    markerfacecolor="white" if model == "gpt-6-luna" else BC[bench], markeredgewidth=0.8, label=label)
    ax.set_xticks(range(6))
    ax.set_xticklabels([str(i + 1) for i in range(6)])
    ax.set_xlabel("turns bin of the seed's episodes (low $\u2192$ high)", fontsize=7, labelpad=1)
    ax.set_ylabel("seed success (%)", fontsize=7, labelpad=1)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=6.5, length=2, pad=1)
    ax.legend(frameon=False, fontsize=6.3, loc="lower left", handlelength=1.8, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_squeeze(ax, cells):
    """(b) White-squeeze bars: groups = model x benchmark, bars = optimizer, task counts on top."""
    import matplotlib.pyplot as plt
    width = 0.26
    for g, (model, mshort) in enumerate(MODELS):
        for b, (bench, bshort) in enumerate(FIG_BENCH):
            gx = 2 * g + b
            cell = cells[f"{model}|{bench}"]
            for j, (arm, label) in enumerate(ARMS[2:]):
                h = cell["arms"][arm]["h2"]
                v = 100 * h["wasted_of_up"]
                x = gx + (j - 1) * width
                ax.bar(x, v, width=width * 0.9, color=OC[label], edgecolor="white", linewidth=0.4)
                ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.3)
                ax.text(x, v / 2, f"{h['n_wasted']}/{h['n_up']}", ha="center", va="center", fontsize=5.6, color="white", rotation=90)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{mshort}\n{bshort}" for mshort in ("DeepSeek", "Luna") for bshort in (r"$\tau^3$", "AutoB")], fontsize=6.5)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("white-squeeze (%)", fontsize=7, labelpad=1)
    ax.tick_params(axis="y", labelsize=6.5, length=2, pad=1)
    ax.tick_params(axis="x", length=0, pad=2)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=OC[label], label=label) for _, label in ARMS[2:]]
    ax.legend(handles=handles, frameon=False, fontsize=6.3, loc="lower left", ncol=3, handlelength=0.9, handletextpad=0.3,
              columnspacing=0.6, borderaxespad=0.1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw():
    """The two-panel mech_dynamics figure from assets/cache/analysis/main/dynamics.json (numbers as recorded in mech_dynamics.md, C1 and C2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.linewidth": 0.6})
    cells = load()
    figure = plt.figure(figsize=(6.75, 2.1))
    ax_a = figure.add_axes([0.065, 0.15, 0.42, 0.70])
    ax_b = figure.add_axes([0.585, 0.15, 0.40, 0.70])
    draw_curve(ax_a, cells)
    draw_squeeze(ax_b, cells)
    figure.text(0.065, 0.985, "(a) The seed's own test-time curve", fontsize=7.5, fontweight="bold", va="top")
    figure.text(0.585, 0.985, "(b) Extra turns buy no success", fontsize=7.5, fontweight="bold", va="top")
    fig.save_figure(figure, "dynamics")


if __name__ == "__main__":
    main()
