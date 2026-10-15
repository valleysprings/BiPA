"""Cost-success Pareto figure: test Q (y) against workload ratio rho (x, log) per model x benchmark.

Run with system python3 (matplotlib). Inputs, all read from files:
  assets/artifacts/results/main-table.csv      native-seed run's seed, SkillOpt, GEPA
  assets/artifacts/results/ours-front.md       every BACH front point on test; returned = yes is the reported skill
  assets/artifacts/results/pareto-inputs.json  SkillAdam points and the seed's Q inside each BACH run (written by pareto_inputs.py with the slim venv)
Writes assets/artifacts/results/figures/pareto.{pdf,png}; pass --copy-to DIR to also copy both there.
"""
import argparse, csv, json, re, shutil
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "assets/artifacts/results"
MODELS = (("deepseek-flash", "DeepSeek-V4.1-Flash"), ("gpt-6-luna", "GPT-6 Luna"))
BENCHES = (("tau3-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2"))
KAPPA = 1.1
C = {"skillopt": "#E69F00", "gepa": "#56B4E9", "skilladam": "#009E73", "bach": "#D55E00", "seed": "#000000"}
M = {"skillopt": "s", "gepa": "^", "skilladam": "D", "bach": "o"}
LABEL = {"seed": "Seed", "skillopt": "SkillOpt", "gepa": "GEPA", "skilladam": "SkillAdam",
         "bach": r"BACH returned", "front": r"BACH front point"}


def num(s):  # "77.3 (75/75)" -> 77.3
    return float(re.match(r"\s*([-\d.]+)", s).group(1))


def load():
    pts = defaultdict(list)  # (model, bench) -> [(method, rho, Q, filled)]
    for r in csv.DictReader(open(RES / "main-table.csv")):
        if r["method"] in ("native-seed", "skillopt", "gepa") and r["status"] == "done":
            pts[(r["model"], r["bench"])].append(("seed" if r["method"] == "native-seed" else r["method"], float(r["rho"]), float(r["Q"]), True))
    for line in open(RES / "ours-front.md"):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 8 and c[0] in dict(MODELS):
            pts[(c[0], c[1])].append(("bach", num(c[7]), num(c[6]), c[3] == "yes"))
    pi = json.loads((RES / "pareto-inputs.json").read_text())["skilladam"]
    for m, d in pi.items():
        for b, v in d.items():
            pts[(m, b)].append(("skilladam", v["rho"], v["Q"], True))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy-to")
    args = ap.parse_args()
    pts = load()
    seeds = json.loads((RES / "pareto-inputs.json").read_text())["run_seed_Q"]
    plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "serif"})
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.1))
    for i, (model, mlabel) in enumerate(MODELS):
        for j, (bench, blabel) in enumerate(BENCHES):
            ax = axes[i][j]
            P = pts[(model, bench)]
            seed_in_run = seeds.get(model, {}).get(bench, {}).get("zo")
            ax.axvspan(KAPPA, 2.0, color="#EEEEEE", zorder=0)
            ax.axvline(KAPPA, color="#666666", lw=0.8, ls="--", zorder=1)
            # markers at the same place are layered by size (seed cross > BACH disc > baselines), none is moved
            for method, rho, q, filled in P:
                if method == "bach":
                    ax.scatter([rho], [q], marker="o", s=78 if filled else 52, zorder=3 if filled else 5,
                               facecolors=C["bach"] if filled else "none", edgecolors=C["bach"], linewidths=1.3)
                elif method != "seed":
                    ax.scatter([rho], [q], marker=M[method], s=26, c=C[method], zorder=6, linewidths=0.4, edgecolors="white")
            if seed_in_run is not None:
                ax.scatter([1.0], [seed_in_run], marker="+", s=110, c=C["seed"], zorder=8, linewidths=1.6)
            for method, rho, q, filled in P:
                if method == "seed" and seed_in_run is not None and abs(q - seed_in_run) > 0.05:
                    ax.scatter([1.0], [q], marker="+", s=70, c="#888888", zorder=7, linewidths=1.0)
            ax.set_xscale("log")
            ax.set_xlim(0.85, 1.7)
            ax.set_xticks([0.9, 1.0, 1.1, 1.3, 1.5])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
            ax.minorticks_off()
            qs = [p[2] for p in P] + ([seed_in_run] if seed_in_run is not None else [])
            lo, hi = min(qs), max(qs)
            pad = max(3.0, 0.15 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
            ax.grid(axis="y", color="#DDDDDD", lw=0.4)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_title(f"{mlabel}, {blabel}", fontsize=7.5)
            if j == 0:
                ax.set_ylabel("test success $Q$ (%)")
            if i == 1:
                ax.set_xlabel(r"workload ratio $\rho$ (log)")
    h = [Line2D([], [], marker="+", ls="", color=C["seed"], ms=8, mew=1.6, label="Seed (in the BACH run)"),
         Line2D([], [], marker="+", ls="", color="#888888", ms=7, mew=1.0, label="Seed (native run)")]
    h += [Line2D([], [], marker=M[k], ls="", color=C[k], ms=5, label=LABEL[k]) for k in ("skillopt", "gepa", "skilladam")]
    h += [Line2D([], [], marker="o", ls="", mfc=C["bach"], mec=C["bach"], ms=6, label=LABEL["bach"]),
          Line2D([], [], marker="o", ls="", mfc="white", mec=C["bach"], mew=1.2, ms=6, label=LABEL["front"]),
          Line2D([], [], color="#666666", ls="--", lw=0.8, label=r"budget $\rho=1.1$")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=7, handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=0.8, w_pad=0.6)
    out = RES / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"pareto.{ext}", dpi=220)
        if args.copy_to:
            Path(args.copy_to).mkdir(parents=True, exist_ok=True)
            shutil.copy(out / f"pareto.{ext}", args.copy_to)
    print("wrote", out)


if __name__ == "__main__":
    main()
