"""mech_reruns: Appendix-B rerun figure of paper 1, appb.compute_reruns / draw_reruns without BACH.
Data (needs .venv): the seed's three test runs and each baseline's three test executions (native-20261008-https run + test-reps rep 1/2), per-run success and
input-token ratio against the pooled seed; appb's own frozen-value asserts run unchanged. PINS is copied without the BACH entries and appb's cache directory is
redirected to assets/cache/analysis/mech/ (appb.py itself is not edited).
  .venv/bin/python -m src.analysis.mech.reruns compute
  python3 -m src.analysis.mech.reruns draw"""
import json
import sys
import numpy as np
from src.analysis.mechanism import appb
from src.utils import fig, logger, dataloader
def patch_appb():
    """Point appb at the baseline-only pins and the mech cache directory."""
    pins = {}
    for bench, cells in appb.PINS.items():
        pins[bench] = {name: pin for name, pin in cells.items() if name != "BACH"}
    appb.PINS = pins
    dataloader.CACHE.mkdir(parents=True, exist_ok=True)
    appb.out_dir = lambda: dataloader.CACHE


def compute():
    """Run appb.compute_reruns on the baseline-only pins; it writes mech/reruns.json."""
    patch_appb()
    appb.compute_reruns()


def draw_reruns(plt, d):
    """appb.draw_reruns with BACH dropped from the skills and from the legend; layout, sizes and colours as in appb."""
    B = d["benches"]
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, axs = plt.subplots(2, 3, figsize=(6.3, 3.4))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.15, top=0.88, hspace=0.75, wspace=0.34)
    for j, (bench, disp) in enumerate(appb.BENCH):
        sk = B[bench]["skills"]
        sd = B[bench]["seed"]
        names = ["seed"] + [s for s in ("GEPA", "SkillOpt", "SkillAdam") if s in sk]
        x = {s: i for i, s in enumerate(names)}
        for row, ax in enumerate(axs[:, j]):
            seedvals = sd["succ"] if row == 0 else sd["rho_pool"]
            ax.axhspan(min(seedvals), max(seedvals), color="0.5", alpha=0.15, lw=0, zorder=0)
            ax.axhline(float(np.mean(seedvals)), color="0.5", lw=0.8, ls="-", zorder=1)
            for s in names:
                vals = sk[s]["succ"] if row == 0 else sk[s]["rho_pool"]
                col = "0.45" if s == "seed" else appb.OC[s]
                mk = "o" if s == "seed" else appb.OM[s]
                n = len(vals)
                xs = x[s] + (np.arange(n) - (n - 1) / 2) * min(0.22, 0.5 / max(n, 1))
                ax.scatter(xs, vals, s=16, marker=mk, color=col, zorder=4, lw=0)
                if s != "seed" and "rho_pooled" in sk[s] and row == 1:
                    # pooled ratio of the skill's own runs
                    ax.plot([x[s] - 0.28, x[s] + 0.28], [sk[s]["rho_pooled"]] * 2, color=col, lw=1.4, zorder=3)
                if row == 0:
                    # mean success of the three runs
                    ax.plot([x[s] - 0.28, x[s] + 0.28], [float(np.mean(vals))] * 2, color=col, lw=1.4, zorder=3)
            ax.set_xticks(list(x.values()))
            ax.set_xticklabels(names, rotation=22, ha="right")
            ax.set_xlim(-0.55, len(names) - 0.45)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(disp, loc="left", fontweight="bold")
                lo = min(min(sk[s]["succ"]) for s in names) - 2
                hi = max(max(sk[s]["succ"]) for s in names) + 2
                ax.set_ylim(lo, hi)
            else:
                ax.set_yscale("log")
                allv = [v for s in names for v in sk[s]["rho_pool"]] + [1.0]
                ax.set_ylim(min(allv) / 1.06, max(allv) * 1.06)
                lo, hi = ax.get_ylim()
                ticks = [t for t in (0.9, 1.0, 1.1, 1.2, 1.4, 1.6) if lo <= t <= hi]
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{t:.1f}x" for t in ticks])
                ax.minorticks_off()
                ax.axhline(1.0, color="k", lw=0.5, ls="--", zorder=1)
    axs[0, 0].set_ylabel("test success (%)")
    axs[1, 0].set_ylabel("input tokens / seed's mean")
    handles = [plt.Line2D([], [], marker="o", ls="", color="0.45", ms=4),
               plt.Rectangle((0, 0), 1, 1, fc="0.5", alpha=0.15, lw=0),
               plt.Line2D([], [], ls="-", color="0.5", lw=1.2)]
    handles += [plt.Line2D([], [], marker=appb.OM[s], ls="", color=appb.OC[s], ms=4) for s in ("SkillOpt", "SkillAdam", "GEPA")]
    labels = ["seed run", "seed 3-run range", "run mean", "SkillOpt", "SkillAdam", "GEPA"]
    # matplotlib fills legend columns first; reorder so that the two rows read seed items, then the optimizers
    order = [0, 1, 2, 3, 4, 5]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 1.0), frameon=False, fontsize=7, ncol=6,
               handlelength=1.0, columnspacing=0.8, handletextpad=0.3)
    fig.save_figure(fig, "reruns")


def check_against_record(d):
    """Compare the plotted per-run success and ratios with the BACH-paper record paper/assets/data/mech/appB_reruns.md (baseline and seed rows)."""
    text = logger.find_existing(logger.OLD_DATA_DIRS, "appB_reruns.md").read_text()
    names = {b: n for b, n in appb.BENCH}
    seen = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[1] in ("GEPA", "SkillOpt", "SkillAdam", "seed"):
            seen[(cells[0], cells[1])] = (cells[3], cells[4], cells[5])
    rows = []
    allok = True
    for bench, disp in appb.BENCH:
        for s, sk in d["benches"][bench]["skills"].items():
            mine = (", ".join(f"{v:.1f}" for v in sk["succ"]), ", ".join(f"{v:.3f}" for v in sk["rho_pool"]),
                    f"x{sk['rho_pooled']:.2f}" if "rho_pooled" in sk else "")
            ref = seen[(disp, s)]
            ok = mine == ref
            allok = allok and ok
            rows.append([disp, s, mine[0], mine[1], mine[2], "equal" if ok else f"DIFF {ref}"])
    return rows, allok


def draw():
    """Draw mech_reruns, check it against the old record, write the audit record."""
    plt = appb.style()
    # Liberation Serif is TrueType; the CFF Nimbus Roman first in appb's list cannot be embedded as Type 42
    plt.rcParams["font.serif"] = ["Liberation Serif"]
    d = json.loads((dataloader.CACHE / "reruns.json").read_text())
    draw_reruns(plt, d)
    rows, allok = check_against_record(d)
    B = d["benches"]
    spread = [[disp, f"{B[b]['seed']['span']:.1f}", f"x{B[b]['seed']['spread_X']:.2f}", f"{100 * B[b]['seed']['flip']:.0f}%", f"{100 * B[b]['seed']['anyflip']:.0f}%"]
              for b, disp in appb.BENCH]
    table = logger.md_table(["Benchmark", "skill", "success per run (%)", "input tokens / pooled seed per run", "pooled ratio", "vs appB_reruns.md"], rows)
    table += "\n\nSeed spread\n\n" + logger.md_table(["Benchmark", "success span (pts)", "pairwise max ratio", "flip", "any-flip"], spread)
    table += f"\n\nAll plotted baseline/seed numbers equal paper/assets/data/mech/appB_reruns.md: {allok}."
    logger.write_record(
        "reruns",
        "Every test run of the seed and of the three baseline optimizers (GEPA, SkillOpt, SkillAdam): success (top row) and the run's geometric-mean input-token ratio against the pooled seed (bottom row); gray band = the seed's three runs. Derived from appB_reruns with the BACH entries removed.",
        "DeepSeek-V4.1-Flash in OpenCode; seed = native-seed/<bench>/native-20261003-trace-final plus rep-1 and rep-2; optimizers = <method>/<bench>/native-20261008-https (its own test evaluation) plus test-reps/<bench>__<method>__native-20261008-https rep 1/2. Test split, three executions per skill.",
        "Per-run success = mean over the common task set; rho = per-task geometric mean ratio of input tokens against the per-task geometric mean over the seed's three runs; pooled ratio pools the skill's own three runs the same way (mechanism/pooled.py). appb.py frozen-value asserts on the baseline cells ran unchanged.",
        table,
        ".venv/bin/python -m src.analysis.mech.reruns compute\npython3 -m src.analysis.mech.reruns draw",
        "Numbers: assets/cache/analysis/mech/reruns.json. Figure: paper/assets/fig/mech/mech_reruns.pdf/.svg. Drawing patch: appb.PINS copied without the BACH entries, appb.out_dir redirected to the mech cache, legend without BACH (ncol 3).")
    print("equal to appB_reruns.md:", allok)
    for r in rows:
        print(r)


if __name__ == "__main__":
    if sys.argv[1] == "compute":
        compute()
    else:
        draw()
