"""Rerun figure appB_reruns and its record.
  python -m aux.noise_and_domains.appb compute   -> CACHE/analysis/appb/reruns.json (every test run of the seed and of the
      returned skills: success and input-token ratio against the pooled seed)
  python -m aux.noise_and_domains.appb draw      -> fig/mech/appB_reruns and data/mech/appB_reruns.md
Reruns: the runs are pinned below (the https reruns `native-20261008-https`, the same runs as Table 1;
each skill's three test executions are the run's own evaluation plus the two under `test-reps/<bench>__<method>__<run>`); per-run rho_X is the
geometric-mean ratio against the pooled seed (per-task geometric mean over the seed's three runs), a skill's pooled rho_X pools its own runs
the same way (as in stability.pooled)."""
import json, math, sys
import numpy as np
from src.utils.dataloader import gmean

BENCH = [("tau3-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2")]
SEED_RUN = "native-20261003-trace-final"
PINS = {"tau3-bench": {"GEPA": ("gepa", "native-20261008-https"), "SkillOpt": ("skillopt", "native-20261008-https"), "SkillAdam": ("skilladam", "native-20261008-https")},
        "automationbench": {"GEPA": ("gepa", "native-20261008-https"), "SkillOpt": ("skillopt", "native-20261008-https"), "SkillAdam": ("skilladam", "native-20261008-https")},
        "terminal-bench-2-1": {"GEPA": ("gepa", "native-20261008-https"), "SkillOpt": ("skillopt", "native-20261008-https"), "SkillAdam": ("skilladam", "native-20261008-https")}}

# frozen values of the rerun record; compute asserts against them, nothing here re-derives them
FROZEN = dict(
    seed_succ={"tau3-bench": [78.7, 76.0, 74.7], "automationbench": [40.8, 49.2, 45.0], "terminal-bench-2-1": [72.2, 77.8, 77.8]},
    rerun_skill={"tau3-bench": "SkillAdam", "automationbench": "SkillOpt", "terminal-bench-2-1": "SkillOpt"},
    rerun_succ={"tau3-bench": [74.7, 77.3, 73.3], "automationbench": [43.3, 40.8, 40.8], "terminal-bench-2-1": [66.7, 66.7, 61.1]},
    seed_span={"tau3-bench": 4.0, "automationbench": 8.4, "terminal-bench-2-1": 5.6},
    seed_spread_X={"tau3-bench": 1.05, "automationbench": 1.14, "terminal-bench-2-1": 1.04},
    flip={"tau3-bench": 14, "automationbench": 12, "terminal-bench-2-1": 4},
    anyflip={"tau3-bench": 21, "automationbench": 18, "terminal-bench-2-1": 6},
    pooled_ratio={("SkillAdam", "tau3-bench"): 1.08, ("SkillOpt", "automationbench"): 1.26, ("SkillOpt", "terminal-bench-2-1"): 1.48},
    single_ratio={("SkillAdam", "automationbench"): 1.08},
)


def out_dir():
    from src.utils.dataloader import out_dir as od
    d = od("appb"); d.mkdir(parents=True, exist_ok=True); return d


# ---- compute ---------------------------------------------------------------------------------------------------------------------------------

def compute_reruns():
    import itertools
    from src.utils.dataloader import DS, SEED_TEXT, eval_file, final_skill, vid
    from src.analysis.table1 import rep_files
    from aux.noise_and_domains.stability import load, gm, gm_ratio
    out, raw = {}, []
    for bench, _ in BENCH:
        F = {"seed": [("main", eval_file(DS / "native-seed" / bench / SEED_RUN, "test", vid(SEED_TEXT)))]}
        for r in ("rep-1", "rep-2"):
            F["seed"].append((r, eval_file(DS / "native-seed" / bench / r, "test", vid(SEED_TEXT))))
        for disp, (meth, rn) in PINS[bench].items():
            run = DS / meth / bench / rn; t = final_skill(meth, run)
            assert t != SEED_TEXT, (bench, disp)
            reps = rep_files([run, DS / "test-reps" / f"{bench}__{meth}__{rn}"], vid(t))
            L = [("main" if rep == 0 else f"rep-{rep}", f) for rep, f in sorted(reps.items())]
            F[disp] = L
        D = {}
        for s, l in F.items():
            for rn, f in l: assert f, (bench, s, rn); D[(s, rn)] = load(f); raw.append(str(f))
        tasks = sorted(set.intersection(*[set(v) for v in D.values()]))
        seedruns = [k for k in D if k[0] == "seed"]
        succ = lambda k: 100 * float(np.mean([D[k][t]["succ"] for t in tasks]))
        pooled = {t: dict(X=gmean([D[k][t]["X"] for k in seedruns])) for t in tasks}
        skills = {}
        for s in sorted(set(k[0] for k in D)):
            ks = [k for k in D if k[0] == s]
            row = dict(runs=[k[1] for k in ks], succ=[round(succ(k), 1) for k in ks], rho_pool=[round(gm(D[k], pooled, tasks)[0], 4) for k in ks])
            if s != "seed":
                row["rho_pooled"] = round(gm_ratio([D[k] for k in ks], [D[k] for k in seedruns], tasks, "X")[0], 4)
            skills[s] = row
        pairs = [gm(D[a], D[b], tasks)[0] for a, b in itertools.combinations(seedruns, 2)]
        flips = [float(np.mean([D[a][t]["succ"] != D[b][t]["succ"] for t in tasks])) for a, b in itertools.combinations(seedruns, 2)]
        anyflip = float(np.mean([len(set(D[k][t]["succ"] for k in seedruns)) > 1 for t in tasks]))
        out[bench] = dict(n_tasks=len(tasks), skills=skills,
                          seed=dict(succ=skills["seed"]["succ"], rho_pool=skills["seed"]["rho_pool"],
                                    spread_X=round(math.exp(max(abs(math.log(x)) for x in pairs)), 4),
                                    flip=round(float(np.mean(flips)), 4), anyflip=round(anyflip, 4),
                                    span=round(max(skills["seed"]["succ"]) - min(skills["seed"]["succ"]), 1)))
        # frozen-value checks of the seed spreads
        sd = out[bench]["seed"]
        assert sd["succ"] == FROZEN["seed_succ"][bench], (bench, sd["succ"])
        assert sd["span"] == FROZEN["seed_span"][bench], (bench, sd["span"])
        assert round(sd["spread_X"], 2) == FROZEN["seed_spread_X"][bench], (bench, sd["spread_X"])
        assert round(100 * sd["flip"]) == FROZEN["flip"][bench] and round(100 * sd["anyflip"]) == FROZEN["anyflip"][bench], bench
        rs = FROZEN["rerun_skill"][bench]
        assert skills[rs]["succ"] == FROZEN["rerun_succ"][bench], (bench, skills[rs]["succ"])
        assert round(skills[rs]["rho_pooled"], 2) == FROZEN["pooled_ratio"][(rs, bench)], (bench, skills[rs]["rho_pooled"])
    assert round(out["automationbench"]["skills"]["SkillAdam"]["rho_pool"][0], 2) == FROZEN["single_ratio"][("SkillAdam", "automationbench")]
    f = out_dir() / "reruns.json"; f.write_text(json.dumps(dict(benches=out, raw=sorted(set(raw))), indent=1)); print("wrote", f)
    return out


def compute():
    compute_reruns()


# ---- draw ------------------------------------------------------------------------------------------------------------------------------------

def style():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    return plt


def save(fig, name):
    import matplotlib
    from src.utils.logger import paperout as PO
    (PO.FIG / "mech").mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "none"}): fig.savefig(PO.FIG / "mech" / f"{name}.svg")
    fig.savefig(PO.FIG / "mech" / f"{name}.pdf"); fig.savefig(out_dir() / f"{name}.png", dpi=200)


OC = {"SkillOpt": "#0072B2", "SkillAdam": "#D55E00", "GEPA": "#009E73"}
OM = {"SkillOpt": "o", "SkillAdam": "s", "GEPA": "^"}


def draw_reruns(plt, d):
    B = d["benches"]
    fig, axs = plt.subplots(2, 3, figsize=(6.3, 3.9))
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.14, top=0.86, hspace=0.62, wspace=0.3)
    for j, (bench, disp) in enumerate(BENCH):
        r = B[bench]; sk = r["skills"]; sd = r["seed"]
        names = ["seed"] + [s for s in ("GEPA", "SkillOpt", "SkillAdam") if s in sk]
        x = {s: i for i, s in enumerate(names)}
        for row, ax in enumerate(axs[:, j]):
            seedvals = sd["succ"] if row == 0 else sd["rho_pool"]
            ax.axhspan(min(seedvals), max(seedvals), color="0.5", alpha=0.15, lw=0, zorder=0)
            ax.axhline(float(np.mean(seedvals)), color="0.5", lw=0.8, ls="-", zorder=1)
            for s in names:
                vals = sk[s]["succ"] if row == 0 else sk[s]["rho_pool"]
                col = "0.45" if s == "seed" else OC[s]; mk = "o" if s == "seed" else OM[s]
                n = len(vals); xs = x[s] + (np.arange(n) - (n - 1) / 2) * min(0.22, 0.5 / max(n, 1))
                ax.scatter(xs, vals, s=16, marker=mk, color=col, zorder=4, lw=0)
                if s != "seed" and "rho_pooled" in sk[s] and row == 1:
                    ax.plot([x[s] - 0.28, x[s] + 0.28], [sk[s]["rho_pooled"]] * 2, color=col, lw=1.4, zorder=3)
                if row == 0:
                    ax.plot([x[s] - 0.28, x[s] + 0.28], [float(np.mean(vals))] * 2, color=col, lw=1.4, zorder=3)
            ax.set_xticks(list(x.values())); ax.set_xticklabels(names, rotation=22, ha="right")
            ax.set_xlim(-0.55, len(names) - 0.45); ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(disp, loc="left", fontweight="bold")
                lo = min(min(sk[s]["succ"]) for s in names) - 2; hi = max(max(sk[s]["succ"]) for s in names) + 2
                ax.set_ylim(lo, hi)
            else:
                ax.set_yscale("log")
                allv = [v for s in names for v in sk[s]["rho_pool"]] + [1.0]
                ax.set_ylim(min(allv) / 1.06, max(allv) * 1.06)
                lo, hi = ax.get_ylim()
                ticks = [t for t in (0.9, 1.0, 1.1, 1.2, 1.4, 1.6) if lo <= t <= hi]
                ax.set_yticks(ticks); ax.set_yticklabels([f"{t:.1f}x" for t in ticks]); ax.minorticks_off()
                ax.axhline(1.0, color="k", lw=0.5, ls="--", zorder=1)
    axs[0, 0].set_ylabel("test success (%)")
    axs[1, 0].set_ylabel("input tokens of the run\n/ the pooled seed")
    h = [plt.Line2D([], [], marker="o", ls="", color="0.45", ms=4), plt.Rectangle((0, 0), 1, 1, fc="0.5", alpha=0.15, lw=0),
         plt.Line2D([], [], ls="-", color="0.5", lw=1.2)] + [plt.Line2D([], [], marker=OM[s], ls="", color=OC[s], ms=4) for s in ("SkillOpt", "SkillAdam", "GEPA")]
    fig.legend(h, ["seed run", "seed's 3-run spread", "run mean / pooled ratio", "SkillOpt", "SkillAdam", "GEPA"],
               loc="upper center", bbox_to_anchor=(0.53, 0.99), frameon=False, fontsize=7.2, ncol=4, handlelength=1.2, columnspacing=1.0, handletextpad=0.3)
    save(fig, "appB_reruns")


def records(reruns):
    from src.utils.logger import paperout as PO
    mt = PO.mdtable
    PO.SPECS.setdefault("appB_reruns", dict(pub="mech",
        shows="every test run of the seed and of the returned skills: success (top) and the run's input-token ratio against the pooled seed (bottom); the gray band spans the seed's three runs",
        files="fig/mech/appB_reruns.pdf, fig/mech/appB_reruns.svg",
        notes=["The seed spreads (0.050, 0.129, 0.037 log input tokens; 0.052, 0.043, 0.054 log tool calls; 4.0, 8.4, 5.6 success points) come from the slopes `reruns` step of aux/cost_appendix (`seed_spread.json`); this figure recomputes per-run ratios from the same evaluation files and asserts them against those frozen values.",
               "The frozen asserts above cover only the baseline cells."]))
    B = reruns["benches"]; nm = dict(BENCH)
    rows = []
    for b, _ in BENCH:
        for s, sk in B[b]["skills"].items():
            rows.append([nm[b], s, ", ".join(sk["runs"]), ", ".join(f"{v:.1f}" for v in sk["succ"]),
                         ", ".join(f"{v:.3f}" for v in sk["rho_pool"]), f"x{sk['rho_pooled']:.2f}" if "rho_pooled" in sk else ""])
        sd = B[b]["seed"]
        rows.append([nm[b], "seed spread", "3 runs", f"span {sd['span']:.1f} pts", f"pairwise max x{sd['spread_X']:.2f}", f"flip {100*sd['flip']:.0f}%, any-flip {100*sd['anyflip']:.0f}%"])
    PO.record("appB_reruns", "fig", result=mt(["Benchmark", "skill", "runs", "success per run (%)", "input tokens / pooled seed per run", "pooled ratio / flips"], rows)
              + "\n\nPer-run success and geometric-mean input-token ratio against the pooled seed (per-task geometric mean over the seed's three runs), all on the test split; a skill's pooled ratio pools its own runs the same way.",
              source=["aux/noise_and_domains/appb.py", "aux/noise_and_domains/stability.py"],
              commands=["python -m aux.noise_and_domains.appb compute", "python -m aux.noise_and_domains.appb draw"], raw=reruns["raw"])


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m aux.noise_and_domains.appb compute|draw")
    if sys.argv[1] == "compute": compute(); return
    plt = style()
    r = json.loads((out_dir() / "reruns.json").read_text())
    draw_reruns(plt, r); records(r)


if __name__ == "__main__": main()
