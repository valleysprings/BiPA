"""Appendix B figures (paper/assets/fig/mech/appB_reruns, appB_nullsim + records).
  <venv python> -m src.analysis.aux.appb compute   -> assets/cache/analysis/appb/reruns.json (every test run of the seed and of the
      returned skills: success and input-token ratio against the pooled seed) and nullsim.json (zero-effect dev-gain distributions)
  <system python3 + matplotlib> -m src.analysis.aux.appb draw -> the two figures and the audit records
  (run from the bipa/ root with the bach-harness venv python; the draw step needs a system python3 with matplotlib)
Reruns: the runs behind the B.2 prose are pinned below (the https reruns `native-20261008-https`, the same runs as Table 1;
each skill's three test executions are the run's own evaluation plus the two under `test-reps/<bench>__<method>__<run>`); per-run rho_X is the
geometric-mean ratio against the pooled seed (per-task geometric mean over the seed's three runs), a skill's pooled rho_X pools its own runs
the same way (as in mechanism/pooled.py).
Null simulation: the dev-selection replay of aux/nullmodel.py (SkillOpt strict-improvement rule, GEPA argmax rule) re-drawn on the
same seed dev executions to obtain the distributions; the test-fitted Beta priors, candidate counts, baselines and observed gains are
frozen from assets/cache/analysis/nullmodel/out.json / out_c4.json, the run behind the Appendix B.3 prose, and the re-drawn means are
asserted against it. 200,000 draws per setting, fixed seed."""
import json, math, sys
from pathlib import Path
import numpy as np
from src.utils.dataloader import gmean

H = Path("/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation")
OLD = H / "assets/cache/analysis/nullmodel"   # the zero-effect run behind the B.3 prose (out.json = fitted, out_c4.json = concentration 4)
BENCH = [("tau3-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2")]
VARIANTS = (("fitted", "out.json"), ("conc4", "out_c4.json"))
NS, SEED = 200000, 20261008
SEED_RUN = "native-20261003-trace-final"
PINS = {"tau3-bench": {"GEPA": ("gepa", "native-20261008-https"), "SkillOpt": ("skillopt", "native-20261008-https"), "SkillAdam": ("skilladam", "native-20261008-https")},
        "automationbench": {"GEPA": ("gepa", "native-20261008-https"), "SkillOpt": ("skillopt", "native-20261008-https"), "SkillAdam": ("skilladam", "native-20261008-https")},
        "terminal-bench-2-1": {"GEPA": ("gepa", "native-20261008-https"), "SkillOpt": ("skillopt", "native-20261008-https"), "SkillAdam": ("skilladam", "native-20261008-https")}}

# frozen paper values (paper/draft2mech/appendix/b-noise.tex B.2/B.3); compute asserts against them, nothing here re-derives them
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
    obs_gain={("SkillOpt", "tau3-bench"): 5, ("GEPA", "tau3-bench"): 4, ("SkillOpt", "automationbench"): 3, ("SkillOpt", "terminal-bench-2-1"): 2},
    base={"tau3-bench": 18, "automationbench": 17, "terminal-bench-2-1": 4},
    Egain_fitted={"tau3-bench": 1.8, "automationbench": 1.5, "terminal-bench-2-1": 0.5},
    Egain_conc4={"tau3-bench": 2.5, "automationbench": 2.7, "terminal-bench-2-1": 1.6},
    Egain_cond={"tau3-bench": 3.9, "automationbench": 1.9, "terminal-bench-2-1": 1.0},
)


def out_dir():
    from src.utils.dataloader import out_dir as od
    d = od("appb"); d.mkdir(parents=True, exist_ok=True); return d


# ---- compute ---------------------------------------------------------------------------------------------------------------------------------

def compute_reruns():
    import itertools
    from src.utils.dataloader import DS, SEED_TEXT, eval_file, final_skill, vid
    from src.analysis.main.table1 import rep_files
    from src.analysis.aux.stability import load, gm
    from src.analysis.aux.stability import gm_ratio
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
        # frozen-value checks (paper/draft2mech/appendix/b-noise.tex B.2; the tab:lever-test caption cites the same spreads)
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


def dev_execs(bench, old):
    """the distinct seed dev executions of the frozen nullmodel run: one per (total, request-id signature), matched by the frozen totals."""
    import glob
    from src.utils.dataloader import EX, SEED_TEXT, vid
    from src.analysis.aux.nullmodel import load_exec
    want = sorted(old["V_seed_counts"])
    groups = {}
    for f in sorted(glob.glob(str(EX / "deepseek-flash" / "*" / bench / "*" / "results/evaluations/dev" / f"{vid(SEED_TEXT)[:12]}-*/evaluation.json"))):
        x = load_exec(f)
        groups.setdefault((sum(x["succ"].values()), x["sig"]), x)
    groups = {k: x for k, x in groups.items() if k[0] in want and len({t for t, _ in x["succ"]}) == old["V_tasks"]}
    assert sorted(t for t, _ in groups) == want, (bench, sorted(groups), want)
    return [groups[k] for k in sorted(groups)]


def compute_nullsim():
    from src.analysis.aux.nullmodel import tally
    out, raw = {}, []
    for bench, _ in BENCH:
        old0 = json.loads((OLD / "out.json").read_text())[bench]
        ex = dev_execs(bench, old0); raw += [e["file"] for e in ex]
        tasks, K, n, mult = tally(ex)
        assert sorted(int(sum(e["succ"].values())) for e in ex) == sorted(old0["V_seed_counts"]) and len(tasks) == old0["V_tasks"]
        rb = out[bench] = dict(nV=int(mult.sum()), obs=old0["obs"], variants={})
        for vi, (vn, fn) in enumerate(VARIANTS):
            o = json.loads((OLD / fn).read_text())[bench]
            a, bb = o["beta_prior_a_b"]; c0 = a + bb
            m = float(K.sum() / n.sum()); aV, bV = m * c0, (1 - m) * c0
            rng = np.random.default_rng([SEED, 7, [b for b, _ in BENCH].index(bench), vi])
            idx = np.repeat(np.arange(len(mult)), mult)
            pV = rng.beta(K + aV, n - K + bV, size=(NS, len(K)))
            runsum = lambda p: (rng.random((NS, len(idx))) < p[:, idx]).sum(1)
            rv = rb["variants"][vn] = dict(concentration=c0, dev_prior=[aV, bV], methods={})
            for meth in ("SkillOpt", "GEPA"):
                Kc = o["V_uncond"][meth]["K"]
                if not Kc: continue
                s0 = runsum(pV)
                cand = np.stack([runsum(pV) for _ in range(Kc)], 1)
                if meth == "SkillOpt":
                    best = s0.copy()
                    for i in range(Kc): best = np.where(cand[:, i] > best, cand[:, i], best)
                    gU = best - s0
                    s0c = np.full(NS, o["obs"]["V_base"]); bestc = s0c.copy()
                    for i in range(Kc): bestc = np.where(cand[:, i] > bestc, cand[:, i], bestc)
                    gC = bestc - s0c
                else:
                    gU = np.concatenate([s0[:, None], cand], 1).max(1) - s0
                    s0c = np.full(NS, o["obs"]["V_base"]); gC = np.concatenate([s0c[:, None], cand], 1).max(1) - s0c
                obs = o["V_uncond"][meth]["obs_gain"]
                rv["methods"][meth] = {k: dict(hist=np.bincount(g, minlength=int(g.max()) + 1).tolist(), mean=float(g.mean()),
                                               P_ge_obs=float((g >= obs).mean()), obs_gain=int(obs))
                                       for k, g in (("uncond", gU), ("cond", gC))}
                # frozen-value checks against the cached run behind the prose
                for k, g in (("uncond", gU), ("cond", gC)):
                    ref = o["V_uncond" if k == "uncond" else "V_cond_on_obs_baseline"][meth]["E_gain_tasks"]
                    assert abs(g.mean() - ref) < 0.02 + 3 * float(g.std() / NS ** .5), (bench, vn, meth, k, g.mean(), ref)
                assert obs == FROZEN["obs_gain"].get((meth, bench), 0), (bench, meth, obs)
            eU = rv["methods"]["SkillOpt"]["uncond"]["mean"]; eC = rv["methods"]["SkillOpt"]["cond"]["mean"]
            assert round(eU, 1) == FROZEN["Egain_fitted" if vn == "fitted" else "Egain_conc4"][bench], (bench, vn, eU)
            if vn == "fitted": assert round(eC, 1) == FROZEN["Egain_cond"][bench], (bench, eC)
        assert rb["obs"]["V_base"] == FROZEN["base"][bench]
    raw += [str(OLD / fn) for _, fn in VARIANTS]
    f = out_dir() / "nullsim.json"; f.write_text(json.dumps(dict(benches=out, NS=NS, seed=SEED, raw=sorted(set(raw))), indent=1)); print("wrote", f)
    return out


def compute():
    compute_reruns(); compute_nullsim()


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


def draw_nullsim(plt, d):
    B = d["benches"]
    VC = {"fitted": "#0072B2", "conc4": "#E69F00"}
    VL = {"fitted": "fitted difficulty spread", "conc4": "Beta concentration 4"}
    fig, axs = plt.subplots(2, 3, figsize=(6.3, 3.9))
    fig.subplots_adjust(left=0.115, right=0.99, bottom=0.11, top=0.88, hspace=0.62, wspace=0.3)
    for j, (bench, disp) in enumerate(BENCH):
        r = B[bench]; xmax = 0
        for i, cond in enumerate(("uncond", "cond")):
            ax = axs[i, j]
            em = {}
            for vn, _ in VARIANTS:
                m = r["variants"][vn]["methods"].get("SkillOpt")
                if not m: continue
                em[vn] = m[cond]["mean"]
                h = np.array(m[cond]["hist"], float); h = h / h.sum()
                xs = np.arange(len(h)); xmax = max(xmax, xs[h > 0].max())
                ax.step(np.append(xs, xs[-1] + 1) - 0.5, np.append(h, 0), where="post", color=VC[vn], lw=1.2, label=VL[vn])
                if vn == "fitted": ax.fill_between(np.append(xs, xs[-1] + 1) - 0.5, np.append(h, 0), step="post", color=VC[vn], alpha=0.18, lw=0)
            for k, (vn, _) in enumerate(VARIANTS):
                if vn in em: ax.text(0.97, 0.97 - 0.11 * k, f"mean {em[vn]:.1f}", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=VC[vn])
            ob = r["obs"]
            ax.axvline(ob["so_gain"], color="k", lw=1.0, ls="-", zorder=5)
            ax.text(ob["so_gain"], 0.98, f" +{ob['so_gain']} SkillOpt", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=7, rotation=0)
            if ob["gepa_gain"]:
                ax.axvline(ob["gepa_gain"], color="k", lw=1.0, ls="--", zorder=5)
                ax.text(ob["gepa_gain"] - 0.15, 0.72, f"+{ob['gepa_gain']} GEPA ", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(-0.6, xmax + 0.8); ax.set_ylim(bottom=0)
            if i == 0: ax.set_title(disp, loc="left", fontweight="bold")
            if i == 1: ax.set_xlabel("dev-set gain (tasks)")
            ax.set_xticks(range(0, xmax + 1, 2 if xmax > 6 else 1))
        axs[0, j].set_xlim(-0.6, xmax + 0.8); axs[1, j].set_xlim(-0.6, xmax + 0.8)
    for i, lab in enumerate(("any baseline draw", "baseline at its lowest draw")):
        axs[i, 0].set_ylabel("P(gain)")
        fig.text(0.008, 0.68 if i == 0 else 0.30, lab, rotation=90, va="center", ha="center", fontsize=8)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper right", bbox_to_anchor=(0.99, 0.995), frameon=False, fontsize=7, ncol=2, handlelength=1.4)
    save(fig, "appB_nullsim")


def records(reruns, nullsim):
    from src.utils.logger import paperout as PO
    mt = PO.mdtable
    PO.SPECS.setdefault("appB_reruns", dict(pub="mech",
        paper=[("fig:appb-reruns (Appendix B.2)", "appendix/b-noise.tex", r"label\{fig:appb-reruns\}",
                "every test run of the seed and of the returned skills: success (top) and the run's input-token ratio against the pooled seed (bottom); the gray band spans the seed's three runs")],
        files="fig/mech/appB_reruns.pdf, fig/mech/appB_reruns.svg",
        notes=["The prose's seed spreads (0.050, 0.129, 0.037 log input tokens; 0.052, 0.043, 0.054 log tool calls; 4.0, 8.4, 5.6 success points), also cited by the tab:lever-test caption, come from `analysis/aux/slopes.py` (`reruns` subcommand, `seed_spread.json`); this figure recomputes per-run ratios from the same evaluation files and asserts them against the frozen prose values.",
               "The frozen asserts above cover only the baseline cells (BiPA reports no BACH rows)."]))
    PO.SPECS.setdefault("appB_nullsim", dict(pub="mech",
        paper=[("fig:appb-nullsim (Appendix B.3)", "appendix/b-noise.tex", r"label\{fig:appb-nullsim\}",
                "distribution of the dev-set gain when every candidate equals the seed (SkillOpt's selection rule), per benchmark, under the fitted task-difficulty spread and Beta concentration 4; vertical lines mark the observed gains")],
        files="fig/mech/appB_nullsim.pdf, fig/mech/appB_nullsim.svg",
        notes=["Priors, candidate counts, baselines and observed gains are frozen from `assets/cache/analysis/nullmodel/out.json` / `out_c4.json`, the run behind the B.3 prose; the replay is re-drawn (200,000 draws, seed 20261008) only to obtain the distributions and its means are asserted against that run within Monte Carlo error.",
               "The newer `assets/cache/analysis/submission-20261008/nullmodel/` output pins a different run set (several cells still pending) and is not the source of these numbers.",
               "GEPA's argmax rule with its own candidate count gives means within 0.1 dev tasks of the SkillOpt rule on every panel; it is not drawn."]))
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
              source=["bipa/src/analysis/aux/appb.py", "bipa/src/analysis/aux/reproducibility.py", "bipa/src/analysis/aux/pooled.py"],
              commands=["<bach-harness venv python> -m src.analysis.aux.appb compute   # from bipa/", "<system python3 + matplotlib> -m src.analysis.aux.appb draw   # from bipa/"], raw=reruns["raw"])
    B = nullsim["benches"]
    rows = [[nm[b], vn, m, f"{x['methods'][m]['uncond']['mean']:.2f}", f"{x['methods'][m]['cond']['mean']:.2f}",
             f"+{x['methods'][m]['uncond']['obs_gain']}", f"{x['methods'][m]['uncond']['P_ge_obs']:.3f}", f"{x['methods'][m]['cond']['P_ge_obs']:.3f}", B[b]["obs"]["V_base"]]
            for b, _ in BENCH for vn, x in B[b]["variants"].items() for m in ("SkillOpt", "GEPA") if m in x["methods"]]
    PO.record("appB_nullsim", "fig", result=mt(["Benchmark", "variant", "rule", "E[gain] any baseline", "E[gain] baseline at lowest draw", "observed gain", "P(gain >= obs) any", "P(gain >= obs) lowest", "observed baseline"], rows)
              + "\n\nZero-effect dev-selection replay: Beta posterior per task from the seed's distinct executions (three on the dev set of tau3-bench, two on AutoBench and TB2; test-fitted priors frozen from the cached nullmodel run), candidates re-drawn, each optimizer's acceptance rule over its own candidate count; 200,000 draws per setting (seed 20261008).",
              source=["bipa/src/analysis/aux/appb.py", "bipa/src/analysis/aux/nullmodel.py"],
              commands=["<bach-harness venv python> -m src.analysis.aux.appb compute   # from bipa/", "<system python3 + matplotlib> -m src.analysis.aux.appb draw   # from bipa/"], raw=nullsim["raw"])


def main():
    if sys.argv[1] == "compute": compute(); return
    plt = style()
    r = json.loads((out_dir() / "reruns.json").read_text()); n = json.loads((out_dir() / "nullsim.json").read_text())
    draw_reruns(plt, r); draw_nullsim(plt, n); records(r, n)


if __name__ == "__main__": main()
