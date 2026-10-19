"""Appendix C figures (paper/assets/fig/mech/appC_slopes, appC_attribution, appC_price + records) and the fig1 data stage they rest on. DeepSeek test episodes of the seed and the three baseline optimizers (attribution and price).
  <venv python> -m src.analysis.mechanism.appc fig1 [--models deepseek-flash] [--out DIR]   -> assets/cache/analysis/fig1/cache.json (merged from fig1.py's
      compute stage: within-task regressions of log input/output/calls on log turns per benchmark, and each optimized skill's four-term split against the seed)
  <venv python> -m src.analysis.mechanism.appc compute   -> assets/cache/analysis/appc/price.json (raw input-token ratio vs cache-priced ratio per skill)   (run from bipa/)
                                                          and assets/cache/analysis/appc/attr.json (the four-term attribution rows, from the pinned https reruns below)
  <system python3 + matplotlib> -m src.analysis.mechanism.appc draw               -> the three figures and records (appC_slopes needs the fig1 subcommand first)
  (fig1.py's draw stage is not carried over: it only wrote a preview fig1.png to the output dir, nothing to paper/.)
Price = uncached input + 0.1 * cached input + 4 * output tokens (cached = tokens served from cache from request 2 on), per task, geometric mean of skill/seed over the Table 1 common task set.
The baseline attribution rows are pinned to the https reruns (ATTR_RUNS), the same runs that feed the avg@3 Table 1. check() asserts the claims."""
import argparse, collections, json, math, sys
from pathlib import Path
import numpy as np
from src.utils.dataloader import gmean

H = Path("/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation")
FIG1 = H / "assets/cache/analysis/fig1"
F1 = FIG1 / "cache.json"
OUT = H / "assets/cache/analysis/appc"
BENCH = [("tau3-bench", r"$\tau^3$-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench", "AutoBench"), ("terminal-bench-2-1", "TB2", "TB2")]  # (key, display, short); shared with analysis/mech/
OPT = [("skillopt", "SkillOpt"), ("skilladam", "SkillAdam"), ("gepa", "GEPA")]   # baselines only (BiPA reports no BACH rows)
DISP = {"SkillOpt": "skillopt", "SkillAdam": "skilladam", "GEPA": "gepa"}

# The attribution (Figure appC_attribution) is read on the https reruns, the same runs that feed the avg@3 Table 1;
# all nine optimizer cells returned a new skill in these runs, so every (method, benchmark) pair has a row.
SEED_RUN = "native-20261003-trace-final"
ATTR_RUNS = {(m, b): "native-20261008-https" for m in ("skillopt", "skilladam", "gepa") for b, *_ in BENCH}


def compute_attr():
    """The four-term split of the slopes checks' skill_fit per optimized skill against the seed, on the pinned https reruns (the fig1 subcommand's math)."""
    from src.utils import dataloader as C
    from src.analysis.mechanism.slopes import metrics, episodes
    skills, raw = [], set()
    for meth, disp in OPT:
        for bench, *_ in BENCH:
            if (meth, bench) not in ATTR_RUNS: continue
            seedrun = C.EX / "deepseek-flash" / "native-seed" / bench / SEED_RUN
            sf = C.eval_file(seedrun, "test", C.vid(C.SEED_TEXT)); seed_rows = C.evaluation(seedrun, "test", C.vid(C.SEED_TEXT))[0]
            run = C.EX / "deepseek-flash" / meth / bench / ATTR_RUNS[(meth, bench)]
            t = C.final_skill(meth, run); f = C.eval_file(run, "test", C.vid(t)); own_rows = C.evaluation(run, "test", C.vid(t))[0]
            common = {k[0] for k, r in own_rows.items() if (k[0], 0) in seed_rows and r["success"] is not None and seed_rows[(k[0], 0)]["success"] is not None
                      and all(x and x > 0 for x in (r["N"], r["X"], seed_rows[(k[0], 0)]["N"], seed_rows[(k[0], 0)]["X"]))}
            raw |= {str(sf), str(f)}
            S = episodes(sf); Cs = episodes(f); rows = []
            for task, ec in Cs.items():
                a = metrics(S[task]) if task in S else None; b = metrics(ec)
                if a and b and task in common: rows.append((task, a, b))
            L = lambda k: np.array([math.log(b[k] / a[k]) for _, a, b in rows])
            dR, dN, dh, dX, dF = L("R"), L("N"), L("h"), L("X"), L("F")
            dm = dN - dR
            sF = np.array([(a["sF"] + b["sF"]) / 2 for _, a, b in rows]); sH = 1 - sF; Rm = np.array([(a["R"] + b["R"]) / 2 for _, a, b in rows]); k = 1 + sH * Rm / (Rm - 1)  # exact e_R
            seg = dict(more_actions=float(np.mean(k * dN)), more_requests=float(np.mean(-k * dm)), longer_content=float(np.mean(sH * dh)), longer_skill=float(np.mean(sF * dF)))
            skills.append(dict(model="deepseek-flash", bench=bench, method=disp, n=len(rows), run=run.name, **seg,
                               segments_sum=sum(seg.values()), dlogX=float(dX.mean()), ratio=gmean(np.exp(dX))))
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "attr.json").write_text(json.dumps(dict(skills=skills, raw=sorted(raw)), indent=1))
    check(skills)
    for s in skills: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "model"})


DOM = {("GEPA", "tau3-bench"): "more_requests", ("SkillAdam", "tau3-bench"): "more_actions", ("SkillOpt", "tau3-bench"): "longer_skill",
       ("SkillOpt", "automationbench"): "more_requests", ("SkillAdam", "automationbench"): "more_requests", ("GEPA", "automationbench"): "more_requests",
       ("SkillOpt", "terminal-bench-2-1"): "more_actions", ("SkillAdam", "terminal-bench-2-1"): "more_actions", ("GEPA", "terminal-bench-2-1"): "more_actions"}


def check(skills):
    """The claims the figure must support, recomputed on the https reruns (2026-10-08); run by compute_attr and by draw.
    The claims are about the baseline rows (DOM)."""
    by = {(s["method"], s["bench"]): s for s in skills}
    assert set(DOM) <= set(by) and len(by) == len(skills), f"rows: {sorted(by)}"
    for key, dom in DOM.items():
        got = max(TC, key=lambda k: by[key][k])
        assert got == dom, f"{key}: largest term {got}, expected {dom}"
    r = by[("SkillOpt", "tau3-bench")]
    assert abs(r["dlogX"] - 0.167) <= 0.005, f"SkillOpt tau3 total {r['dlogX']:.4f}, expected 0.167 log units"
    for s in skills:
        assert abs(sum(s[k] for k in TC) - s["dlogX"]) <= 0.01, f"{s['method']} {s['bench']}: terms sum {sum(s[k] for k in TC):.4f} vs measured {s['dlogX']:.4f}"
        if (s["method"], s["bench"]) in DOM and s["bench"] != "tau3-bench":
            share = s["longer_skill"] / s["dlogX"]
            assert 0 - 0.005 <= share <= 0.25 + 0.005, f"{s['method']} {s['bench']}: skill text is {share:.1%} of the change, expected 0--25%"


def compute():
    from src.utils import dataloader as C
    from src.analysis.mechanism.slopes import build_rows, derive
    from src.analysis.mechanism.slopes import METHODS3
    from src.graph import episode as W
    # The https reruns, the same runs that feed the avg@3 Table 1.
    PRICE_RUNS = {(m, b): "native-20261008-https" for m in ("gepa", "skillopt", "skilladam") for b, *_ in BENCH}
    def load(f):
        d = json.load(open(f)); raw = {}
        for e in d["episodes"]: raw.setdefault((e["task_id"], e.get("repetition", 0)), e)
        out = {}
        for we in W.load(f):
            e = raw.get((we.task, 0)) or next((v for kk, v in raw.items() if kk[0] == we.task), None)
            seen = set(); reqs = []
            for r in e["requests"]:
                if r["request_id"] not in seen: seen.add(r["request_id"]); reqs.append(r)
            if len(reqs) != we.R or we.R < 2: continue
            cache = np.array([min((r.get("captured_usage") or {}).get("cache_tokens") or 0, r["input_tokens"]) for r in reqs], float)
            out[we.task] = (dict(R=we.R, X=float(we.X), res=float(cache[1:].sum()), out_tot=float(sum(r["output_tokens"] for r in reqs))))
        return out
    price = lambda r, c=0.1, w=4: (r["X"] - r["res"]) + c * r["res"] + w * r["out_tot"]
    res = []
    for bench, *_ in BENCH:
        seedrun = C.EX / "deepseek-flash" / "native-seed" / bench / SEED_RUN; sf = str(C.eval_file(seedrun, "test", C.vid(C.SEED_TEXT)))
        seed_rows = C.evaluation(seedrun, "test", C.vid(C.SEED_TEXT))[0]
        S = load(sf)
        for disp, meth in METHODS3.items():
            if (meth, bench) not in PRICE_RUNS: continue
            run = C.EX / "deepseek-flash" / meth / bench / PRICE_RUNS[(meth, bench)]; t = C.final_skill(meth, run)
            if t == C.SEED_TEXT: continue
            own_rows = C.evaluation(run, "test", C.vid(t))[0]
            common = {k[0] for k, r in own_rows.items() if (k[0], 0) in seed_rows and r["success"] is not None and seed_rows[(k[0], 0)]["success"] is not None
                      and all(x and x > 0 for x in (r["N"], r["X"], seed_rows[(k[0], 0)]["N"], seed_rows[(k[0], 0)]["X"]))}
            O = load(C.eval_file(run, "test", C.vid(t)))
            ts = [k for k in O if k in S and k in common and S[k]["R"] >= 2 and O[k]["R"] >= 2]
            lx = np.array([math.log(O[k]["X"] / S[k]["X"]) for k in ts]); lp = np.array([math.log(price(O[k]) / price(S[k])) for k in ts])
            lo = np.array([math.log(O[k]["out_tot"] / S[k]["out_tot"]) for k in ts if S[k]["out_tot"] > 0 and O[k]["out_tot"] > 0])
            res.append(dict(bench=bench, method=disp, n=len(ts), raw=gmean(np.exp(lx)), priced=gmean(np.exp(lp)),
                            raw_sum=float(sum(O[k]["X"] for k in ts) / sum(S[k]["X"] for k in ts)), priced_sum=float(sum(price(O[k]) for k in ts) / sum(price(S[k]) for k in ts)),
                            out_share=float(sum(O[k]["out_tot"] for k in ts) / sum(O[k]["out_tot"] + O[k]["X"] for k in ts)),
                            cache_share=float(sum(O[k]["res"] for k in ts) / sum(O[k]["X"] for k in ts))))
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "price.json").write_text(json.dumps(res, indent=1))
    for r in res: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    compute_attr()


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
    fig.savefig(PO.FIG / "mech" / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=200)


BC = {"tau3-bench": "#0072B2", "automationbench": "#D55E00", "terminal-bench-2-1": "#009E73"}
TC = {"more_actions": "#0072B2", "more_requests": "#E69F00", "longer_content": "#CC79A7", "longer_skill": "#999999"}


QUANT = [("input", "input tokens"), ("output", "output tokens"), ("calls", "calls")]


def draw_slopes(plt, c):
    reg = c["reg"]; pts = {b: {qn: (np.array(v[0]), np.array(v[1])) for qn, v in d.items()} for b, d in c["pts"].items()}
    fig, axs = plt.subplots(3, 3, figsize=(6.3, 5.6), sharex=True)
    fig.subplots_adjust(left=0.115, right=0.99, bottom=0.06, top=0.95, wspace=0.26, hspace=0.42)
    rng = np.random.default_rng(0)
    xx = np.exp(np.array([-0.6, 0.6]))
    for row, (qn, qlab) in enumerate(QUANT):
        ally = np.concatenate([pts[b][qn][1] for b, *_ in BENCH] + [np.log(xx) * max(reg[b][qn]["slope"], 1.0) for b, *_ in BENCH])
        lo, hi = np.exp(ally.min()), np.exp(ally.max()); span = hi / lo; lo, hi = lo / span ** 0.06, hi * span ** 0.06
        ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8) if lo <= t <= hi]
        for ax, (b, disp, *_) in zip(axs[row], BENCH):
            xs, ys = pts[b][qn]; r = reg[b][qn]
            sel = rng.choice(len(xs), min(len(xs), 250), replace=False)
            ax.scatter(np.exp(xs[sel]), np.exp(ys[sel]), s=3, color=BC[b], alpha=0.25, lw=0, rasterized=True)
            ax.plot(xx, xx, color="k", ls="--", lw=0.8, label="$k=1$")
            ax.plot(xx, xx ** r["slope"], color=BC[b], lw=1.6, label=f"$k={r['slope']:.2f}$")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(np.exp(-0.7), np.exp(0.7)); ax.set_ylim(lo, hi)
            ax.set_xticks([0.5, 1, 2]); ax.set_xticklabels(["0.5x", "1x", "2x"]); ax.minorticks_off()
            ax.legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.4, borderpad=0.1, labelspacing=0.25)
            if row == 0: ax.set_title(disp, loc="left", fontweight="bold")
            if row == 2: ax.set_xlabel("turns vs. the task's mean")
        for ax in axs[row]: ax.set_yticks(ticks)
        axs[row][0].set_yticklabels([f"{t:g}x" for t in ticks])
        for ax in axs[row][1:]: ax.tick_params(axis="y", labelleft=False)
        axs[row][0].set_ylabel(f"{qlab} vs.\nthe task's mean")
    save(fig, "appC_slopes")


COLS = ["more_actions", "more_requests", "longer_content", "longer_skill", "dlogX"]
HEAD = ["more\nactions", "turn-\nsplitting", "content\nper turn", "skill\ntext", "total\n(measured)"]


def draw_attr(plt, skills):
    check(skills)
    bshort = {"tau3-bench": r"$\tau^3$", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}
    M = np.array([[s[k] for k in COLS] for s in skills])
    methods = [s["method"] for s in skills]
    bounds = [i for i in range(1, len(skills)) if methods[i] != methods[i - 1]]
    groups = [(m, sum(i for i, x in enumerate(methods) if x == m) / methods.count(m)) for m in dict.fromkeys(methods)]
    fig, ax = plt.subplots(figsize=(6.3, 2.45 + 0.15 * (len(skills) - 9))); fig.subplots_adjust(left=0.185, right=0.895, bottom=0.02, top=1 - 0.2 * 2.45 / (2.45 + 0.15 * (len(skills) - 9)))
    v = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(M[i, j]) > 0.62 * v else "k", fontweight="bold" if j == 4 else "normal")
    ax.set_xticks(range(5)); ax.set_xticklabels(HEAD); ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_yticks(range(len(skills))); ax.set_yticklabels([bshort[s["bench"]] for s in skills]); ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True); ax.set_yticks(np.arange(-0.5, len(skills), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4); ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    for y in [b - 0.5 for b in bounds]: ax.plot([-0.5, 4.5], [y, y], color="k", lw=1.2, clip_on=False)
    for name, yc in groups:
        ax.text(-1.05, yc, name, ha="right", va="center", fontsize=8.5, fontweight="bold", clip_on=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[-0.4, 0, 0.4])
    cb.ax.set_yticklabels(["-0.4", "0", "+0.4"]); cb.outline.set_visible(False); cb.ax.tick_params(length=0)
    cb.set_label("change in log input tokens\nvs. the seed (red: more)", fontsize=7)
    save(fig, "appC_attribution")


def draw_price(plt, p):
    fig, ax = plt.subplots(figsize=(3.4, 3.1)); fig.subplots_adjust(left=0.16, right=0.98, bottom=0.15, top=0.97)
    lo = min(0.97, min(min(r["raw"], r["priced"]) for r in p) - 0.04); hi = max(1.65, max(max(r["raw"], r["priced"]) for r in p) + 0.04)
    ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=0.8); ax.axhline(1, color="0.7", lw=0.5); ax.axvline(1, color="0.7", lw=0.5)
    mk = {"SkillOpt": "o", "SkillAdam": "s", "GEPA": "^"}
    for r in p:
        ax.scatter(r["raw"], r["priced"], s=34, marker=mk[r["method"]], color=BC[r["bench"]], zorder=4, lw=0)
    for b, d, *_ in BENCH: ax.scatter([], [], s=34, color=BC[b], label=d)
    for m, k in mk.items(): ax.scatter([], [], s=34, marker=k, color="0.35", label=m)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, handletextpad=0.1, labelspacing=0.25)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("input tokens, optimized / seed"); ax.set_ylabel("price with cached input at 0.1x\nand output at 4x, optimized / seed")
    ax.text(hi - 0.01, lo + 0.01, "dashed: both equal", ha="right", va="bottom", fontsize=7.5)
    save(fig, "appC_price")


def records(c, p, attr):
    from src.utils.logger import paperout as PO
    mt = PO.mdtable
    nm = {"tau3-bench": "tau3-bench", "automationbench": "AutoBench", "terminal-bench-2-1": "TB2"}
    rows = [[nm[b], qn, f"{v['slope']:.2f}", f"{v['se']:.2f}", v["n"], v["n_tasks"], f"{v['r2']:.2f}"] for b, *_ in BENCH for qn, _ in QUANT for v in [c["reg"][b][qn]]]
    PO.record("appC_slopes", "fig", result=mt(["Benchmark", "quantity", "slope k", "SE", "episodes", "tasks", "R^2 (within)"], rows)
              + "\n\nWithin-task slope of log y on log turns (per-task means removed), y = input tokens / output tokens / calls (executed tool calls), seed and GEPA/SkillOpt/SkillAdam episodes pooled; dashed line is slope 1. Axes show the ratio to the task's geometric mean. Episodes with zero or missing output tokens are dropped from the output row (n reports the actual count).", source=["bipa/src/analysis/mechanism/appc.py", "bipa/src/analysis/mechanism/fig1.py"],
              commands=[".venv/bin/python -m src.analysis.mechanism.appc fig1   # from bipa/", ".venv/bin/python -m src.analysis.mechanism.appc draw   # from bipa/"], raw=c.get("raw", []))
    rows = [[s["method"], nm[s["bench"]], s["n"], *[f"{s[k]:+.3f}" for k in TC], f"{s['dlogX']:+.3f}", f"x{s['ratio']:.2f}", max(TC, key=lambda k: s[k])] for s in attr["skills"]]
    PO.record("appC_attribution", "fig", result=mt(["Optimizer", "Benchmark", "n", "more actions", "turn-splitting", "more content per turn", "longer text", "total dlogX", "ratio", "largest term"], rows)
              + "\n\nChange in log input tokens against the seed on the same test tasks (Table 1 common task set); limit elasticity k = 1 + s_H. Drawn as an annotated heatmap: rows are the skills grouped by optimizer, columns the four terms and the measured total, cells the change in log units. check() in the script asserts the Section 5.2 claims against the baseline rows.", source=["bipa/src/analysis/mechanism/appc.py"],
              commands=["<bach-harness venv python> -m src.analysis.mechanism.appc compute   # from bipa/", "<system python3 + matplotlib> -m src.analysis.mechanism.appc draw   # from bipa/"], raw=attr.get("raw", []))
    rows = [[r["method"], nm[r["bench"]], r["n"], f"x{r['raw']:.2f}", f"x{r['priced']:.2f}", f"x{r['raw_sum']:.2f}", f"x{r['priced_sum']:.2f}", f"{100*r['out_share']:.1f}%", f"{100*r['cache_share']:.0f}%"] for r in p]
    PO.record("appC_price", "fig", result=mt(["Optimizer", "Benchmark", "n", "input tokens (geo. mean)", "price (geo. mean)", "input tokens (ratio of sums)", "price (ratio of sums)", "output share of tokens", "cached share of input"], rows)
              + "\n\nPrice = uncached input + 0.1 x cached input + 4 x output tokens (illustrative weights); ratio optimized / seed over the Table 1 common task set.", source=["bipa/src/analysis/mechanism/appc.py", "bipa/src/analysis/mechanism/pricing.py"],
              commands=["<bach-harness venv python> -m src.analysis.mechanism.appc compute   # from bipa/", "<system python3 + matplotlib> -m src.analysis.mechanism.appc draw   # from bipa/"])


# ---- fig1 data stage (merged from fig1.py's compute; draw stays with appc draw / the dropped preview) -----------------------------------


def fig1_compute(models, out):
    """Per optimized skill the four terms against the seed (test tasks both completed), and the within-task regressions of log input tokens /
    log output tokens / log calls on log R per benchmark over seed and skill episodes."""
    from src.utils import dataloader as C
    from src.analysis.mechanism.slopes import metrics, episodes
    q = lambda rows: (lambda kn: 100 * sum(r["success"] is True for r in kn) / len(kn) / 100)([r for r in rows.values() if r["success"] is not None])  # Table 1's Q (percent) over the known outcomes of the whole test panel, as a fraction
    skills, all_eps, all_out, raw = [], collections.defaultdict(dict), collections.defaultdict(dict), []
    for model in models:
        for bench, _, _ in BENCH:
            seedrun = C.current_run("native-seed", model, bench); sf = C.eval_file(seedrun, "test", C.vid(C.SEED_TEXT))
            seed_rows = C.evaluation(seedrun, "test", C.vid(C.SEED_TEXT))[0]
            for disp, (run, t) in C.returned_skills(model, bench).items():
                f = C.eval_file(run, "test", C.vid(t)); own_rows = C.evaluation(run, "test", C.vid(t))[0]
                common = {k[0] for k, r in own_rows.items() if (k[0], 0) in seed_rows and r["success"] is not None and seed_rows[(k[0], 0)]["success"] is not None
                          and all(x and x > 0 for x in (r["N"], r["X"], seed_rows[(k[0], 0)]["N"], seed_rows[(k[0], 0)]["X"]))}  # Table 1's common task set
                if f is None: continue
                raw += [str(sf), str(f)]
                S = episodes(sf); Cs = episodes(f); S_out, C_out = output_totals(sf), output_totals(f); rows = []
                for task, ec in Cs.items():
                    a = metrics(S[task]) if task in S else None; b = metrics(ec)
                    if a and b and task in common: rows.append((task, a, b))
                    if b: all_eps[(model, bench)].setdefault((disp, task), b); all_out[(model, bench)].setdefault((disp, task), C_out.get(task))
                for task, es in S.items():
                    m = metrics(es)
                    if m: all_eps[(model, bench)].setdefault(("seed", task), m); all_out[(model, bench)].setdefault(("seed", task), S_out.get(task))
                if not rows: continue
                L = lambda k: np.array([math.log(b[k] / a[k]) for _, a, b in rows])
                dR, dN, dh, dX, dF = L("R"), L("N"), L("h"), L("X"), L("F")
                dm = dN - dR
                sF = np.array([(a["sF"] + b["sF"]) / 2 for _, a, b in rows]); sH = 1 - sF; Rm = np.array([(a["R"] + b["R"]) / 2 for _, a, b in rows]); k = 1 + sH * Rm / (Rm - 1)  # exact e_R
                seg = dict(more_actions=float(np.mean(k * dN)), more_requests=float(np.mean(-k * dm)), longer_content=float(np.mean(sH * dh)), longer_skill=float(np.mean(sF * dF)))
                skills.append(dict(model=model, bench=bench, method=disp, n=len(rows), **seg, segments_sum=sum(seg.values()), dlogX=float(dX.mean()), ratio=gmean(np.exp(dX)),
                                   succ_seed=q(seed_rows), succ=q(own_rows)))
    reg, pts = {}, {}
    for bkey, _, _ in BENCH:
        recs = [(k[1], m, all_out[(mo, be)].get(k)) for (mo, be), d in all_eps.items() if be == bkey for k, m in d.items()]
        if len(recs) < 5: continue
        rb, pb = {}, {}
        for qn, val in (("input", lambda m, o: m["X"]), ("output", lambda m, o: o), ("calls", lambda m, o: m["N"])):
            items = [(t, m["R"], v) for t, m, o in recs for v in [val(m, o)] if v and v > 0]
            rb[qn], pb[qn] = fe_reg(items)
        reg[bkey], pts[bkey] = rb, pb
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache.json").write_text(json.dumps(dict(skills=skills, reg=reg, raw=sorted(set(raw)),
                                                    pts={k: {qn: [v[0].tolist(), v[1].tolist()] for qn, v in d.items()} for k, d in pts.items()})))
    for k, v in reg.items():
        for qn, r in v.items(): print(k, qn, {a: round(b, 3) for a, b in r.items()})


def output_totals(path):
    """{task: output tokens summed over the distinct requests of the task's episode} (last episode of a task wins, as episodes())."""
    out = {}
    for e in json.loads(Path(path).read_text())["episodes"]:
        seen, o = set(), 0
        for r in e["requests"]:
            if r["request_id"] in seen: continue
            seen.add(r["request_id"]); o += r.get("output_tokens") or 0
        out[e["task_id"]] = float(o)
    return out


def fe_reg(items):
    """Within-task (fixed effects) slope of log y on log R over [(task, R, y)], with SE and the demeaned point cloud."""
    x = np.log([r[1] for r in items]); y = np.log([r[2] for r in items])
    g = collections.defaultdict(list)
    for i, it in enumerate(items): g[it[0]].append(i)
    xs, ys = x.copy(), y.copy()
    for idx in g.values(): xs[idx] -= x[idx].mean(); ys[idx] -= y[idx].mean()
    sl = float((xs * ys).sum() / (xs * xs).sum()); res = ys - sl * xs
    dof = len(xs) - len(g) - 1; se = float(np.sqrt((res ** 2).sum() / dof / (xs * xs).sum()))
    return dict(slope=sl, se=se, n=len(xs), n_tasks=len(g), r2=float(1 - (res ** 2).sum() / (ys ** 2).sum())), (xs, ys)


def fig1_cli(argv):
    ap = argparse.ArgumentParser(description="fig1 data stage: panel (a) within-task regressions and panel (b) four-term segments -> cache.json")
    ap.add_argument("--models", nargs="+", default=["deepseek-flash"])
    ap.add_argument("--out", default=str(FIG1))
    a = ap.parse_args(argv); fig1_compute(a.models, Path(a.out))


def main():
    if sys.argv[1:2] == ["fig1"]: fig1_cli(sys.argv[2:]); return
    if sys.argv[1] == "compute": compute(); return
    plt = style(); c = json.loads(F1.read_text()); p = json.loads((OUT / "price.json").read_text()); attr = json.loads((OUT / "attr.json").read_text())
    draw_slopes(plt, c); draw_attr(plt, attr["skills"]); draw_price(plt, p); records(c, p, attr)


if __name__ == "__main__": main()
