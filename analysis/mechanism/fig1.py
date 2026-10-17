"""Old Figure 1, now only the data stage of the Appendix C figures (appc.py reads cache.json; draw writes a preview fig1.png to the output dir, nothing to paper/), DeepSeek test episodes, read-only: (a) within-task slopes of log input tokens / log output tokens / log calls on log requests, one line per benchmark with the
seed and optimized skills' episodes pooled (values quoted in Sec. 3 and Appendix slopes); (b) each optimized skill's change in log input tokens against the seed on the same test tasks, split into more actions,
the same actions over more requests, longer content per request and longer skill text (the four terms of slopes_checks.skill_fit); (c) schematic of the method (hand-drawn).
Two stages because the data python (the bach-harness venv) has no matplotlib and the plotting python (system python3) does:
  <venv python> -m analysis.mechanism.fig1 compute [--models deepseek-flash]     -> cache.json (panel a regression, panel b segments)   (run from bipa/)
  <system python3> -m analysis.mechanism.fig1 draw [--models deepseek-flash]                 -> fig1.pdf, fig1.png, numbers.json
Outputs in assets/cache/analysis/fig1/ (--out DIR). GEPA, SkillOpt, SkillAdam (the optimizers of common.OPTIMIZERS). Test tasks are Table 1's common task set (verified outcome and positive N, X on both sides); the success numbers are Table 1's Q."""
import argparse, collections, json, math, sys
from pathlib import Path
import numpy as np
from analysis.common import gmean

BENCH = [("tau3-bench", r"$\tau^3$-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench", "AutoBench"), ("terminal-bench-2-1", "TB2", "TB2")]
ROOT = Path("/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation/assets/cache/analysis/fig1")


def compute(models, out):
    """Per optimized skill the four terms against the seed (test tasks both completed), and the within-task regressions of log input tokens /
    log output tokens / log calls on log R per benchmark over seed and skill episodes."""
    from analysis import common as C
    from analysis.mechanism.slopes_checks import metrics, episodes
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


def draw(models, out):
    c = json.loads((out / "cache.json").read_text()); skills = c["skills"]
    reg = {b: v["input"] for b, v in c["reg"].items()}; pts = {k: (np.array(v["input"][0]), np.array(v["input"][1])) for k, v in c["pts"].items()}
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.linewidth": 0.6})
    BC = {"tau3-bench": "#0072B2", "automationbench": "#D55E00", "terminal-bench-2-1": "#009E73"}
    SC = {"more_actions": "#0072B2", "more_requests": "#E69F00", "longer_content": "#CC79A7", "longer_skill": "#999999"}
    SL = {"more_actions": "more actions", "more_requests": "same actions over more turns", "longer_content": "longer content per turn", "longer_skill": "longer skill text"}
    fig = plt.figure(figsize=(6.3, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 0.85], left=0.085, right=0.99, bottom=0.27, top=0.86, wspace=0.8)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    rng = np.random.default_rng(0)
    for bkey, disp, _ in BENCH:
        if bkey not in reg: continue
        xs, ys = pts[bkey]; r = reg[bkey]
        sel = rng.choice(len(xs), min(len(xs), 250), replace=False)
        axa.scatter(xs[sel], ys[sel], s=2.5, color=BC[bkey], alpha=0.18, lw=0, rasterized=True)
        xx = np.array([-0.6, 0.6])
        axa.plot(xx, r["slope"]*xx, color=BC[bkey], lw=1.4, label=f"{disp}: {r['slope']:.2f}")
    axa.plot([-0.6, 0.6], [-0.6, 0.6], color="k", ls="--", lw=0.8, label="1.00: proportional")
    axa.set_xlim(-0.7, 0.7); axa.set_ylim(-1.1, 2.25)
    axa.set_xlabel("turns per task\n(log, centered per task)"); axa.set_ylabel("input tokens per task\n(log, centered per task)")
    axa.legend(loc="upper left", frameon=False, fontsize=6.5, handlelength=1.4, borderpad=0.1, labelspacing=0.25)
    axa.set_title("Input tokens grow faster\nthan turns", loc="left", fontweight="bold")
    order = [s for bk, _, _ in BENCH for s in skills if s["bench"] == bk]
    short = {b[0]: b[2] for b in BENCH}
    ypos = np.arange(len(order))[::-1]
    for y, s in zip(ypos, order):
        pos = neg = 0
        for k in SC:
            v = s[k]
            if v >= 0: axb.barh(y, v, left=pos, color=SC[k], height=0.62, lw=0); pos += v
            else: axb.barh(y, v, left=neg, color=SC[k], height=0.62, lw=0); neg += v
        axb.plot(s["dlogX"], y, marker="D", ms=3.6, mfc="white", mec="k", mew=0.8, zorder=5)
        axb.text(0.545, y, f"x{s['ratio']:.2f}   {100*s['succ_seed']:.0f}→{100*s['succ']:.0f}%", va="center", ha="left", fontsize=6.8)
    axb.set_yticks(ypos); axb.set_yticklabels([f"{s['method']} · {short[s['bench']]}" for s in order])
    axb.axvline(0, color="k", lw=0.6); axb.set_xlim(-0.2, 0.52); axb.set_ylim(-0.6, len(order)-0.4)
    axb.set_xticks([-0.2, 0, 0.2, 0.4]) ; axb.spines[["top", "right"]].set_visible(False)
    axb.set_xlabel("change in log input tokens vs. seed skill"); axb.text(0.53, len(order)-0.45, "cost   success", fontsize=6.8, ha="left", va="bottom", style="italic")
    h = [plt.Rectangle((0, 0), 1, 1, color=SC[k]) for k in SC] + [plt.Line2D([], [], marker="D", ls="", mfc="white", mec="k", ms=3.6)]
    axb.legend(h, [SL[k] for k in SC]+["total"], loc="upper center", bbox_to_anchor=(0.45, -0.19), ncol=2, frameon=False, fontsize=6.6, handlelength=1, columnspacing=1, labelspacing=0.2)
    axb.set_title("Where an optimized skill's extra cost\ncomes from", loc="left", fontweight="bold")
    axc.set_xscale("log"); axc.set_xlim(0.55, 2.2); axc.set_ylim(0, 1)
    axc.set_xticks([]); axc.set_yticks([]); axc.minorticks_off(); axc.spines[["top", "right"]].set_color("0.6"); axc.spines[["left", "bottom"]].set_color("0.6")
    GR = "#555555"
    cx = [0.62, 0.8, 0.8, 1.0, 1.0, 1.25, 1.25, 1.75, 1.75, 2.1]; cy = [0.13, 0.13, 0.26, 0.26, 0.4, 0.4, 0.5, 0.5, 0.57, 0.57]
    axc.plot(cx, cy, color="#7f7f7f", lw=1.0, ls=(0, (3, 2)), zorder=1)
    axc.text(2.1, 0.03, "success\u2013cost curve\nof one run", fontsize=7, color=GR, ha="right", va="bottom")
    axc.scatter([1.0], [0.4], s=28, color="k", zorder=4); axc.text(0.93, 0.4, "seed skill", ha="right", va="center", fontsize=7)
    axc.scatter([1.8], [0.85], s=34, color=BC["automationbench"], zorder=4)
    axc.text(1.8, 0.9, "optimized skill", ha="right", va="bottom", fontsize=7)
    axc.annotate("", xy=(1.0, 0.85), xytext=(1.72, 0.85), arrowprops=dict(arrowstyle="-|>", color="#0072B2", lw=1.5, shrinkA=0, shrinkB=0), zorder=3)
    axc.text(1.25, 0.8, "keep actions,\nremove overhead", color="#0072B2", fontsize=7, ha="center", va="top")
    axc.set_xlabel("input tokens vs. seed (log)"); axc.set_ylabel("task success")
    axc.set_title("How the method uses it\n(schematic only)", loc="left", fontweight="bold")
    fig.savefig(out/"fig1.png", dpi=200)
    nums = dict(models=models, panel_a=reg, panel_b=skills, excluded_methods=[])
    json.dump(nums, open(out/"numbers.json", "w"), indent=1)
    for k, v in reg.items(): print(k, {a: round(b, 3) for a, b in v.items()})
    for s in skills: print(s["method"], s["bench"], s["n"], *[round(s[k], 3) for k in SC], round(s["dlogX"], 3), round(s["succ_seed"], 2), round(s["succ"], 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["compute", "draw"])
    ap.add_argument("--models", nargs="+", default=["deepseek-flash"])
    ap.add_argument("--out", default=str(ROOT))
    a = ap.parse_args(); out = Path(a.out)
    if a.stage == "compute": compute(a.models, out)
    else: draw(a.models, out)


if __name__ == "__main__":
    main()
