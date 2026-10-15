"""Coupling figure: every (method, model, bench) result as (ln rho, dQ) against the run's own seed (compute), or the paper's Table 1 against the Seed row (table1).
  compute (slim venv python) | table1 -> results/figures/coupling.json     plot (system python3) -> coupling.{pdf,png}
Points: SkillOpt/GEPA/SkillAdam finals (main_table.evaluation/pair vs the run's own seed test eval), BACH front nodes (ours-front.md, seed Q of its own zo run),
cross-model transfers (ood-cross-model.json, seed = eval model's seed in execution/<eval>/native-seed/<bench>/ood-seed). Hollow = transfer."""
import csv, json, math, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "assets/artifacts/results/figures/coupling"
RES = ROOT / "assets/artifacts/results"
MODELS, BENCHES, KAPPA = ("deepseek-flash", "gpt-6-luna"), ("tau3-bench", "automationbench", "terminal-bench-2-1"), 1.1


def ranks(a):
    import numpy as np
    o = np.argsort(a, kind="stable"); r = np.empty(len(a)); i = 0
    s = np.asarray(a)[o]
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2; i = j + 1
    return r


def spearman(x, y):
    import numpy as np
    rx, ry = ranks(x), ranks(y)
    return float("nan") if rx.std() == 0 or ry.std() == 0 else float(np.corrcoef(rx, ry)[0, 1])


def boot(x, y, n=10000, seed=0):
    import numpy as np
    g, x, y = np.random.default_rng(seed), np.asarray(x), np.asarray(y)
    v = [spearman(x[i], y[i]) for i in (g.integers(0, len(x), len(x)) for _ in range(n))]
    return [float(p) for p in np.nanpercentile(v, [2.5, 97.5])]


def compute():
    sys.path[:0] = [str(HERE), str(ROOT / "baselines")]
    import main_table as M
    from bipa.layout import SKILLS, current_run
    vid = M.version((SKILLS / "seed/SKILL.md").read_text())
    pts = []
    def add(kind, method, model, bench, Q, seedQ, rho, **kw):
        pts.append(dict(kind=kind, method=method, model=model, bench=bench, Q=Q, seedQ=seedQ, dQ=Q - seedQ, rho=rho, lnrho=math.log(rho), **kw))
    for m in MODELS:
        for b in BENCHES:
            for arm in ("skillopt", "gepa", "skilladam"):
                run = current_run(arm, m, b)
                if not (run and (run / "summary.json").is_file()): continue
                fin = M.evaluation(run, M.version(json.loads((run / "summary.json").read_text())["final_skill"])); sd = M.evaluation(run, vid)
                if not (fin and sd): continue
                p, s = M.pair(fin, sd), M.pair(sd, None)
                add("in", arm, m, b, p["Q"], s["Q"], p["rho"], returned_seed=abs(p["rho"] - 1) < 1e-12 and p["Q"] == s["Q"])
    for m, d in json.loads((ROOT / "bach-harness/config/ours_runs.json").read_text()).items():
        if m not in MODELS: continue
        for b, t in d.items():
            run = current_run("zo", m, b)  # t = ours_runs.json tag
            if not (run / "summary.json").is_file(): continue
            res = json.loads((run / "zo-result.json").read_text()); nodes = {n["id"]: n for n in res["nodes"]}; sd = M.evaluation(run, vid); sq = M.pair(sd, None)["Q"]
            for i in res["front"]:
                ev = M.evaluation(run, M.version(nodes[i]["text"]))
                if ev:
                    p = M.pair(ev, sd); add("in", "ours", m, b, p["Q"], sq, p["rho"], returned=i == res["returned"], node=i)
    for o in json.loads((RES / "ood-cross-model.json").read_text()):
        sd = M.evaluation(ROOT / "assets/artifacts/execution" / o["eval"] / "native-seed" / o["bench"] / "ood-seed", vid)
        add("transfer", o["method"], o["eval"], o["bench"], o["Q"], M.pair(sd, None)["Q"], o["rho"], train=o["train"], returned_seed=o["returned_seed"])
    st = {}
    for name, sel in (("all", lambda p: True), ("optimizers_in", lambda p: p["kind"] == "in" and p["method"] != "ours"),
                      ("all_nonzero_rho", lambda p: p["rho"] != 1.0 or p["dQ"] != 0)):
        S = [p for p in pts if sel(p)]; x, y = [p["lnrho"] for p in S], [p["dQ"] for p in S]
        st[name] = dict(n=len(S), spearman=spearman(x, y), ci95=boot(x, y))
    gains = [p for p in pts if p["dQ"] > 1e-9]
    st["gains"] = dict(n=len(gains), inside_budget=[(p["method"], p["kind"], p["model"], p["bench"], round(p["dQ"], 2), round(p["rho"], 3)) for p in gains if p["rho"] <= KAPPA],
                       by_method_inside={k: sum(p["method"] == k and p["rho"] <= KAPPA for p in gains) for k in {p["method"] for p in gains}},
                       by_method_total={k: sum(p["method"] == k for p in gains) for k in {p["method"] for p in gains}})
    json.dump(dict(points=pts, stats=st, kappa=KAPPA), open(str(OUT) + ".json", "w"), indent=1)
    print(json.dumps(st, indent=1))


def table1():
    """Points from the paper's Table 1 (assets/cache/analysis/table1/table1.json): the skill each run returns, Q and rho against the Seed skill row (a returned
    seed shares the Seed row's evaluation, so it sits at the origin). Writes results/figures/coupling.json in compute()'s format, ref = seed row."""
    t1 = json.loads((ROOT / "assets/cache/analysis/table1/table1.json").read_text()); pts = []
    for cell, rows in t1.items():
        m, b = cell.split("|"); sq = rows["Seed skill"]["Q"]
        for name, arm in (("GEPA", "gepa"), ("SkillOpt", "skillopt"), ("SkillAdam", "skilladam"), ("BACH", "ours")):
            r = rows.get(name)
            if not r or r.get("Q") is None: continue
            pts.append(dict(kind="in", method=arm, model=m, bench=b, Q=r["Q"], seedQ=sq, dQ=r["Q"] - sq, rho=r["rho"], lnrho=math.log(r["rho"]), returned_seed=r["returned_seed"]))
    st = {}
    for name, sel in (("all", lambda p: True), ("all_nonzero_rho", lambda p: p["rho"] != 1.0 or p["dQ"] != 0)):
        S = [p for p in pts if sel(p)]; x, y = [p["lnrho"] for p in S], [p["dQ"] for p in S]
        st[name] = dict(n=len(S), spearman=spearman(x, y), ci95=boot(x, y))
    gains = [p for p in pts if p["dQ"] > 1e-9]
    st["gains"] = dict(n=len(gains), inside_budget=[(p["method"], p["model"], p["bench"], round(p["dQ"], 2), round(p["rho"], 3)) for p in gains if p["rho"] <= KAPPA])
    json.dump(dict(points=pts, stats=st, kappa=KAPPA, ref="seed row"), open(str(OUT) + ".json", "w"), indent=1)
    print(json.dumps(st, indent=1))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = json.load(open(str(OUT) + ".json")); P, st = d["points"], d["stats"]
    C = {"skillopt": "#E69F00", "gepa": "#56B4E9", "skilladam": "#009E73", "ours": "#D55E00"}
    MK = {"skillopt": "s", "gepa": "^", "skilladam": "D", "ours": "o"}
    NAME = {"skillopt": "SkillOpt", "gepa": "GEPA", "skilladam": "SkillAdam", "ours": "BACH"}
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    k = math.log(d["kappa"]); ax.axvspan(-0.2, k, color="#EEEEEE", zorder=0); ax.axvline(k, color="#666666", lw=0.8, ls="--", zorder=1)
    ax.axhline(0, color="#999999", lw=0.5, zorder=1)
    cnt = {}
    for p in P:
        m, hollow = p["method"], p["kind"] == "transfer"
        x, y = p["lnrho"], p["dQ"]
        if abs(x) < 1e-12 and abs(y) < 1e-9:  # seed returned: counted, drawn once at the origin
            cnt[m] = cnt.get(m, 0) + 1; continue
        ret = p.get("returned", True)
        s = 40 if m == "ours" and ret and not hollow else 22
        ax.scatter([x], [y], marker=MK[m], s=s, facecolors="none" if hollow or (m == "ours" and not ret) else C[m], edgecolors=C[m],
                   linewidths=1.0 if hollow or m == "ours" else 0.4, zorder=5 if m == "ours" else 4, alpha=1)
    n0 = sum(cnt.values()); ax.scatter([0], [0], marker="o", s=34, facecolors="none", edgecolors="#000000", linewidths=0.9, zorder=6)
    ax.annotate(f"seed returned $\\times${n0}", (0, 0), (0.012, -4.6), fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.5, color="#000000"), va="center")
    a, z = st["all"], st["all_nonzero_rho"]; nz = sum(p["rho"] == 1.0 and p["dQ"] == 0 for p in P)
    ax.text(0.02, 0.98, f"Spearman, {z['n']} points off the origin:\n$\\rho_s$ = {z['spearman']:.2f}, 95% CI [{z['ci95'][0]:.2f}, {z['ci95'][1]:.2f}]\n"
            f"all {a['n']} ({nz} at the origin): {a['spearman']:.2f} [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}]", transform=ax.transAxes, va="top", fontsize=6)
    xt = [0.9, 1.0, 1.1, 1.25, 1.5]
    ax.set_xticks([math.log(v) for v in xt]); ax.set_xticklabels([f"{v:g}" for v in xt]); ax.set_xlim(-0.14, 0.43)
    ref = "seed skill" if d.get("ref") == "seed row" else "own seed"
    ax.set_xlabel(rf"workload ratio $\rho$ vs. {ref} (log axis)"); ax.set_ylabel(rf"$\Delta Q$ vs. {ref} (pp)")
    ax.set_ylim(-13, 24); ymin, ymax = ax.get_ylim(); ax.text(k - 0.006, ymin + 0.03 * (ymax - ymin), "within budget", ha="right", va="bottom", fontsize=6, color="#555555")
    ax.text(k + 0.006, ymin + 0.03 * (ymax - ymin), r"$\rho>1.1$", ha="left", va="bottom", fontsize=6, color="#555555")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    h = [Line2D([], [], marker=MK[m], ls="", color=C[m], ms=4.5, label=NAME[m]) for m in MK]
    h += ([] if d.get("ref") == "seed row" else [Line2D([], [], marker="o", ls="", mfc="white", mec="#444444", ms=4.5, label="hollow: transfer, BACH front")]) + [Line2D([], [], color="#666666", ls="--", lw=0.8, label=r"budget $\rho=1.1$")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3, columnspacing=0.8, frameon=False, fontsize=6, handletextpad=0.2, labelspacing=0.25)
    fig.tight_layout()
    for e in ("pdf", "png"): fig.savefig(f"{OUT}.{e}", dpi=250)
    print("wrote", OUT)


if __name__ == "__main__":
    {"compute": compute, "table1": table1, "plot": plot}[sys.argv[1]]()
