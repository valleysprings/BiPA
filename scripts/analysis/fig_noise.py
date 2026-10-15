"""Figure "noise": small selection panels cannot verify a workload budget.
  compute (slim venv):  bach-harness/.venv/bin/python baselines/scripts/analysis/fig_noise.py compute
  plot    (system py3): python3 baselines/scripts/analysis/fig_noise.py plot
Reads execution/<model>/ours/<bench>/<tag>/zo-result.json + test evaluations and the seed evaluations in execution/; writes results/figures/noise.{json,pdf,png}."""
import sys, json, glob, math, os, itertools, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "assets/artifacts/results/figures"
KAPPA = math.log(1.1)
SE_GRID = (0.085, 0.18)  # sel SE_L at 30 tasks (tau3/AutoB) and 9 tasks (TB2)


def phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def compute():
    import numpy as np
    sys.path.insert(0, str(ROOT / "baselines/scripts/analysis")); sys.path.insert(0, str(ROOT / "baselines"))
    import main_table as MT
    tags = {k: v for k, v in json.load(open(ROOT / "bach-harness/config/ours_runs.json")).items() if not k.startswith("_")}
    se = lambda x: float(np.std(x, ddof=1) / math.sqrt(len(x)))

    def test_eval(run, version):
        f = max(glob.glob(f"{run}/results/evaluations/test/{version}-*/evaluation.json"), key=os.path.getmtime)
        data = json.load(open(f)); eps = {}
        for e in data["episodes"]: eps.setdefault((e["task_id"], e.get("repetition", 0)), e)
        out = {}
        for t in data["panel"]:
            r = MT.task_row(eps.get((t["task_id"], t.get("repetition", 0)), {})); out[t["task_id"]] = (r["success"], r["c"])
        return out
    nodes_out, screens, rng = [], [], np.random.default_rng(0)
    for m, bs in tags.items():
        for b, tag in bs.items():
            run = f"{ROOT}/assets/artifacts/execution/{m}/ours/{b}/{tag}"; d = json.load(open(run + "/zo-result.json")); N = {n["id"]: n for n in d["nodes"]}
            seed = test_eval(run, N["n0"]["version"])
            for n in d["nodes"]:
                if n["id"] == "n0" or not (n["id"] in d["front"] or n["id"] == d["returned"]) or not n["sel"]: continue
                t = test_eval(run, n["version"])
                l = [math.log(t[k][1] / seed[k][1]) for k in t if t[k][0] is not None and seed.get(k) and seed[k][0] is not None and t[k][1] and seed[k][1]]
                nodes_out.append(dict(model=m, bench=b, node=n["id"], sel_L=n["sel"]["L"], sel_se=n["sel"]["se"], sel_n=n["sel"]["n"],
                                      test_logrho=float(np.mean(l)), test_se=se(l), test_n=len(l), returned=d["returned"] == n["id"]))
            # screens for the null pass rate: paired sign-flip of each screen's own per-task (dQ, dS, z), 4000 flips
            hmax = 0.405
            for n in d["nodes"]:
                sc = n["screen"]
                if not sc: continue
                P = N[n["parent"]]; ids = sc["tasks"]; ph = next(h for h in reversed(P["hist"]) if all(k in h for k in ids)); ch = n["hist"][0]
                dy = np.array([ch[k][0] - ph[k][0] for k in ids], float); ds = np.array([ch[k][1] - ph[k][1] for k in ids], float); z = np.array(sc["z"])
                Lp = P["sel"]["L"] if P["sel"] else 0.0
                sg = rng.choice([-1, 1], (4000, len(ids))); dq, dsb, zb = (sg * dy).mean(1), (sg * ds).mean(1), (sg * z).mean(1)  # loop.passes
                ok = (((dq > 0) | ((dq == 0) & (dsb > 0))) & (Lp + zb <= hmax)) | ((zb < 0) & (dq >= -0.1))
                screens.append(float(ok.mean()))
    # sel gate: P(L-hat <= kappa | true L) = Phi((kappa - L) / SE)
    Ls = [round(x, 3) for x in np.linspace(-0.3, 0.7, 101)]
    gate = {str(s): [phi((KAPPA - L) / s) for L in Ls] for s in SE_GRID}
    # seed re-evaluations: independent executions of the seed skill (4fd5ca44) per cell
    R = str(ROOT); fs = sorted(set(glob.glob(R + "/assets/artifacts/execution/*/*/*/*/results/evaluations/test/4fd5ca44*/evaluation.json")))
    cells = collections.defaultdict(dict)
    for f in fs:
        p = f.replace(R + "/", "").split("/")
        model, bench, name = p[3], p[5], p[4] + "/" + p[6]
        dd = json.load(open(f)); cells[(model, bench)][name] = (dd["slot_accounting"], {(e["task_id"], e["repetition"]): e for e in dd["episodes"]})
    fp = lambda e: [(r["input_tokens"], r["output_tokens"]) for r in e["requests"]]
    seeds = []
    for (model, bench), runs in sorted(cells.items()):
        ex = [r for r, (sa, _) in runs.items() if sa["executed"] > 0 and not r.startswith(("ours/", "skilladam/")) and "/ood-" not in r and "/probe-" not in r]
        best = None
        for a, b2 in itertools.combinations(ex, 2):
            ea, eb = runs[a][1], runs[b2][1]
            ind = [k for k in ea if k in eb and ea[k]["outcome_valid"] and eb[k]["outcome_valid"] and fp(ea[k]) != fp(eb[k])]
            if not ind: continue
            com = [k for k in ea if k in eb and ea[k]["outcome_valid"] and eb[k]["outcome_valid"]]
            row = dict(model=model, bench=bench, a=a, b=b2, n_ind=len(ind), n_common=len(com), flip=sum(ea[k]["success"] != eb[k]["success"] for k in ind) / len(ind),
                       dQ=100 * abs(sum(ea[k]["success"] for k in com) - sum(eb[k]["success"] for k in com)) / len(com))
            best = best if best and best["n_ind"] >= row["n_ind"] else row
        if best: seeds.append(best)
    json.dump(dict(kappa=KAPPA, nodes=nodes_out, L_grid=Ls, gate=gate, se_grid=SE_GRID, screen_null_pass=float(np.mean(screens)), n_screens=len(screens), seed_reeval=seeds),
              open(OUT / "noise.json", "w"), indent=1)
    print(json.dumps(dict(nodes=nodes_out, null=np.mean(screens), n=len(screens), seeds=seeds), indent=1))


def plot():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    D = json.load(open(OUT / "noise.json")); k = D["kappa"]; V = "#D55E00"
    plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "serif"})
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.3, 5.0), gridspec_kw=dict(height_ratios=[1.15, 1]))
    lim = (-0.3, 0.8)
    a.axvspan(k, lim[1], color="#EEEEEE", zorder=0); a.axvline(k, color="#666", lw=0.8, ls="--"); a.axhline(k, color="#666", lw=0.8, ls="--")
    a.plot(lim, lim, color="k", lw=0.6)
    for n in D["nodes"]:
        a.errorbar(n["sel_L"], n["test_logrho"], xerr=n["sel_se"], yerr=n["test_se"], fmt="o", ms=5, color=V, ecolor=V, elinewidth=0.8, capsize=0, mfc=V if n["returned"] else "none")
        if n["bench"].startswith("terminal"): a.annotate("TB2", (n["sel_L"], n["test_logrho"]), (4, -9), textcoords="offset points", fontsize=6.5)
    a.set(xlim=lim, ylim=(-0.4, 0.5), xlabel=r"selection $\hat L$ (30 / 9 tasks)", ylabel=r"test $\log\rho$ (paired test tasks)")
    a.text(0.02, 0.97, "(a)", transform=a.transAxes, va="top", fontweight="bold"); a.text(k + 0.01, -0.38, r"$\kappa=\log 1.1$", fontsize=6.5, color="#444")
    a.text(0.02, 0.88, "filled = returned; bars $\\pm1$ SE\nseed node n0 omitted", transform=a.transAxes, va="top", fontsize=6)
    Ls = D["L_grid"]; sty = {"0.085": ("-", "30 tasks, SE 0.085"), "0.18": ("--", "9 tasks, SE 0.18")}
    for s, p in D["gate"].items(): b.plot(Ls, p, color=V, ls=sty[s][0], lw=1.4, label=sty[s][1])
    b.axhline(D["screen_null_pass"], color="#555", lw=0.8, ls=":"); b.text(0.69, D["screen_null_pass"] + 0.02, f"screen, no true effect: {D['screen_null_pass']:.2f}", ha="right", fontsize=6.5)
    b.axvline(k, color="#666", lw=0.8, ls="--"); b.set(xlim=(-0.2, 0.7), ylim=(0, 1.02), xlabel=r"true $L$ of a candidate", ylabel=r"P($\hat L\leq\kappa$)")
    b.legend(frameon=False, fontsize=6.5, loc="upper right", bbox_to_anchor=(1, 0.88)); b.text(0.02, 0.97, "(b)", transform=b.transAxes, va="top", fontweight="bold")
    S = D["seed_reeval"]; fl = [x["flip"] for x in S]; dq = [x["dQ"] for x in S]
    b.text(0.98, 0.45, f"seed re-run, same skill:\n{100*min(fl):.0f}-{100*max(fl):.1f}% of tasks flip,\n$\\Delta$Q {min(dq):.0f}-{max(dq):.0f} pp", transform=b.transAxes, fontsize=6, ha="right", va="top")
    for s in "top", "right": a.spines[s].set_visible(False); b.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    for e in ("pdf", "png"): fig.savefig(OUT / f"noise.{e}", dpi=300)


if __name__ == "__main__": {"compute": compute, "plot": plot}[sys.argv[1]]()
