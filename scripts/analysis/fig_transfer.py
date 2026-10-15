"""Cross-model transfer figure. `compute` (slim venv python) -> figures/transfer.json; `plot` (system python3) -> transfer.{pdf,png}.
Each panel = bench x direction; arrow from the method's in-model point (train model's own result) to its transferred point on the eval model,
both as (rho vs eval seed, dQ vs eval seed in main-table.csv). Hollow = returned seed (placed at 0)."""
import csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "assets/artifacts/results"
OUT = RES / "figures/transfer"
MODELS = (("deepseek-flash", "DeepSeek"), ("gpt-6-luna", "Luna"))
BENCHES = (("tau3-bench", r"$\tau^3$-bench"), ("automationbench", "AutoBench"), ("terminal-bench-2-1", "TB2"))
C = {"skillopt": "#E69F00", "gepa": "#56B4E9", "skilladam": "#009E73", "ours": "#D55E00"}
NAME = {"skillopt": "SkillOpt", "gepa": "GEPA", "skilladam": "SkillAdam", "ours": "BACH"}


def compute():
    mt = {(r["model"], r["bench"], r["method"]): r for r in csv.DictReader(open(RES / "main-table.csv")) if r["status"] == "done"}
    sa = json.load(open(RES / "pareto-inputs.json"))["skilladam"]
    def front(model, bench):  # BACH: the returned node of the cell's ours run and its own seed (results/<model>/ours/<bench>/front.json)
        f = RES / model / "ours" / bench / "front.json"
        if not f.is_file(): return None
        fr = {r["node"]: r for r in json.load(open(f))["front"]}; r = next(r for r in fr.values() if r["returned"])
        return r["Q"], r["rho"], r["Q"] - fr["n0"]["Q"]
    def inm(model, bench, m):
        if m == "ours":
            v = front(model, bench)
            return (v[0], v[1]) if v else None
        if m == "skilladam":
            v = sa.get(model, {}).get(bench)
            return (v["Q"], v["rho"]) if v else None
        r = mt.get((model, bench, m))
        return (float(r["Q"]), float(r["rho"])) if r else None
    rows = []
    ood = json.load(open(RES / "ood-cross-model.json")); seen = {(o["bench"], o["train"], o["method"]) for o in ood}
    for f in sorted((RES.parents[1] / "execution").glob("*/*/*/ood-from-*/summary.json")):  # finished cells of a run still in progress
        o = json.load(open(f))
        if o.get("method") in ("skilladam", "ours") and (o["bench"], o["train"], o["method"]) not in seen: ood.append(o); seen.add((o["bench"], o["train"], o["method"]))
    for o in ood:
        s = float(mt[(o["eval"], o["bench"], "native-seed")]["Q"])
        tr = inm(o["train"], o["bench"], o["method"])
        ts = float(mt[(o["train"], o["bench"], "native-seed")]["Q"])
        rows.append(dict(bench=o["bench"], train=o["train"], eval=o["eval"], method=o["method"], returned_seed=o["returned_seed"],
                         seed_Q_eval=s, Q=o["Q"], rho=o["rho"], dQ=0.0 if o["returned_seed"] else o["Q"] - s,
                         dQ_raw=o["Q"] - s, in_Q=tr[0], in_rho=tr[1], in_dQ=front(o["train"], o["bench"])[2] if o["method"] == "ours" else tr[0] - ts))
    ours = [dict(model=m, bench=b, dQ=front(m, b)[2], rho=front(m, b)[1]) for m, _ in MODELS for b, _ in BENCHES if front(m, b)]
    have = {(r["bench"], r["train"], r["method"]) for r in rows}
    missing = [dict(bench=b, train=t, method=m) for b, _ in BENCHES for t, _ in MODELS for m in ("skilladam", "ours") if (b, t, m) not in have]
    json.dump(dict(transfer=rows, ours_in_model=ours, missing_transfer=missing), open(str(OUT) + ".json", "w"), indent=1)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(str(OUT) + ".json"))
    plt.rcParams.update({"font.size": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42, "font.family": "serif"})
    fig, axes = plt.subplots(3, 2, figsize=(3.3, 4.6), sharex=True, sharey=True)
    for j, (b, bl) in enumerate(BENCHES):
        for i, (tr, tl) in enumerate(MODELS):
            ax = axes[j][i]
            ev = [m for m in MODELS if m[0] != tr][0]
            ax.axhline(0, c="k", lw=0.5); ax.axvline(1, c="k", lw=0.5, ls=":")
            for r in d["transfer"]:
                if (r["bench"], r["train"]) != (b, tr): continue
                c = C[r["method"]]
                ax.annotate("", (r["rho"], r["dQ"]), (r["in_rho"], r["in_dQ"]), arrowprops=dict(arrowstyle="->", color=c, lw=0.7, alpha=0.6, shrinkA=1.5, shrinkB=2))
                ax.plot(r["in_rho"], r["in_dQ"], "o", ms=3, mfc="w", mec=c, mew=0.7, zorder=3)
                ax.plot(r["rho"], r["dQ"], "s", ms=4, mfc="w" if r["returned_seed"] else c, mec=c, mew=0.9, zorder=4)
            ax.set_xscale("log"); ax.set_xlim(0.92, 1.58); ax.set_ylim(-14, 20)
            ax.set_xticks([1, 1.2, 1.4]); ax.set_xticklabels(["1.0", "1.2", "1.4"])
            ax.minorticks_off(); ax.tick_params(length=2, width=0.5)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            if j == 0: ax.set_title(f"{tl} $\\to$ {ev[1]}", fontsize=7)
            if i == 0: ax.set_ylabel(f"{bl}\n$\\Delta Q$ (pp)")
            if j == 2: ax.set_xlabel(r"$\rho$ vs seed")
    h = [plt.Line2D([], [], c=C[m], lw=1.2, label=NAME[m]) for m in ("skillopt", "gepa")]
    h += [plt.Line2D([], [], marker="o", ls="", mfc="w", mec="gray", ms=3, label="in-model"),
          plt.Line2D([], [], marker="s", ls="", c="gray", ms=4, label="transferred"),
          plt.Line2D([], [], marker="s", ls="", mfc="w", mec="gray", ms=4, label="returned seed")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=6, handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.6, w_pad=0.6)
    for e in ("pdf", "png"): fig.savefig(f"{OUT}.{e}", dpi=300)


if __name__ == "__main__":
    plot() if sys.argv[1:] == ["plot"] else compute()
