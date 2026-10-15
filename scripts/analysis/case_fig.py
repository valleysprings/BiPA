"""AutoBench case-study figure: per-cluster change in test success and cost ratio of every learned skill against the seed.

Run with system python3. Reads assets/artifacts/attribution/<model>/ours/case-study-autob/clusters.json.
Skills identical to the seed (returned seed, cost ratio 1 everywhere) are left out. Writes
assets/artifacts/results/figures/case-autob.{pdf,png}; --copy-to DIR copies both.
"""
import argparse, json, shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
MODELS = (("gpt-6-luna", "GPT-6 Luna"), ("deepseek-flash", "DeepSeek-V4.1-Flash"))
CL = ["finance", "hr", "marketing", "operations", "sales", "support"]
STYLE = {"skillopt": ("SkillOpt", "#E69F00"), "gepa": ("GEPA", "#56B4E9"), "BACH n1": ("BACH front", "#D55E00"),
         "BACH n3": ("BACH front", "#D55E00"), "BACH n4": ("BACH front 2", "#CC79A7")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy-to")
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "serif"})
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.3), sharex=True)
    for j, (model, label) in enumerate(MODELS):
        d = json.loads((ROOT / f"assets/artifacts/attribution/{model}/ours/case-study-autob/clusters.json").read_text())
        skills = [k for k in d if k in STYLE and any(abs(d[k][c]["rho"] - 1) > 1e-9 for c in CL)]
        w = 0.8 / len(skills)
        for i, k in enumerate(skills):
            x = [c + (i - (len(skills) - 1) / 2) * w for c in range(len(CL))]
            axes[0][j].bar(x, [100 * (d[k][c]["Q"] - d["seed"][c]["Q"]) for c in CL], w, color=STYLE[k][1], label=STYLE[k][0])
            axes[1][j].bar(x, [d[k][c]["rho"] for c in CL], w, color=STYLE[k][1])
        axes[0][j].axhline(0, color="black", lw=0.6)
        axes[1][j].axhline(1, color="#666666", lw=0.8, ls="--")
        axes[1][j].axhline(1.1, color="#666666", lw=0.6, ls=":")
        axes[0][j].set_title(label, fontsize=7.5)
        axes[0][j].legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1, columnspacing=0.8)
        axes[1][j].set_xticks(range(len(CL)), ["fin.", "hr", "mkt.", "ops.", "sales", "supp."], fontsize=6.5)
        axes[1][j].set_ylim(0.6, 1.8)
    axes[0][0].set_ylabel(r"$\Delta Q$ vs seed (pp)")
    axes[1][0].set_ylabel(r"$\rho$ vs seed")
    for ax in axes.flat:
        ax.grid(axis="y", color="#DDDDDD", lw=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[0]:
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.22 * (hi - lo))
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    out = ROOT / "assets/artifacts/results/figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"case-autob.{ext}", dpi=220)
        if args.copy_to:
            Path(args.copy_to).mkdir(parents=True, exist_ok=True)
            shutil.copy(out / f"case-autob.{ext}", args.copy_to)


if __name__ == "__main__":
    main()
