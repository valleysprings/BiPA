"""The transfer figure (mech_transfer3): learned skills in their own setting and transferred to another benchmark or model.

(a) test success minus the seed (points, transfer_dq.json), (b) input-token ratio to the seed; one marker per optimizer.
Ratios in (b): own setting DeepSeek = Table 1's rho_trj; Luna = luna_transfer.json (X against the Luna seed);
cross-benchmark = the single transfer run against the target benchmark's ood-bench-seed run (aux.headline_numbers.transfer).
Inputs: python -m src.analysis table1, python -m aux.headline_numbers.transfer luna, python -m aux.transfer_and_trajectories.transfer_dq.
Run:  python -m aux.transfer_and_trajectories.transfer_fig
"""
import json

from aux.headline_numbers import transfer as L
from aux.transfer_and_trajectories.transfer_dq import HERE
from src.utils.dataloader import CACHE, DS, out_dir
from src.utils.fig import save_figure

OC = {"GEPA": "#009E73", "SkillOpt": "#0072B2", "SkillAdam": "#D55E00"}
MK = {"GEPA": "^", "SkillOpt": "o", "SkillAdam": "s"}
ARM = {"GEPA": "gepa", "SkillOpt": "skillopt", "SkillAdam": "skilladam"}
BENCH = {"TB2": "terminal-bench-2-1", "tau3": "tau3-bench", "AutoB": "automationbench"}
INK, MUTED, GRID = "#222222", "#5A5A5A", "#E2E5EA"
TAU = r"$\tau^3$-bench"
BLOCKS = [("Skill in its own setting", [("in-domain", "DeepSeek", "TB2", "DeepSeek, TB2"),
                                        ("in-domain", "DeepSeek", "AutoB", "DeepSeek, AutoBench"),
                                        ("in-domain", "DeepSeek", "tau3", "DeepSeek, " + TAU),
                                        ("in-domain", "Luna", "tau3", "Luna, " + TAU)]),
          ("Skill transferred", [("cross-benchmark", "DeepSeek", "TB2->AutoB", r"DeepSeek, TB2 $\rightarrow$ AutoBench"),
                                 ("cross-benchmark", "DeepSeek", "AutoB->TB2", r"DeepSeek, AutoBench $\rightarrow$ TB2"),
                                 ("cross-model", "Luna", "tau3", r"DeepSeek $\rightarrow$ Luna, " + TAU)])]
DODGE = {"GEPA": 0.22, "SkillOpt": 0.0, "SkillAdam": -0.22}
# the ratios as reported, at two decimals; the recomputed values must round to them
PRINTED = {("in-domain", "DeepSeek", "TB2"): (1.34, 1.43, 1.37), ("in-domain", "DeepSeek", "tau3"): (1.15, 1.22, 1.09),
           ("in-domain", "DeepSeek", "AutoB"): (1.21, 1.25, 1.08), ("cross-benchmark", "DeepSeek", "TB2->AutoB"): (1.32, 1.49, 1.24),
           ("cross-benchmark", "DeepSeek", "AutoB->TB2"): (1.22, 1.01, 1.00), ("in-domain", "Luna", "tau3"): (1.04, 1.01, 0.98),
           ("cross-model", "Luna", "tau3"): (1.17, 1.38, 1.08)}


def need(path, step):
    if not path.is_file():
        raise SystemExit(f"{path} missing: run python -m {step}")
    return json.loads(path.read_text())


def ratios():
    """{(group, model, bench): {optimizer: input-token ratio to the seed, rounded to two decimals}}."""
    table1 = need(out_dir("table1") / "table1.json", "src.analysis table1")
    luna = need(CACHE / "luna_transfer.json", "aux.headline_numbers.transfer luna")["tau3-bench"]
    rho = {}
    for b in ("TB2", "tau3", "AutoB"):
        rho[("in-domain", "DeepSeek", b)] = {m: table1[f"deepseek-flash|{BENCH[b]}"][m]["rho_trj"] for m in OC}
    for tag, group in (("native-20261012", "in-domain"), ("ood-from-deepseek-flash", "cross-model")):
        rho[(group, "Luna", "tau3")] = {m: luna[f"{ARM[m]}@{tag}"]["rho"]["X"] for m in OC}
    for src, tgt in (("TB2", "AutoB"), ("AutoB", "TB2")):
        seed = L.means(L.reps_of([DS / "native-seed" / BENCH[tgt] / "ood-bench-seed"]))
        rho[("cross-benchmark", "DeepSeek", f"{src}->{tgt}")] = {
            m: L.gm(L.means(L.reps_of([DS / ARM[m] / BENCH[tgt] / f"ood-from-bench-{BENCH[src]}"])), seed, "X")[0] for m in OC}
    got = {k: tuple(round(v[m], 2) for m in OC) for k, v in rho.items()}
    assert got == PRINTED, {k: (got[k], PRINTED[k]) for k in PRINTED if got[k] != PRINTED[k]}
    return {k: dict(zip(OC, v)) for k, v in got.items()}  # drawn at the two decimals the text prints


def rows_and_heads():
    """y position of every row and block header, top to bottom."""
    y, rows, heads = 0.0, [], []
    for b, (head, items) in enumerate(BLOCKS):
        if b:
            y -= 0.55
        heads.append((y, head))
        y -= 1.0
        for it in items:
            rows.append((y, it))
            y -= 1.0
    return rows, heads, y


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8, "mathtext.fontset": "stix",
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "xtick.major.size": 2.5})
    dq = {(r["group"], r["model"], r["bench"], r["method"]): r["dQ"]
          for r in need(HERE / "transfer_dq.json", "aux.transfer_and_trajectories.transfer_dq")}
    rho = ratios()
    rows, heads, ybot = rows_and_heads()
    figure, (axq, axr) = plt.subplots(1, 2, figsize=(6.3, 2.75), sharey=True, gridspec_kw=dict(wspace=0.08))
    figure.subplots_adjust(left=0.265, right=0.985, bottom=0.125, top=0.76)
    sep = (rows[3][0] + heads[1][0]) / 2
    for ax, ref in ((axq, 0.0), (axr, 1.0)):
        for y, (group, model, bench, _) in rows:
            ax.axhline(y, color=GRID, lw=0.5, zorder=0)
            for m in OC:
                v = dq[(group, model, bench, m)] if ax is axq else rho[(group, model, bench)][m]
                ax.plot(v, y + DODGE[m], marker=MK[m], ms=4.6, color=OC[m], markeredgecolor="white", markeredgewidth=0.4, lw=0, zorder=3)
        ax.axvline(ref, color=INK, lw=0.7, ls="--", zorder=1)
        ax.text(ref, 1.0, "seed", transform=blended_transform_factory(ax.transData, ax.transAxes), ha="center", va="bottom",
                fontsize=7, color=MUTED)
        ax.axhline(sep, color="#9AA0A8", lw=0.5, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=7.5, pad=1.5)
    axq.set_ylim(ybot + 0.45, heads[0][0] + 0.45)
    axq.set_yticks([y for y, _ in rows])
    axq.set_yticklabels([it[3] for _, it in rows], fontsize=8, color=INK)
    lab = blended_transform_factory(figure.transFigure, axq.transData)
    for y, head in heads:
        axq.text(0.012, y, head, transform=lab, ha="left", va="center", fontsize=8, fontweight="bold", color=INK)
    axq.set_xlim(-12.5, 4.5)
    axq.set_xticks([-10, -5, 0])
    axq.set_xticklabels(["$-$10", "$-$5", "0"])
    axq.set_xlabel("test difference to the seed (points)", fontsize=8, labelpad=2)
    axr.set_xlim(0.94, 1.56)
    axr.set_xticks([1.0, 1.2, 1.4])
    axr.set_xlabel("input tokens / seed", fontsize=8, labelpad=2)
    own = [dq[(g, mo, b, m)] for _, (g, mo, b, _) in rows[:4] for m in OC]
    moved = [dq[(g, mo, b, m)] for _, (g, mo, b, _) in rows[4:] for m in OC]
    n_own, n_moved = sum(v < 0 for v in own), sum(v < 0 for v in moved)
    print(f"below the seed: own setting {n_own} of {len(own)}, transferred {n_moved} of {len(moved)}")
    print("transferred rho > 1.05:", sum(rho[(g, mo, b)][m] > 1.05 for _, (g, mo, b, _) in rows[4:] for m in OC), "of", len(moved))
    axq.set_title(f"(a) Skills score below the seed in their own\nsetting ({n_own} of {len(own)}) and transferred ({n_moved} of {len(moved)})",
                  fontsize=8, loc="left", pad=11, linespacing=1.15)
    axr.set_title("(b) Most of the extra input tokens travel\nwith the skill", fontsize=8, loc="left", pad=11, linespacing=1.15)
    handles = [plt.Line2D([], [], marker=MK[m], color=OC[m], markeredgecolor="white", markeredgewidth=0.4, lw=0, ms=5, label=m) for m in OC]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.625, 1.01), ncol=3, frameon=False, fontsize=8,
                  handletextpad=0.2, columnspacing=1.4)
    save_figure(figure, "transfer3")


if __name__ == "__main__":
    main()
