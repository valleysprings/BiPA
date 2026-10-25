"""Figures devdyn and devchance, one column each.

Both put what chance alone reports next to what the optimizers reported and what happened on test.
Chance = the replay of the optimizer's own selection rule on copies of the seed (selection.py): dev outcomes of every candidate drawn from the posterior of each dev task, candidates identical to the seed,
gain measured against the run's starting score.  Band = 5th to 95th percentile of the replayed gain, line = its mean.

devdyn: per benchmark, DeepSeek-V4.1-Flash, SkillOpt and GEPA.  x = candidates scored on dev.  Lines: the dev gain the run reported so far (best so far, points over the starting score).
  Grey band and dashed line: the gain best-of-k copies of the seed report.  Hollow markers: the test difference to the seed of the returned skill (avg@3, Table 1 runs); intermediate path skills
  (one test pass each) are added when path_test.json of aux.path_test exists.
devchance: for each of the nine skills, the reported dev gain (solid), the mean and 5-95 percent range of the gain chance alone reports (diamond with bar), and the test difference (hollow).
  SkillAdam's chance value is the replay of its gate (skilladam_reported_gain), a sum over its accepted patches.
Run:  python -m aux.selection_replay.fig_chance
"""
import json, os
import numpy as np
from aux.selection_replay.style import INK, MUTED, GRID, BENCH_NAME, setup_mpl  # noqa
from src.analysis import selection as S
from aux.selection_replay import fig_devdyn as D  # curve(), END, COL, NAME, MARK

from src import layout

OUT = str(layout.OUT / "selection_replay")
PATHTEST = str(layout.OUT / "path_test/path_test.json")
BENCH = ["tau3-bench", "automationbench", "terminal-bench-2-1"]
DRAWS = 20000


def chance(bench):
    dev_files, test_files = S.seed_files(bench)
    dev, test = S.executions(dev_files), S.executions(test_files)
    a, b = S.fit_prior([(sum(v), len(v)) for v in list(dev.values()) + list(test.values())])
    n = len(dev)
    obs = S.observed(bench, n)
    S.RNG = np.random.default_rng(20261024)
    p = S.task_probabilities(dev, (a, b), n, DRAWS)
    base = obs["SkillOpt"]["baseline"]
    ks = list(range(1, 9))
    best = {}
    for k in ks:
        g = 100 * S.best_of(p, k, base) / n
        best[k] = (float(g.mean()), float(np.percentile(g, 5)), float(np.percentile(g, 95)))
    rng = np.random.default_rng(20261024)
    sa = obs["SkillAdam"]
    tot, _ = S.skilladam_reported_gain(p, sa["batch"], sa["budgets"], rng)
    ada = (float(tot.mean()), float(np.percentile(tot, 5)), float(np.percentile(tot, 95)))
    return dict(n_dev=n, base=base, best_of_k=best, skilladam=ada, obs=obs)


def main():
    os.makedirs(OUT, exist_ok=True)
    plt = setup_mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ch = {b: chance(b) for b in BENCH}
    path = json.load(open(PATHTEST)) if os.path.exists(PATHTEST) else {}
    # ---------------- devdyn
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 3.4), sharex=True)
    for ax, bench in zip(axes, BENCH):
        c = ch[bench]
        ks = np.arange(1, 9)
        mean = np.array([c["best_of_k"][k][0] for k in ks]); lo = np.array([c["best_of_k"][k][1] for k in ks]); hi = np.array([c["best_of_k"][k][2] for k in ks])
        ax.fill_between(ks, lo, hi, color="#C8CDD3", alpha=0.7, lw=0, step="mid", zorder=1)
        ax.plot(ks, mean, color="#6B7480", lw=1.0, ls=(0, (3, 2)), zorder=2)
        for opt in ("skillopt", "gepa"):
            x, y = D.curve(opt, bench)
            K = len(y) - 1
            xs = np.arange(0, K + 1)
            ax.step(xs, y, where="post", color=D.COL[opt], lw=1.6, zorder=4)
            dev, test = D.END[(bench, opt)]
            pts = sorted((k, g) for k, g, role in path.get("deepseek-flash|%s|%s" % (opt, bench), []) if k is not None)
            if pts:
                # tau3: every skill on the path, the returned one included, was tested once on the current model version against the seed rerun on it
                dx = 0.07 if opt == "skillopt" else -0.07  # the two optimizers share k = 1 and 2 with equal gains
                ax.plot([k + dx for k, _ in pts], [g for _, g in pts], color=D.COL[opt], lw=0.8, alpha=0.7, zorder=5)
                ax.scatter([k + dx for k, _ in pts], [g for _, g in pts], marker="o", s=18, facecolor="white", edgecolor=D.COL[opt], lw=1.2, zorder=6)
            else:
                ax.scatter([K + 0.4], [test], marker="o", s=22, facecolor="white", edgecolor=D.COL[opt], lw=1.3, zorder=6, clip_on=False)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xlim(0, 9.0); ax.set_ylim(-14, 30)
        ax.grid(axis="y", color=GRID, lw=0.4); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7, pad=1.5)
        ax.set_title(BENCH_NAME[bench], fontsize=8, loc="left", pad=2, fontweight="bold")
        ax.set_ylabel("points vs the seed", fontsize=7, labelpad=1)
    axes[-1].set_xlabel("candidates scored on dev", fontsize=7.5, labelpad=2)
    h = [Line2D([0], [0], color=D.COL["gepa"], lw=1.6, label="GEPA, dev gain so far"), Line2D([0], [0], color=D.COL["skillopt"], lw=1.6, label="SkillOpt, dev gain so far"),
         Patch(color="#C8CDD3", label="what copies of the seed report (5 to 95%)"), Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white", lw=0, markersize=4.5, label="test difference")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=6.2, handlelength=1.6, columnspacing=1.0, bbox_to_anchor=(0.52, 0.0))
    fig.tight_layout(rect=(0, 0.1, 0.98, 1), h_pad=0.5)
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(f"{OUT}/devdyn.{ext}", **kw)
    # ---------------- devchance
    fig, ax = plt.subplots(figsize=(3.35, 2.75))
    x = 0.0
    ticks, labels, rows = [], [], []
    for bench in BENCH:
        c = ch[bench]
        for opt in ("gepa", "skillopt", "skilladam"):
            dev, test = D.END[(bench, opt)]
            if opt == "skilladam":
                m, lo, hi = c["skilladam"]
            else:
                K = c["obs"]["SkillOpt" if opt == "skillopt" else "GEPA"]["K"]
                m, lo, hi = c["best_of_k"][min(K, 8)]
            col = D.COL[opt]
            ax.plot([x, x], [lo, hi], color=col, lw=2.2, alpha=0.35, solid_capstyle="butt", zorder=2)
            ax.scatter([x - 0.16], [dev], marker="o", s=26, facecolor=col, edgecolor="white", lw=0.5, zorder=5)
            ax.scatter([x], [m], marker="D", s=14, facecolor="white", edgecolor=col, lw=1.0, zorder=4)
            ax.scatter([x + 0.16], [test], marker="o", s=26, facecolor="white", edgecolor=col, lw=1.3, zorder=5)
            rows.append(dict(bench=bench, opt=opt, dev_reported=dev, chance_mean=round(m, 1), chance_5=round(lo, 1), chance_95=round(hi, 1), test=test))
            ticks.append(x); labels.append(D.NAME[opt].replace("SkillAdam", "SkillAdam"))
            x += 1.0
        x += 0.7
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=5.4, rotation=35, ha="right")
    ax.set_ylabel("points vs the seed", fontsize=7.5, labelpad=2)
    ax.set_ylim(-14, 50); ax.grid(axis="y", color=GRID, lw=0.4); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, pad=1.5, length=2)
    for i, bench in enumerate(BENCH):
        xs = [t for t, r in zip(ticks, rows) if r["bench"] == bench]
        ax.text(np.mean(xs), -0.25, BENCH_NAME[bench], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.2, fontweight="bold", color=INK)
    h = [Line2D([0], [0], marker="o", color="#555555", lw=0, markersize=5, label="reported dev gain"), Line2D([0], [0], marker="D", color="#555555", markerfacecolor="white", lw=0, markersize=4, label="what copies of the seed report"),
         Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white", lw=0, markersize=5, label="test difference")]
    ax.legend(handles=h, loc="upper left", frameon=False, fontsize=6.0, handletextpad=0.2, bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(f"{OUT}/devchance.{ext}", **kw)
    json.dump(dict(chance={b: dict(n_dev=c["n_dev"], baseline_tasks=c["base"], best_of_k={str(k): [round(v, 2) for v in vals] for k, vals in c["best_of_k"].items()}, skilladam=[round(v, 1) for v in c["skilladam"]]) for b, c in ch.items()}, fig5=rows,
                   path_test_used=bool(path)), open(OUT + "/devchance_numbers.json", "w"), indent=1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
