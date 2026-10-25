"""The path-test figure: along the tau3 paths of SkillOpt and GEPA (DeepSeek-V4.1-Flash), the reported dev gain climbs while the test difference
to the seed rerun stays within a few points.

Dev gain = the optimizer's own reported gain, points over its starting score (SkillOpt baseline_selection_hard, GEPA val_scores[0], from the
run's summary.json); the skill's dev score is from path_test_detail.json (python -m aux.path_test.collect_path_test). Test difference, tasks
gained / lost and the paired SE: from the path-test passes (one pass each on the updated DeepSeek release) against the seed rerun on that
release, verified outcomes only.
Run:  python -m aux.path_test.pathtest_fig   -> fig/mech/mech_pathtest.{pdf,svg}, data/mech/mech_pathtest_numbers.json
"""
import glob, json, math

from aux.path_test import run_path_test as R
from src import layout
from src.utils import logger
from src.utils.dataloader import SEED_TEXT, vid
from src.utils.fig import save_figure

DETAIL = layout.OUT / "path_test/path_test_detail.json"
COL = {"gepa": "#2A9D4B", "skillopt": "#1F6FB5"}
NAME = {"gepa": "GEPA", "skillopt": "SkillOpt"}
INK, MUTED, GRID = "#222222", "#5A5A5A", "#E2E5EA"
# (round, returned?, dev gain, difference, gained, lost, SE) of each drawn point; compute() checks the passes reproduce them
FROZEN = {"skillopt": [(1, False, 6.7, 4.0, 7, 4, 4.4), (2, False, 10.0, 6.7, 9, 4, 4.7), (4, True, 13.3, 1.3, 7, 6, 4.8)],
          "gepa": [(8, False, 10.0, 4.0, 8, 5, 4.8), (11, False, 13.3, 5.3, 6, 2, 3.7), (18, True, 16.7, 5.3, 6, 2, 3.7)]}
# centre height (data units) of each test point's two-line label: the middle of the grid gap holding the point
LABEL_Y = {"skillopt": {1: 2.5, 2: 7.5, 4: 2.5}, "gepa": {8: 2.5, 11: 7.5, 18: 7.5}}


def load(d, version):
    f = glob.glob(f"{R.OUT}/deepseek-flash/{d}/results/evaluations/test/{version[:12]}-*/evaluation.json")
    assert len(f) == 1, (d, version, f)
    return {e["task_id"]: bool(e["success"]) for e in json.load(open(f[0]))["episodes"]
            if e.get("outcome_valid", True) is True and type(e.get("success")) is bool}


def compute():
    if not DETAIL.exists():
        raise FileNotFoundError(f"{DETAIL} is missing; run python -m aux.path_test.collect_path_test first")
    seed = load("seed__tau3-bench", vid(SEED_TEXT))
    assert (sum(seed.values()), len(seed)) == (53, 75), (sum(seed.values()), len(seed))
    detail = [r for r in json.load(open(DETAIL)) if r["model"] == "deepseek-flash" and r["bench"] == "tau3-bench" and r["test"]]
    out = {}
    for opt in ("skillopt", "gepa"):
        summ = json.load(open(R.EXECUTION / "deepseek-flash" / opt / "tau3-bench" / R.RUNS["deepseek-flash"] / "summary.json"))["extra"]
        start = summ["baseline_selection_hard"] if opt == "skillopt" else summ["val_scores"][0]
        rows = []
        for r in sorted((r for r in detail if r["opt"] == opt), key=lambda r: r["round"]):
            s = load(f"{opt}__tau3-bench", r["vid"])
            t = [k for k in seed if k in s]
            n = len(t)
            g = sum(1 for k in t if not seed[k] and s[k]); l = sum(1 for k in t if seed[k] and not s[k])
            se = math.sqrt(g + l - (g - l) ** 2 / n) / n * 100
            dev = 100 * (r["dev"][0] / r["dev"][1] - start)
            rows.append(dict(round=r["round"], returned=r["role"] == "returned", vid=r["vid"], dev_solved=r["dev"][0], start=start, n=n,
                             dev_gain=round(dev, 1), diff=round(100 * (g - l) / n, 1), gained=g, lost=l, se=round(se, 1), solved=sum(s.values())))
        out[opt] = rows
        got = [(x["round"], x["returned"], x["dev_gain"], x["diff"], x["gained"], x["lost"], x["se"]) for x in rows]
        assert got == FROZEN[opt], (opt, got, FROZEN[opt])
    a, b = (load("gepa__tau3-bench", out["gepa"][i]["vid"]) for i in (1, 2))
    out["gepa_r11_vs_returned_tasks_differ"] = sum(1 for k in a if k in b and a[k] != b[k])
    out["seed_rerun"] = dict(solved=sum(seed.values()), n=len(seed))
    return out


def draw(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 7, "mathtext.fontset": "stix", "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.25), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.15, top=0.725, wspace=0.12)
    dx = 0.11
    titles = {}
    for ax, opt in zip(axes, ("skillopt", "gepa")):
        rows, c = res[opt], COL[opt]
        x = list(range(len(rows)))
        dev = [r["dev_gain"] for r in rows]
        dif = [r["diff"] for r in rows]
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        ax.plot([i - dx for i in x], dev, color=c, lw=1.1, zorder=3)
        ax.scatter([i - dx for i in x], dev, s=22, marker="o", facecolor=c, edgecolor=c, lw=0.8, zorder=4)
        for i, v in zip(x, dev):
            ax.text(i - dx, v + 1.0, f"+{v:.1f}", ha="center", va="bottom", fontsize=6.8, color=c)
        ax.errorbar([i + dx for i in x], dif, yerr=[r["se"] for r in rows], fmt="none", ecolor=c, elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=3)
        ax.scatter([i + dx for i in x], dif, s=22, marker="o", facecolor="white", edgecolor=c, lw=1.0, zorder=4)
        for i, r in zip(x, rows):
            yc = LABEL_Y[opt][r["round"]]
            ax.annotate(f"+{r['diff']:.1f}", (i + dx, yc), xytext=(5, 0.5), textcoords="offset points", ha="left", va="bottom", fontsize=6.8, color=c)
            ax.annotate(f"{r['gained']}/{r['lost']}", (i + dx, yc), xytext=(5, -0.5), textcoords="offset points", ha="left", va="top", fontsize=6.8, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([f"round {r['round']}" + (", returned" if r["returned"] else "") for r in rows])
        ax.set_xlim(-0.45, len(rows) - 0.4)
        ax.set_ylim(-5, 20.5)
        ax.set_yticks([0, 5, 10, 15])
        ax.grid(axis="y", color=GRID, lw=0.4); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=3)
        ddev = 100 * (rows[-1]["dev_solved"] - rows[0]["dev_solved"]) / 30
        tests = [100 * (r["gained"] - r["lost"]) / r["n"] for r in rows]
        within = max(abs(t - tests[0]) for t in tests)
        titles[opt] = (f"{NAME[opt]}: dev gain climbs {ddev:.1f} points,\n"
                       f"test difference stays within {within:.1f} points of its first value")
        ax.set_title(titles[opt], loc="left", fontsize=7.5, color=INK, pad=4, linespacing=1.15)
    axes[0].set_ylabel("points", labelpad=2)
    h = [Line2D([], [], color=INK, lw=1.1, marker="o", ms=4, mfc=INK, mec=INK),
         Line2D([], [], color="none", marker="o", ms=4, mfc="white", mec=INK)]
    fig.legend(h, ["reported dev gain, over the starting score", r"test difference to the seed rerun, $\pm$1 SE, and tasks gained/lost"],
               loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=2, frameon=False, fontsize=7, handlelength=2.2, columnspacing=2.0, handletextpad=0.5)
    save_figure(fig, "pathtest")
    return titles


def main():
    res = compute()
    res["titles"] = draw(res)
    logger.DATA.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(logger.DATA / "mech_pathtest_numbers.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
