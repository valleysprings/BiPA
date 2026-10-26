"""The figures of dev selection: Figure 4 (mech_dev2test4) and the appendix figures mech_selection, mech_devpertask and
mech_priorfit, with the records mech_devdyn (the numbers of Figure 4 and of the per-task comparison) and mech_priorfit.

Figure 4, DeepSeek-V4.1-Flash. (a) Each run's reported dev gain against the gain its own acceptance rule reports when
every candidate is unchanged (aux.selection.replay, fitted task model, the run's starting score held fixed): grey bar,
the middle nine tenths of the replayed gains; tick, their mean. SkillAdam's gain is the sum of its accepted edits'
margins on small batches, so it has its own axis. (b) The test difference of each returned skill to the seed (Table 1,
avg@3) against its reported dev gain, with the least-squares slope and the correlation over the nine skills.

mech_selection: the distribution of the dev gain, in tasks, that SkillOpt's and GEPA's rules report for unchanged
candidates under the fitted task model, from a random seed score (a) and from the run's own starting score (b); lines:
the gains the runs reported.

mech_devpertask: inside each SkillOpt and GEPA run that returned a learned skill, which evaluated the seed and the
returned skill once each on the whole dev set and on the test set, the share of tasks the seed fails and the returned
skill solves (up) and the reverse (down); verified outcomes only. Ticks: the share a second run of the seed flips in
each direction, the mean over tasks of k(n - k) / (n(n - 1)) for a task solved k of n times in the seed's executions.
SkillAdam scores no returned skill on the whole dev set.

mech_priorfit: the share of tasks the seed solves in none, some or all of its executions, dev and test, against the
share the beta-binomial prior fitted to them expects, each task with its own number of executions."""
import json, math
from collections import defaultdict

import numpy as np

from aux.selection import taskmodel as T
from src.analysis import settings as S
from src.utils import fig as F
from src.utils.logger import md_table, record

DS, LUNA = S.DEEPSEEK, S.LUNA
ORDER = ("tau3-bench", "automationbench", "terminal-bench-2-1")
CACHE = S.cache("selection")


def load(name):
    return json.loads((CACHE / name).read_text())


# ---- Figure 4

def returned(panel):
    """Panel (b): the nine returned skills' reported dev gain and test difference to the seed, points."""
    t1 = json.loads((S.cache("table1") / "table1.json").read_text())
    G = {(r["bench"], r["opt"]): r["G"] for r in panel}
    pts = [dict(bench=b, opt=o, dev=round(G[b, o], 1), test=round(t1[f"{DS}|{b}"][S.ARM_NAME[o]]["Q"] - t1[f"{DS}|{b}"]["Seed skill"]["Q"], 1))
           for _, b, o in S.learned((DS,))]
    fit = lambda a: (round(float(np.polyfit(a[:, 0], a[:, 1], 1)[0]), 3), round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3))
    (s9, c9), (s6, c6) = (fit(np.array([(p["dev"], p["test"]) for p in pts if keep(p)])) for keep in (bool, lambda p: p["opt"] != "skilladam"))
    return dict(returned=pts, nine_slope=s9, nine_corr=c9, skillopt_gepa_slope=s6, skillopt_gepa_corr=c6)


def mark(ax, x, y, opt, bench, s):
    ax.scatter([x], [y], s=s * (0.85 if bench == "automationbench" else 1.0), marker=F.BENCH_MARKER[bench], facecolor=F.ARM[opt],
               edgecolor="white", lw=0.6, zorder=4, clip_on=False)


def frame(ax, grid):
    ax.grid(axis=grid, color=F.GRID, lw=0.4)
    ax.set_axisbelow(True)
    F.despine(ax)
    ax.tick_params(pad=1.5)


def forest(ax, opts, xlabel, xlim, xticks, R, h=0.30):
    import matplotlib.transforms as mt
    from matplotlib.patches import Rectangle
    ys, y = [], 0.0
    for opt in opts:
        y0 = y
        for bench in ORDER:
            r = R[bench, opt]
            ax.add_patch(Rectangle((r["p5"], y - h), max(r["p95"] - r["p5"], 0.0), 2 * h, facecolor=F.SLATE["bar"], edgecolor="none", zorder=1))
            ax.plot([r["mean"]] * 2, [y - 1.45 * h, y + 1.45 * h], color=F.SLATE["tick"], lw=1.1, zorder=2, solid_capstyle="butt")
            mark(ax, r["G"], y, opt, bench, 34)
            ys.append(y)
            y += 1.0
        ax.text(-0.36, (y0 + y - 1) / 2, S.ARM_NAME[opt], transform=mt.blended_transform_factory(ax.transAxes, ax.transData), ha="right",
                va="center", fontsize=F.TEXT, fontweight="bold", color=F.ARM[opt])
        y += 0.5
    ax.set_yticks(ys, [F.BENCH_LABEL[b] for _ in opts for b in ORDER])
    ax.set(ylim=(y - 0.85, -0.65), xlim=xlim, xticks=xticks)
    frame(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel(xlabel, labelpad=1.5)


def dev2test():
    from matplotlib.lines import Line2D
    plt = F.pyplot(F.TEXT, tick=F.SMALL, label=F.SMALL, legend=6.8)
    num = load("dev2test.json")
    num["panel_b"] = b = returned(num["panel_a"])
    (CACHE / "dev2test.json").write_text(json.dumps(num, indent=1))
    R, ht = {(r["bench"], r["opt"]): r for r in num["panel_a"]}, 2.30
    fig = plt.figure(figsize=(F.TEXT_W, ht))
    forest(fig.add_axes([0.215, 1.24 / ht, 0.275, 0.88 / ht]), ("skillopt", "gepa"), "dev gain, points", (-1, 27), range(0, 26, 5), R)
    forest(fig.add_axes([0.215, 0.52 / ht, 0.275, 0.39 / ht]), ("skilladam",), "SkillAdam dev gain, points", (-5, 140), range(0, 141, 20), R)
    fig.text(0.008, 2.21 / ht, "(a) Unchanged candidates reach every reported dev gain", fontweight="bold", va="center")
    handles = [Line2D([0], [0], marker="o", color=F.MUTED, lw=0, markersize=4.5, label="marker: reported dev gain"),
               Line2D([0], [0], color=F.SLATE["bar"], lw=4.5, solid_capstyle="butt", label="bar: 5 to 95% of the gain unchanged candidates report"),
               Line2D([0], [0], color=F.SLATE["tick"], lw=0, marker="|", markersize=7, markeredgewidth=1.1, label="tick: its mean")]
    fig.legend(handles=handles, loc="lower left", ncol=3, frameon=False, handlelength=1.0, columnspacing=0.7, handletextpad=0.3,
               bbox_to_anchor=(0.008, 0.02 / ht), borderaxespad=0.0, borderpad=0.0)
    bx = fig.add_axes([0.655, 0.66 / ht, 0.335, 1.46 / ht])
    bx.axhline(0, color=F.SLATE["tick"], lw=0.7, zorder=1)
    for p in b["returned"]:
        mark(bx, p["dev"], p["test"], p["opt"], p["bench"], 52)
    bx.set(xlim=(-1.5, 43), ylim=(-13, 6), yticks=[-10, -5, 0, 5], xticks=[0, 10, 20, 30, 40])
    bx.text(42.5, 5.6, f"slope {b['nine_slope']:.2f}, correlation {b['nine_corr']:.2f}", fontsize=6.8, color=F.SLATE["note"], ha="right", va="top")
    bx.set_xlabel("reported dev gain, points", labelpad=1.5)
    bx.set_ylabel("test difference to the seed, points", labelpad=1)
    frame(bx, "y")
    fig.text(0.585, 2.21 / ht, "(b) Test does not follow the dev gain", fontweight="bold", va="center")
    hb = [h for opt, bench in zip(("skillopt", "gepa", "skilladam"), ORDER) for h in (
        Line2D([0], [0], marker="o", color=F.ARM[opt], lw=0, markersize=5, label=S.ARM_NAME[opt]),
        Line2D([0], [0], marker=F.BENCH_MARKER[bench], color=F.MUTED, lw=0, markersize=4.5 if bench == "automationbench" else 5, label=F.BENCH_LABEL[bench]))]
    fig.legend(handles=hb, loc="lower left", ncol=3, frameon=False, handlelength=0.8, columnspacing=1.2, handletextpad=0.3, labelspacing=0.35,
               bbox_to_anchor=(0.70, 0.02 / ht), borderaxespad=0.0, borderpad=0.0)
    F.save_figure(fig, "dev2test4")
    return num


# ---- the replayed gain distributions

def selection():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    plt = F.pyplot(F.TEXT, title=F.TITLE, tick=F.SMALL)
    res = load("selection.json")
    look = {"SkillOpt": dict(offset=-0.2, face=F.ARM["skillopt"], edge=F.ARM["skillopt"], hatch=None, line="-", marker="v"),
            "GEPA": dict(offset=0.2, face="white", edge=F.ARM["gepa"], hatch="////", line="--", marker="D")}
    fig, axs = plt.subplots(2, 3, figsize=(F.TEXT_W, 4.3))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.1, top=0.83, wspace=0.3, hspace=0.85)
    for row, (key, title) in enumerate((("hist_random_baseline", "(a) random seed draw as baseline"),
                                        ("hist_observed_baseline", "(b) baseline = the run's own seed draw"))):
        for ax, bench in zip(axs[row], ("terminal-bench-2-1", "tau3-bench", "automationbench")):
            r = res[bench]
            g = r["priors"]["fitted"]["rules"]
            hists = {rule: np.array(g[rule][key], dtype=float) / sum(g[rule][key]) for rule in look}
            top = max(max(int(np.nonzero(h > 0.002)[0].max()) for h in hists.values()), max(r["observed"][k]["gain"] for k in look)) + 1
            ymax = max(h[:top + 1].max() for h in hists.values())
            for rule, lk in look.items():
                ax.bar(np.arange(top + 1) + lk["offset"], hists[rule][:top + 1], width=0.38, color=lk["face"], edgecolor=lk["edge"], linewidth=0.8,
                       hatch=lk["hatch"])
                x = r["observed"][rule]["gain"] + lk["offset"]
                ax.plot([x, x], [0, ymax * 1.08], color=F.INK, lw=1.8, ls=lk["line"], zorder=5)
                ax.plot([x], [ymax * 1.12], marker=lk["marker"], color=F.INK, markersize=4.5, zorder=6, clip_on=False)
            ax.set_title(f"{F.BENCH_LABEL[bench]}  " + (f"(K = {g['SkillOpt']['K']} / {g['GEPA']['K']})" if row == 0
                                                       else f"(baseline {r['observed']['SkillOpt']['baseline']}/{r['dev_tasks']})"), pad=8)
            ax.set(ylim=(0, ymax * 1.18), xlabel="announced dev gain (tasks)", ylabel="probability")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            F.despine(ax)
        axs[row][1].annotate(title, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 21), textcoords="offset points", ha="center",
                             va="bottom", fontsize=F.HEAD, fontweight="bold")
    handles = [Patch(facecolor=F.ARM["skillopt"], edgecolor=F.ARM["skillopt"], label="SkillOpt rule, no effect"),
               Patch(facecolor="white", edgecolor=F.ARM["gepa"], hatch="////", label="GEPA rule, no effect"),
               Line2D([], [], color=F.INK, lw=1.8, ls="-", marker="v", markersize=4.5, label="SkillOpt run's gain"),
               Line2D([], [], color=F.INK, lw=1.8, ls="--", marker="D", markersize=4, label="GEPA run's gain")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.53, 1.0))
    F.save_figure(fig, "selection")


# ---- dev and test, task by task

def outcomes(run, split, version, n):
    """{task: 0/1} of a version's evaluation of a split in a run: the one with most verified outcomes (at least 0.8 n)."""
    best = {}
    for path in sorted((run / "results/evaluations" / split).glob(f"{version[:12]}-*/evaluation.json")):
        got = {e["task_id"]: int(bool(e["success"])) for e in json.loads(path.read_text())["episodes"] if T.verified(e)}
        best = got if len(got) > len(best) else best
    return best if len(best) >= 0.8 * n else None


def flips(a, b):
    common = [t for t in a if t in b]
    return {"gained": sum(a[t] < b[t] for t in common), "lost": sum(a[t] > b[t] for t in common), "n": len(common)}


def per_task():
    """Per SkillOpt and GEPA run that returned a learned skill: tasks gained and lost on dev and on test, and the seed's flips."""
    rows = []
    for model in (DS, LUNA):
        for opt in ("skillopt", "gepa"):
            for bench in ORDER:
                if S.returned_seed(model, bench, opt):
                    continue
                run = S.run(model, bench, opt)
                panels = json.loads((run / "summary.json").read_text())["panels"]
                final = S.version(S.skill(model, bench, opt))
                side = {split: [outcomes(run, split, v, panels[key]) for v in (S.SEED_VERSION, final)] for split, key in (("dev", "sel"), ("test", "test"))}
                rows.append(dict(model=model, opt=opt, bench=bench, dev=flips(*side["dev"]), test=flips(*side["test"])))
    noise = {}
    for bench in ORDER:
        dev, test, _ = T.seed_executions(DS, bench)
        noise[bench] = [100 * float(np.mean([k * (n - k) / (n * (n - 1)) for k, n in ((sum(v), len(v)) for v in ex.values()) if n > 1])) for ex in (dev, test)]
    return rows, noise


def devpertask(rows, noise):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt = F.pyplot(F.TEXT, tick=F.SMALL)
    fig = plt.figure(figsize=(7.1, 2.75))
    bx = fig.add_axes([0.075, 0.27, 0.915, 0.61])
    xt, groups, x, prev = [], defaultdict(list), 0.0, None
    for d in rows:
        x += 0.9 if prev not in (None, d["model"]) else 0.0
        prev, c = d["model"], F.ARM[d["opt"]]
        for off, split, ref in ((-0.2, "dev", noise[d["bench"]][0]), (0.2, "test", noise[d["bench"]][1])):
            f = d[split]
            for v in (100 * f["gained"] / f["n"], -100 * f["lost"] / f["n"]):
                bx.bar(x + off, v, width=0.34, color=c if split == "dev" else "white", edgecolor=c, lw=0.9, zorder=3, hatch="////" if split == "test" else None)
            for v in (ref, -ref):
                bx.plot([x + off - 0.19, x + off + 0.19], [v, v], color=F.INK, lw=0.8, zorder=5)
        xt.append(x)
        groups[d["model"], d["opt"]].append(x)
        x += 1.0
    bx.axhline(0, color=F.INK, lw=0.6)
    bx.set_xticks(xt, [F.BENCH_TICK[d["bench"]] for d in rows], fontsize=6.8)
    bx.tick_params(axis="x", pad=1.5, length=0)
    bx.tick_params(axis="y", pad=1.5)
    bx.set_ylabel("share of tasks, %", fontsize=F.TEXT, labelpad=2)
    bx.set(xlim=(-0.7, x - 0.3), ylim=(-16, 25))
    frame(bx, "y")
    for (model, opt), xs in groups.items():
        bx.text(np.mean(xs), -0.13, S.ARM_NAME[opt], transform=bx.get_xaxis_transform(), ha="center", va="top", fontsize=7.2, fontweight="bold", color=F.ARM[opt])
    for model in (DS, LUNA):
        xs = [x for (m, _), v in groups.items() if m == model for x in v]
        bx.text((min(xs) + max(xs)) / 2, 22.5, S.MODEL_NAME[model], ha="center", fontsize=7.2, color=F.MUTED)
    bx.set_title("Per task: newly solved (up) and newly failed (down), dev and test", fontsize=F.TITLE, loc="left", pad=4, fontweight="bold")
    handles = [Patch(facecolor=F.ARM[o], edgecolor=F.ARM[o], label=S.ARM_NAME[o]) for o in ("gepa", "skillopt")] + [
        Patch(facecolor=F.MID, edgecolor=F.MID, label="dev (solid)"), Patch(facecolor="white", edgecolor=F.MID, hatch="////", label="test (hatched)"),
        Line2D([0], [0], color=F.INK, lw=0.9, label="a second run of the seed")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6.4, handlelength=1.5, columnspacing=1.0, bbox_to_anchor=(0.52, 0.0))
    F.save_figure(fig, "devpertask")


# ---- how often the seed solves a task

def beta_binomial(k, n, a, b):
    return math.comb(n, k) * math.exp(math.lgamma(k + a) + math.lgamma(n - k + b) - math.lgamma(n + a + b) - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b))


def priorfit():
    plt = F.pyplot(F.TEXT, tick=F.SMALL)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), sharey=True)
    res = {}
    for ax, bench in zip(axes, ORDER):
        dev, test, _ = T.seed_executions(DS, bench)
        a, b = T.fit_prior([(sum(v), len(v)) for v in list(dev.values()) + list(test.values())])
        got = {}
        for split, ex in (("dev", dev), ("test", test)):
            counted = [sum(sum(v) == 0 for v in ex.values()), sum(0 < sum(v) < len(v) for v in ex.values()), sum(sum(v) == len(v) for v in ex.values())]
            none, every = (sum(beta_binomial(k, len(v), a, b) for v in ex.values() for k in [0 if end == 0 else len(v)]) for end in (0, 1))
            got[split] = (counted, [none, len(ex) - none - every, every])
        w, xs = 0.36, np.arange(3)
        for off, split, style in ((-w / 2, "dev", dict(color=F.SPLIT["dev"], edgecolor=F.SPLIT["dev"], lw=0.8)),
                                  (w / 2, "test", dict(color="white", edgecolor=F.SPLIT["test"], hatch="////", lw=0.9))):
            n = len(dev if split == "dev" else test)
            ax.bar(xs + off, [100 * c / n for c in got[split][0]], width=w, label=f"{split} tasks", zorder=3, **style)
            ax.scatter(xs + off, [100 * e / n for e in got[split][1]], marker="D", s=13, color=F.INK, zorder=5, label="fitted prior" if split == "dev" else None)
        ax.set_xticks(xs, ["in no run", "in some runs", "in all runs"])
        ax.set_xlabel("task solved", labelpad=2)
        ax.set_title(f"{F.BENCH_LABEL[bench]}   (a, b) = ({a:.2f}, {b:.2f})", fontsize=7.6, loc="left", pad=3, fontweight="bold")
        runs = sorted({len(v) for v in dev.values()})
        ax.text(0.03, 0.93, f"dev: {len(dev)} tasks, {'/'.join(map(str, runs))} runs each\ntest: {len(test)} tasks, 3 runs each", transform=ax.transAxes,
                ha="left", va="top", fontsize=5.8, color=F.MUTED, linespacing=1.15)
        frame(ax, "y")
        res[bench] = dict(a=a, b=b, dev_tasks=len(dev), runs=runs, test_tasks=len(test), dev=got["dev"], test=got["test"])
    axes[0].set_ylabel("share of tasks, %", labelpad=2)
    axes[0].legend(frameon=False, fontsize=6.4, loc="upper left", handlelength=1.4, bbox_to_anchor=(0.0, 0.80))
    fig.tight_layout(w_pad=0.8)
    F.save_figure(fig, "priorfit")
    return res


# ---- records

def record_devdyn(num, rows, noise):
    a = [[F.BENCH_TICK[r["bench"]], S.ARM_NAME[r["opt"]], r["K"], r.get("start", f"batch {r.get('batch')}"), f"{r['G']:.1f}", f"{r['P']:.2f}",
          f"{r['mean']:.1f}", f"{r['p5']:.1f} to {r['p95']:.1f}"] for r in num["panel_a"]]
    b = [[F.BENCH_TICK[p["bench"]], S.ARM_NAME[p["opt"]], f"{p['dev']:.1f}", f"{p['test']:.1f}"] for p in num["panel_b"]["returned"]]
    t = [[S.MODEL_SHORT[d["model"]], S.ARM_NAME[d["opt"]], F.BENCH_TICK[d["bench"]]] + [f"{d[s]['gained']} / {d[s]['lost']} of {d[s]['n']}" for s in ("dev", "test")]
         for d in rows]

    def shares(keep):
        tot = {s: [sum(d[s][k] for d in rows if keep(d)) for k in ("gained", "lost", "n")] for s in ("dev", "test")}
        return tot, {s: [100 * v[0] / v[2], 100 * v[1] / v[2]] for s, v in tot.items()}
    (tot, pct), (_, out) = shares(lambda d: True), shares(lambda d: (d["model"], d["bench"]) != (LUNA, "automationbench"))
    numbers = "\n\n".join([
        "### Figure 4a: reported dev gain and the replay with unchanged candidates (points of the dev set)\n\n"
        + md_table(["bench", "optimizer", "K", "starting score or batch", "reported gain G", "P(G0 >= G)", "mean replayed gain",
                    "replayed gain, middle nine tenths"], a),
        "### Figure 4b: returned skills\n\n" + md_table(["bench", "optimizer", "reported dev gain", "test difference to the seed (Table 1)"], b)
        + f"\n\nLeast-squares slope {num['panel_b']['nine_slope']:.3f}, correlation {num['panel_b']['nine_corr']:.3f} over the {len(b)} skills.",
        "### Dev and test, task by task\n\n" + md_table(["model", "optimizer", "bench", "dev gained / lost", "test gained / lost"], t)
        + f"\n\nOver the {len(t)} skills: dev {tot['dev'][0]} gained and {tot['dev'][1]} lost of {tot['dev'][2]} ({pct['dev'][0]:.1f}% and {pct['dev'][1]:.1f}%); "
          f"test {tot['test'][0]} and {tot['test'][1]} of {tot['test'][2]} ({pct['test'][0]:.1f}% and {pct['test'][1]:.1f}%). Without GPT-6-Luna on AutoBench: "
          f"dev {out['dev'][0]:.1f}% and {out['dev'][1]:.1f}%, test {out['test'][0]:.1f}% and {out['test'][1]:.1f}%. A second run of the seed flips, dev / test: "
          + ", ".join(f"{F.BENCH_TICK[b]} {d:.1f}% / {s:.1f}%" for b, (d, s) in noise.items()) + "."])
    cells = [(m, b, o) for m in (DS, LUNA) for b in ORDER for o in S.OPTIMIZERS if m == DS or o != "skilladam"]
    record(__spec__.name, "devdyn", numbers, S.runs_text(cells),
           what="Figure 4 and the per-task comparison: each DeepSeek-V4.1-Flash run's reported dev gain against the replay with unchanged "
                "candidates, the returned skills' test differences against their dev gains, and, inside each SkillOpt and GEPA run, the tasks the "
                "returned skill newly solves and newly fails on dev and on test.")


def record_priorfit(res):
    rows = [[F.BENCH_TICK[b], r["dev_tasks"], r["runs"], r["test_tasks"], f"({r['a']:.2f}, {r['b']:.2f})"]
            + [" / ".join(f"{v:.0f}" if i == 0 else f"{v:.1f}" for v in r[s][i]) for s in ("dev", "test") for i in (0, 1)] for b, r in res.items()]
    record(__spec__.name, "priorfit", md_table(["bench", "dev tasks", "seed executions per dev task", "test tasks", "fitted prior (a, b)",
                                                "dev tasks solved in none / some / all", "expected", "test tasks solved in none / some / all", "expected"], rows),
           S.runs_text([(DS, b, "native-seed") for b in ORDER], [p for b in ORDER for p in S.extra("seed_dev_pool", DS, b)]),
           what="How often the seed solves each task, counted and as the fitted beta-binomial prior expects (DeepSeek-V4.1-Flash): the check of the "
                "task model behind the selection replay.")


def main():
    num = dev2test()
    selection()
    rows, noise = per_task()
    devpertask(rows, noise)
    record_devdyn(num, rows, noise)
    record_priorfit(priorfit())


if __name__ == "__main__":
    main()
