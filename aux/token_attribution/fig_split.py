"""Figure bip_split: the decomposition on GPT-6-Luna and pooled over skills, cut two ways (behavior, content).

(a) Luna twin of the DeepSeek per-skill heatmap: per skill, the four-term split of the change in log input tokens against the Luna seed (heatmap, same colour scale as the DeepSeek one).
(b) The same split summed over skills, for DeepSeek (all, per benchmark) and Luna (all, per benchmark): stacked signed bars, black tick = measured total with a 95% task-bootstrap interval.
(c) The same totals cut by content (exact split by context component): which context component carries the change (skill text, rest of the prompt, tool outputs, conversation messages).

Data: attribution_luna.json (per-task terms of the paper's four-term split, both models; written by aux/token_attribution/attribution_luna.py) for (a), (b);
bipdata.pkl (3-run mean tokens per component group, same tasks) for (c).  The bootstrap redoes attribution_luna.bootstrap with its own seed.
Run:  python -m aux.token_attribution.fig_split
"""
import collections
import json
import math
import pickle

import numpy as np

from aux.token_attribution.bipcommon import *  # noqa: F401,F403

LJ = None
D = None
B = 10000
RNG = np.random.default_rng(20261023)
OPT = [("GEPA", "gepa"), ("SkillOpt", "skillopt"), ("SkillAdam", "skilladam")]
BENCH = [("tau3-bench", "tau3"), ("automationbench", "AutoB"), ("terminal-bench-2-1", "TB2")]
CGROUPS = ["skill", "prompt", "tools", "messages"]
CLABEL = dict(skill="skill text", prompt="rest of the prompt", tools="tool outputs", messages="agent and user messages")
CCOLOR = dict(skill="#D55E00", prompt="#BBBBBB", tools="#009E73", messages="#56B4E9")
GI = {"prompt": 0, "skill": 1, "tools": 2, "messages": 3}   # column order in bipdata.pkl
OPT_COLOR = dict(GEPA="#2E9E44", SkillOpt="#1F6FB5", SkillAdam="#F28E1C")


def logmean(a, b):
    return np.where(np.isclose(a, b), a, (b - a) / np.log(b / a))


def cells_of(model):
    key = "luna" if model == "gpt-6-luna" else "deepseek"
    out = []
    for c in LJ[key]:
        c = dict(c)
        c["m"] = dict(GEPA="gepa", SkillOpt="skillopt", SkillAdam="skilladam")[c["method"]]
        c["model"] = model
        out.append(c)
    return out


def content_terms(c):
    """Per-task content split of one cell: {group: array over tasks}; also checks that the groups sum to the measured change."""
    cell = D["cells"][(c["model"], c["bench"], c["m"])]
    assert cell["tasks_rebuilt"] == c["tasks"], (c["model"], c["bench"], c["m"])
    X0 = np.array([cell["seed"][t].sum() for t in c["tasks"]])
    X1 = np.array([cell["own"][t].sum() for t in c["tasks"]])
    mu = logmean(X0, X1)
    terms = {g: np.array([(cell["own"][t][GI[g]] - cell["seed"][t][GI[g]]) for t in c["tasks"]]) / mu for g in CGROUPS}
    total = sum(terms.values())
    meas = np.log(X1 / X0)
    err = float(np.abs(total - meas).max())
    err_json = float(np.abs(meas - np.array(c["dX"])).max())
    return terms, meas, err, err_json


def scope_sums(cells, kind):
    """Sum over cells of the per-cell task mean of each term (the paper's pooled sum)."""
    if kind == "behavior":
        s = {k: sum(float(np.mean(c["terms"][k])) for c in cells) for k in TERMS}
    else:
        s = collections.defaultdict(float)
        for c in cells:
            for g, v in c["cterms"].items():
                s[g] += float(np.mean(v))
        s = dict(s)
    s["total"] = sum(float(np.mean(c["dX"])) for c in cells)
    return s


def bootstrap(cells):
    """attribution_luna.bootstrap: resample tasks within each benchmark, same draw for every skill of that benchmark; each cell averages over the drawn tasks it uses."""
    by_bench = collections.defaultdict(list)
    for c in cells:
        by_bench[c["bench"]].append(c)
    keys = TERMS + ["total"]
    acc = {k: np.zeros(B) for k in keys}
    for bench, cs in by_bench.items():
        pool = sorted({t for c in cs for t in c["tasks"]})
        n = len(pool)
        counts = RNG.multinomial(n, np.full(n, 1 / n), size=B)
        for c in cs:
            idx = np.array([pool.index(t) for t in c["tasks"]])
            w = counts[:, idx].astype(float)
            ws = w.sum(1)
            ws[ws == 0] = np.nan
            for k in keys:
                v = np.array(c["dX"] if k == "total" else c["terms"][k])
                acc[k] += (w @ v) / ws
    turns = acc["more_actions"] + acc["turn_splitting"]
    return dict(total=[float(np.nanpercentile(acc["total"], q)) for q in (2.5, 97.5)],
                turns=[float(np.nanpercentile(turns, q)) for q in (2.5, 97.5)],
                text=[float(np.nanpercentile(acc["skill_text"], q)) for q in (2.5, 97.5)],
                p_turns_gt_text=float(np.mean(turns > acc["skill_text"])))


def build():
    res = dict(scopes=[], checks=[])
    allc = {}
    for model in ("deepseek-flash", "gpt-6-luna"):
        cells = cells_of(model)
        for c in cells:
            c["cterms"], meas, err, err_json = content_terms(c)
            res["checks"].append(dict(model=model, method=c["method"], bench=c["short"], n=c["n"], max_abs_content_vs_measured=err, max_abs_measured_vs_json=err_json,
                                      content_mean={g: float(np.mean(v)) for g, v in c["cterms"].items()}))
        allc[model] = cells
        scopes = [("all", cells)] + [(short, [c for c in cells if c["short"] == short]) for _, short in BENCH if any(c["short"] == short for c in cells)]
        for name, cs in scopes:
            b = bootstrap(cs)
            res["scopes"].append(dict(model=model, scope=name, n_skills=len(cs), n_tasks=sorted({(c["bench"], t) for c in cs for t in c["tasks"]}).__len__(),
                                      behavior=scope_sums(cs, "behavior"), content=scope_sums(cs, "content"), boot=b))
    return res, allc


def draw(res, allc):
    plt = setup_mpl()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(7.1, 3.05))
    # ---------------- (a) Luna heatmap
    axa = fig.add_axes([0.085, 0.13, 0.215, 0.62])
    cells = allc["gpt-6-luna"]
    order = [(o, b) for o, _ in OPT for b in ("tau3", "AutoB")]
    M, labels = [], []
    for o, b in order:
        c = next(x for x in cells if x["method"] == o and x["short"] == b)
        M.append([c["A"][k] for k in TERMS] + [c["A"]["total"]])
    M = np.array(M)
    ds_max = max(abs(x) for c in allc["deepseek-flash"] for x in [c["A"][k] for k in TERMS] + [c["A"]["total"]])
    v = ds_max   # same colour scale as the DeepSeek heatmap of the paper (largest |cell| there)
    im = axa.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axa.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6.8, color="white" if abs(M[i, j]) > 0.62 * v else INK,
                     fontweight="bold" if j == 4 else "normal")
    axa.set_xticks(range(5))
    axa.set_xticklabels(["more\nactions", "turn-\nsplitting", "content\nper turn", "skill\ntext", "total\n(measured)"], fontsize=6.3, linespacing=0.95)
    axa.xaxis.set_ticks_position("top")
    axa.tick_params(axis="x", length=0, pad=2)
    axa.set_yticks(range(len(order)))
    axa.set_yticklabels([BENCH_NAME[b] for _, b in order], fontsize=7)
    axa.tick_params(axis="y", length=0, pad=2)
    axa.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    axa.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    axa.grid(which="minor", color="white", lw=1.0)
    axa.tick_params(which="minor", length=0)
    for sp in axa.spines.values():
        sp.set_visible(False)
    for k, (o, _) in enumerate(OPT):
        axa.text(-0.36, 2 * k + 0.5, o, transform=axa.get_yaxis_transform(), ha="center", va="center", rotation=90, fontsize=7, fontweight="bold",
                 color=OPT_COLOR[o], clip_on=False)
    for y in (1.5, 3.5):
        axa.plot([-0.5, 4.5], [y, y], color=INK, lw=0.6, clip_on=False)
    fig.text(0.085, 0.925, "(a) GPT-6-Luna, per skill", fontsize=8, fontweight="bold", ha="left", va="center", color=INK)
    fig.text(0.085, 0.040, "log units, colour scale as in the DeepSeek heatmap.\nTB2 has no Luna row: every optimizer returned the seed.", fontsize=6.0, ha="left", va="center", color=MUTED, linespacing=1.1)

    # ---------------- (b), (c): rows (a header row per model, then the pooled row and one row per benchmark)
    rows = [("hdr", "deepseek-flash"), ("deepseek-flash", "all"), ("deepseek-flash", "tau3"), ("deepseek-flash", "AutoB"), ("deepseek-flash", "TB2"),
            ("hdr", "gpt-6-luna"), ("gpt-6-luna", "all"), ("gpt-6-luna", "tau3"), ("gpt-6-luna", "AutoB")]
    ypos, hdr = {}, {}
    y = 0.0
    for r in rows:
        if r[0] == "hdr":
            hdr[r[1]] = y
            y += 0.8
            continue
        ypos[r] = y
        y += 1.0
    scope = {(s["model"], s["scope"]): s for s in res["scopes"]}
    xmin, xmax = -0.35, 2.3
    axb = fig.add_axes([0.40, 0.12, 0.275, 0.75])
    axp = fig.add_axes([0.678, 0.12, 0.05, 0.75], sharey=axb)
    axc = fig.add_axes([0.745, 0.12, 0.245, 0.75], sharey=axb, sharex=axb)
    bar_h = 0.62
    for ax, kind in ((axb, "behavior"), (axc, "content")):
        for r, yy in ypos.items():
            s = scope[r]
            vals = s[kind]
            keys = TERMS if kind == "behavior" else CGROUPS
            colors = TERM_COLOR if kind == "behavior" else CCOLOR
            pos, neg = 0.0, 0.0
            for k in keys:
                x = vals[k]
                if x >= 0:
                    ax.barh(yy, x, left=pos, height=bar_h, color=colors[k], ec="white", lw=0.4, zorder=3)
                    pos += x
                else:
                    ax.barh(yy, x, left=neg, height=bar_h, color=colors[k], ec="white", lw=0.4, zorder=3)
                    neg += x
            # measured total with its 95% task-bootstrap interval
            lo, hi = s["boot"]["total"]
            ax.plot([lo, hi], [yy, yy], color=INK, lw=0.9, zorder=5, solid_capstyle="butt")
            ax.plot([vals["total"]], [yy], marker="D", ms=3.6, color=INK, mec="white", mew=0.5, zorder=6)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(y - 0.5, -0.9)
        ax.axvline(0, color=INK, lw=0.6, zorder=4)
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_xticklabels(["0", "0.5", "1.0", "1.5", "2.0"], fontsize=7)
        ax.set_xlabel("change in log input tokens, summed over skills", fontsize=7)
    names = {"all": None, "tau3": r"$\tau^3$-bench", "AutoB": "AutoBench", "TB2": "TB2"}
    axb.set_yticks([ypos[r] for r in ypos])
    lab = []
    for r in ypos:
        s = scope[r]
        lab.append(f"all {s['n_skills']} skills" if r[1] == "all" else names[r[1]])
    axb.set_yticklabels(lab, fontsize=7)
    for model, yy in hdr.items():
        axb.text(xmin + 0.02, yy, MODEL_NAME[model], ha="left", va="center", fontsize=7.3, fontweight="bold", color=INK, zorder=9,
                 bbox=dict(fc="white", ec="none", pad=1.2))
    plt.setp(axc.get_yticklabels(), visible=False)
    axc.tick_params(axis="y", length=0)
    # narrow column between (b) and (c): share of bootstrap draws in which the turn terms exceed the skill text
    axp.set_xlim(0, 1)
    axp.axis("off")
    for r, yy in ypos.items():
        axp.text(0.5, yy, f"{100 * scope[r]['boot']['p_turns_gt_text']:.0f}%", fontsize=6.8, color=INK, ha="center", va="center")
    axp.text(0.5, list(hdr.values())[0] - 0.1, "turns >\nskill text", fontsize=5.8, color=MUTED, ha="center", va="center", linespacing=0.95)
    fig.text(0.40, 0.925, "(b) By behavior: the four terms", fontsize=8, fontweight="bold", ha="left", va="center", color=INK)
    fig.text(0.745, 0.925, "(c) By content: the components", fontsize=8, fontweight="bold", ha="left", va="center", color=INK)
    # legends inside the empty lower right of each panel
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    hb = [Patch(fc=TERM_COLOR[k], label=TERM_LABEL[k]) for k in TERMS] + [Line2D([0], [0], color=INK, lw=0.9, marker="D", ms=3.4, label="measured total, 95% interval")]
    axb.legend(handles=hb, loc="lower right", frameon=False, fontsize=6.4, handlelength=0.9, handleheight=0.7, labelspacing=0.28, borderaxespad=0.1, bbox_to_anchor=(1.0, 0.0))
    hc = [Patch(fc=CCOLOR[g], label=CLABEL[g]) for g in CGROUPS]
    axc.legend(handles=hc, loc="lower right", frameon=False, fontsize=6.4, handlelength=0.9, handleheight=0.7, labelspacing=0.28, borderaxespad=0.1, bbox_to_anchor=(1.0, 0.0))
    save(fig, "bip_split")
    plt.close(fig)


def load():
    """Read the inputs this figure draws from."""
    global LJ, D
    LJ = json.loads((OUT / "attribution_luna.json").read_text())
    D = pickle.loads((OUT / "bipdata.pkl").read_bytes())


def main():
    load()
    res, allc = build()
    for ch in res["checks"]:
        print("check", ch["model"], ch["method"], ch["bench"], "content vs measured %.2e" % ch["max_abs_content_vs_measured"], "measured vs json %.2e" % ch["max_abs_measured_vs_json"])
    for s in res["scopes"]:
        b = s["behavior"]
        print(s["model"], s["scope"], "n skills", s["n_skills"], "total %.3f" % b["total"], {k: round(b[k], 3) for k in TERMS}, "| content", {k: round(v, 3) for k, v in s["content"].items() if k != "total"},
              "| boot total", [round(x, 3) for x in s["boot"]["total"]], "turns", [round(x, 3) for x in s["boot"]["turns"]], "P", round(s["boot"]["p_turns_gt_text"], 3))
    (OUT / "bip_split_numbers.json").write_text(json.dumps(res, indent=1, default=float))
    draw(res, allc)


if __name__ == "__main__":
    main()
