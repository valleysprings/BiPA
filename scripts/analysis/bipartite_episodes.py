"""Bipartite episode graphs (request x harness component) per (model, method, bench) cell, stored under
assets/artifacts/attribution/<model>/<method>/<bench>/bipartite/.
Stage 1 (slim venv python): bipartite_episodes.py compute [model bench method ...] -> episode-<task>.json, components.json (+ summary.json)
Stage 2 (system python3, matplotlib): bipartite_episodes.py plot -> .pdf + .png next to every json.
Per cell: the test task whose per-task log ratio log(c_skill/c_seed) is closest to the cell's median (tasks with known outcome and complete capture on
both sides, the table's own pairs); its seed and skill episode as request x component graphs (invocation edge = tool call issued by a request, exposure
edge = tokens of a component inside a request, width ~ tokens); components.json = lmdi split of the cell's test log-workload change, grouped as fig_components.py.
native-seed: only the seed episode, on the task the other methods of the cell use (else the median-c task).
ours = the zo run's returned node; when it returned the seed, the non-seed front node with the highest selection Q (recorded as fallback)."""
import glob, json, math, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ATTR = ROOT / "assets/artifacts/attribution"
MODELS = ("deepseek-flash", "gpt-6-luna")
BENCHES = ("tau3-bench", "automationbench", "terminal-bench-2-1")
METHODS = ("native-seed", "skillopt", "gepa", "skilladam", "ours")
GROUPS = ["Reads & retrieval", "Bash & other tools", "History", "Skill text", "Fixed context"]
READ = ("tool:read.", "tool:KB_search", "tool:lookup", "tool:discover", "tool:grep", "tool:glob", "tool:bash.read")


def group(g):
    if g.startswith(READ): return GROUPS[0]
    if g.startswith("tool:"): return GROUPS[1]
    return {"history": GROUPS[2], "skill": GROUPS[3]}.get(g, GROUPS[4])


def outdir(m, meth, b): return ATTR / m / meth / b / "bipartite"
def safe(t): return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in t)


def episode_json(e):
    inv = Counter((k, g) for k, g, _, _, _ in e.calls)
    exp = [{"request": r, "component": g, "tokens": float(e.Xrg[r, j])} for r in range(e.R) for j, g in enumerate(e.comps) if e.Xrg[r, j] > 0]
    cg = e.c_g()
    return {"task": e.task, "success": e.success, "requests": e.R, "calls": e.N, "input_tokens": e.X, "c": e.c,
            "req_input_tokens": [float(x) for x in e.req_in], "components": {g: {"group": group(g), "c_g": float(v)} for g, v in cg.items()},
            "invocation_edges": [{"request": k, "component": g, "calls": n} for (k, g), n in sorted(inv.items())],
            "exposure_edges": exp,
            "calls_detail": [{"request": k, "component": g, "exposure_tokens_later": x} for k, g, _, x, _ in e.calls]}


def compute(sel=None):
    sys.path[:0] = [str(ROOT / "bach-harness/src"), str(HERE), str(HERE.parents[1])]
    import numpy as np
    import main_table as MT
    from bach import decompose as D, lineage as LN, workload as M
    from bipa.layout import SKILLS, current_run
    seed = MT.version((SKILLS / "seed/SKILL.md").read_text())
    ours = json.loads((ROOT / "bach-harness/config/ours_runs.json").read_text())
    path = lambda run, vid: max(glob.glob(f"{run}/results/evaluations/test/{vid[:12]}*/evaluation.json"), key=lambda f: Path(f).stat().st_mtime)
    has = lambda run, vid: bool(glob.glob(f"{run}/results/evaluations/test/{vid[:12]}*/evaluation.json"))
    summary, picked = [], {}
    for m in MODELS:
        for b in BENCHES:
            cand = []  # (method, run, vid, note)
            for meth, arm in (("skillopt", "skillopt"), ("gepa", "gepa"), ("skilladam", None)):
                run = current_run(arm or "skilladam", m, b)
                if run and (run / "summary.json").is_file():
                    cand.append((meth, run, MT.version(json.loads((run / "summary.json").read_text())["final_skill"]), None))
                else:
                    summary.append({"model": m, "bench": b, "method": meth, "status": "skipped", "reason": "no finished run (summary.json missing)"})
            zo = current_run("zo", m, b)
            if (zo / "zo-result.json").is_file():
                d = json.loads((zo / "zo-result.json").read_text())
                nodes = {n["id"]: n for n in d["nodes"]}
                note, n = None, nodes[d["returned"]]
                if n["version"][:12] == seed[:12]:
                    alt = [nodes[i] for i in d["front"] if nodes[i]["version"][:12] != seed[:12] and nodes[i].get("sel")]
                    n = max(alt, key=lambda x: x["sel"]["Q"], default=None)
                    note = f"returned node {d['returned']} is the seed; fallback = non-seed front node with highest selection Q"
                if n: cand.append(("ours", zo, n["version"], f"{ours[m][b]} node {n['id']}" + (f"; {note}" if note else "")))
                else: summary.append({"model": m, "bench": b, "method": "ours", "status": "skipped", "reason": "returned the seed and no non-seed front node"})
            else:
                summary.append({"model": m, "bench": b, "method": "ours", "status": "skipped", "reason": "no zo-result.json"})
            for meth, run, vid, note in cand:
                if sel and (m, b, meth) not in sel: continue
                if vid[:12] == seed[:12] or not has(run, vid) or not has(run, seed):
                    summary.append({"model": m, "bench": b, "method": meth, "status": "skipped", "reason": "final skill is the seed" if vid[:12] == seed[:12] else "no test evaluation of the final skill or the in-run seed"})
                    continue
                r1, r0 = MT.evaluation(run, vid), MT.evaluation(run, seed)
                ok = sorted({k for k, r in r1.items() if k in r0 and None not in (r["c"], r["success"], r0[k]["c"], r0[k]["success"])})
                if not ok:
                    summary.append({"model": m, "bench": b, "method": meth, "status": "skipped", "reason": "no task with known outcome and workload on both sides"}); continue
                e1 = {e.task: e for e in reversed(M.load(path(run, vid)))}
                e0 = {e.task: e for e in reversed(M.load(path(run, seed)))}
                ok = [k for k in ok if k[0] in e1 and k[0] in e0 and e1[k[0]].success is not None and e0[k[0]].success is not None]
                ell = {k: math.log(r1[k]["c"] / r0[k]["c"]) for k in ok}
                med = float(np.median(list(ell.values())))
                tk = min(ok, key=lambda k: (abs(ell[k] - med), k))
                picked[(m, b, meth)] = tk[0]
                ids, base, cand_ = LN.paired([e for e in M.load(path(run, vid)) if e.task in {k[0] for k in ok}], [e for e in M.load(path(run, seed)) if e.task in {k[0] for k in ok}])
                s = D.split(base, cand_)
                gd = {g: 0.0 for g in GROUPS}; fine = {}
                for g, v in s["delta"].items(): gd[group(g)] += float(v); fine[g] = float(v)
                od = outdir(m, meth, b); od.mkdir(parents=True, exist_ok=True)
                tid = tk[0]
                (od / f"episode-{safe(tid)}.json").write_text(json.dumps({
                    "model": m, "bench": b, "method": meth, "run": str(run.relative_to(ROOT)), "skill_version": vid, "seed_version": seed, "note": note,
                    "task": tid, "selection": {"rule": "per-task log(c_skill/c_seed) closest to the cell median", "n_pairs": len(ok), "median_log_ratio": med, "task_log_ratio": ell[tk],
                                               "task_ratio": math.exp(ell[tk]), "median_ratio": math.exp(med)},
                    "seed": episode_json(e0[tid]), "skill": episode_json(e1[tid])}, indent=1))
                (od / "components.json").write_text(json.dumps({
                    "model": m, "bench": b, "method": meth, "run": str(run.relative_to(ROOT)), "note": note, "n": len(ids), "Lambda": s["Lambda"], "rho": math.exp(s["Lambda"]),
                    "groups_order": GROUPS, "groups": gd, "fine": fine, "se_fine": {g: float(v) for g, v in s["se"].items()}}, indent=1))
                summary.append({"model": m, "bench": b, "method": meth, "status": "done", "task": tid, "n": len(ids), "rho": math.exp(s["Lambda"]), "note": note})
                print(m, b, meth, tid, f"rho {math.exp(s['Lambda']):.3f}", flush=True)
    # native-seed: seed episode of the task used by the other methods of the cell
    for m in MODELS:
        for b in BENCHES:
            if sel and (m, b, "native-seed") not in sel: continue
            run = current_run("seed", m, b)
            if not run or not (run / "summary.json").is_file() or not has(run, seed):
                summary.append({"model": m, "bench": b, "method": "native-seed", "status": "skipped", "reason": "no finished native-seed run / seed test evaluation"}); continue
            rows = MT.evaluation(run, seed)
            eps = {e.task: e for e in reversed(M.load(path(run, seed)))}
            ok = sorted(k[0] for k, r in rows.items() if r["c"] is not None and r["success"] is not None and k[0] in eps and eps[k[0]].success is not None)
            used = [picked[(m, b, x)] for x in METHODS[1:] if (m, b, x) in picked and picked[(m, b, x)] in ok]
            if used: tid, why = Counter(used).most_common(1)[0][0], "task used by the other methods of the cell"
            elif ok:
                med = float(np.median([rows[(t, 0)]["c"] for t in ok if (t, 0) in rows] or [rows[k]["c"] for k in rows if k[0] in ok])); tid = min(ok, key=lambda t: abs(min(r["c"] for k, r in rows.items() if k[0] == t) - med)); why = "median-c task (no other method in the cell)"
            else:
                summary.append({"model": m, "bench": b, "method": "native-seed", "status": "skipped", "reason": "no seed episode with known outcome and complete capture"}); continue
            od = outdir(m, "native-seed", b); od.mkdir(parents=True, exist_ok=True)
            (od / f"episode-{safe(tid)}.json").write_text(json.dumps({"model": m, "bench": b, "method": "native-seed", "run": str(run.relative_to(ROOT)), "seed_version": seed,
                                                                      "task": tid, "selection": {"rule": why}, "seed": episode_json(eps[tid])}, indent=1))
            summary.append({"model": m, "bench": b, "method": "native-seed", "status": "done", "task": tid, "note": why})
            print(m, b, "native-seed", tid, why, flush=True)
    out = ATTR / "bipartite-summary.json"
    old = json.loads(out.read_text()) if out.is_file() and sel else []
    keep = [r for r in old if (r["model"], r["bench"], r["method"]) not in {(s["model"], s["bench"], s["method"]) for s in summary}]
    out.write_text(json.dumps(keep + summary, indent=1))


# ---------------------------------------------------------------- plotting
COMP = {GROUPS[0]: "#0072B2", GROUPS[1]: "#CC79A7", GROUPS[2]: "#F0E442", GROUPS[3]: "#FFFFFF", GROUPS[4]: "#BBBBBB"}
MCOL = {"native-seed": "#000000", "skillopt": "#E69F00", "gepa": "#56B4E9", "skilladam": "#009E73", "ours": "#D55E00"}
MNAME = {"native-seed": "Seed", "skillopt": "SkillOpt", "gepa": "GEPA", "skilladam": "SkillAdam", "ours": "BACH"}
MODEL_LABEL = {"deepseek-flash": "Flash", "gpt-6-luna": "Luna"}


def rows_for(docs, ntool=4):
    """Component rows shared by both panels: skill, fixed, history, the heaviest tool components, merged remainder per tool group."""
    w = Counter()
    for d in docs:
        for g, v in d["components"].items(): w[g] += v["c_g"]
    tools = [g for g, _ in w.most_common() if g.startswith("tool:")]
    keep = tools[:ntool]
    fixed = [g for g in w if group(g) == GROUPS[4]]
    rows = [("skill", ["skill"], GROUPS[3]), ("fixed", fixed, GROUPS[4]), ("history", ["history"], GROUPS[2])]
    rows += [(g[5:], [g], group(g)) for g in keep]
    for gr in (GROUPS[0], GROUPS[1]):
        rest = [g for g in tools[ntool:] if group(g) == gr]
        if rest: rows.append(("other " + ("reads" if gr == GROUPS[0] else "tools"), rest, gr))
    return rows


def draw_episode(ax, d, rows, wscale):
    """Two rows: requests (top, time order) and components (bottom)."""
    import matplotlib.patches as P
    R, ny = d["requests"], len(rows)
    xs = [0.03 + 0.94 * (r + 0.5) / R for r in range(R)]
    cx = [0.07 + 0.86 * (i + 0.5) / ny for i in range(ny)]
    ytop, ybot = 0.80, 0.20
    idx = {g: i for i, (_, gs, _) in enumerate(rows) for g in gs}
    agg, inv = {}, {}
    for e in d["exposure_edges"]:
        if e["component"] in idx: agg[(e["request"], idx[e["component"]])] = agg.get((e["request"], idx[e["component"]]), 0) + e["tokens"]
    for e in d["invocation_edges"]:
        if e["component"] in idx: inv[(e["request"], idx[e["component"]])] = inv.get((e["request"], idx[e["component"]]), 0) + e["calls"]
    for (r, i), t in sorted(agg.items(), key=lambda kv: -kv[1]):
        sk = rows[i][2] == GROUPS[3]
        ax.plot([xs[r], cx[i]], [ytop, ybot], color="#D55E00" if sk else COMP[rows[i][2]], lw=max(0.12, wscale * t), alpha=0.5, solid_capstyle="butt", zorder=2)
    for (r, i), n in inv.items():
        ax.plot([xs[r], cx[i]], [ytop, ybot], color="black", lw=0.5, ls=(0, (1.5, 1.2)), zorder=4)
    for r in range(R):
        ax.add_patch(P.Rectangle((xs[r] - 0.4 * 0.94 / R, ytop), 0.8 * 0.94 / R, 0.05, fc="#444444", ec="none", zorder=5))
    for i, (name, _, gr) in enumerate(rows):
        ax.add_patch(P.Rectangle((cx[i] - 0.03, ybot - 0.05), 0.06, 0.05, fc=COMP[gr], ec="black", lw=0.4, hatch="////" if gr == GROUPS[3] else None, zorder=6))
        ax.text(cx[i], ybot - 0.07 - (0.05 if (i % 2 and len(cx) > 6) else 0), name, fontsize=5.5, va="top", ha="center", zorder=7)


def plot_episode(f):
    import matplotlib.pyplot as plt
    d = json.loads(f.read_text())
    sides = [("seed", "Seed", "#000000")] + ([("skill", MNAME[d["method"]] + " skill", MCOL[d["method"]])] if "skill" in d else [])
    rows = rows_for([d[s] for s, _, _ in sides])
    wmax = max(e["tokens"] for s, _, _ in sides for e in d[s]["exposure_edges"])
    wscale = 2.2 / wmax
    plt.rcParams.update({"font.size": 6.5, "pdf.fonttype": 42, "font.family": "serif"})
    fig, axes = plt.subplots(1, len(sides), figsize=(6.8 if len(sides) == 2 else 3.4, 3.1), squeeze=False)
    for ax, (s, title, colr) in zip(axes[0], sides):
        ep = d[s]
        draw_episode(ax, ep, rows, wscale)
        ok = {True: "pass", False: "fail", None: "unknown"}[ep["success"]]
        ax.text(0.5, 0.97, f"{title}: {ep['requests']} requests, {ep['calls']} calls, {ep['input_tokens'] / 1000:.0f}k input tokens, {ok}", ha="center", va="top", fontsize=6.5, color=colr, fontweight="bold")
        ax.text(0.5, 0.90, f"workload c = {ep['c']:.1f}", ha="center", va="top", fontsize=6)
        ax.text(0.03, 0.865, "requests, time order", fontsize=5.5, ha="left", va="bottom", color="#555555")
        ax.set_xlim(0, 1); ax.set_ylim(0.0, 1.0); ax.axis("off")
    sel = d["selection"]
    tid = d["task"] if len(d["task"]) <= 48 else d["task"][:45] + "..."
    sub = f"{MODEL_LABEL[d['model']]}, {d['bench']}, task {tid}"
    if "task_ratio" in sel: sub += f"  \n(c ratio {sel['task_ratio']:.2f}, cell median {sel['median_ratio']:.2f}, n={sel['n_pairs']})"
    fig.suptitle(sub, fontsize=6.5, y=0.99)
    fig.text(0.5, 0.005, ("dotted black: tool call issued by a request; coloured: component tokens inside a request (width ~ tokens, same scale in both panels)" if len(sides) == 2 else "dotted black: tool call issued by a request;\ncoloured: component tokens in a request (width ~ tokens)"), ha="center", fontsize=5.5, color="#444444")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.04, wspace=0.04)
    fig.savefig(f.with_suffix(".pdf")); fig.savefig(f.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_components(f):
    import matplotlib.pyplot as plt
    d = json.loads(f.read_text())
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    G = d["groups_order"]
    vals = [d["groups"][g] for g in G]
    ax.barh(range(len(G)), vals, color=[COMP[g] for g in G], edgecolor="black", linewidth=0.4, height=0.7, hatch=None)
    for i, g in enumerate(G):
        if g == GROUPS[3]: ax.patches[i].set_hatch("////")
        ax.text(vals[i] + (0.004 if vals[i] >= 0 else -0.004), i, f"{vals[i]:+.3f}", va="center", ha="left" if vals[i] >= 0 else "right", fontsize=6)
    ax.set_yticks(range(len(G))); ax.set_yticklabels(G); ax.set_ylim(len(G) - 0.4, -0.6)
    ax.axvline(0, color="black", lw=0.6); ax.axvline(d["Lambda"], color=MCOL[d["method"]], lw=1.2, ls="--")
    lo, hi = min(vals + [0]), max(vals + [0]); pad = 0.22 * (hi - lo or 0.1)
    ax.set_xlim(lo - 1.6 * pad, hi + pad)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel(r"$\delta_g$: change in mean log workload vs. seed")
    ax.set_title(f"{MODEL_LABEL[d['model']]}, {d['bench']}, {MNAME[d['method']]}: " + r"$\Lambda$" + f"={d['Lambda']:+.3f}, " + r"$\rho$" + f"={d['rho']:.2f}, n={d['n']}", fontsize=6.5, loc="left")
    fig.savefig(f.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(f.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    for m in MODELS:
        for meth in METHODS:
            for b in BENCHES:
                od = outdir(m, meth, b)
                if not od.is_dir(): continue
                for f in sorted(od.glob("episode-*.json")): plot_episode(f); print("plotted", f)
                if (od / "components.json").is_file(): plot_components(od / "components.json")
                lines = [f"# {MODEL_LABEL[m]} / {MNAME[meth]} / {b}: bipartite graphs", "", "Generated by baselines/scripts/analysis/bipartite_episodes.py (compute, then plot). Each graph has .pdf, .png and .json (data).", ""]
                for f in sorted(od.glob("episode-*.json")):
                    lines.append(f"- `{f.stem}`: request x component graph of the in-run seed episode" + ("" if meth == "native-seed" else " and the final skill's episode") + " on the representative test task (log workload ratio closest to the cell median); json holds both edge lists.")
                if (od / "components.json").is_file(): lines.append("- `components`: split of the skill's test log-workload change vs. the seed into the five component groups (lmdi, pooled over tasks).")
                (od / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "compute":
        a = sys.argv[2:]
        compute({tuple(a[i:i + 3]) for i in range(0, len(a), 3)} or None)
    else:
        plot()
