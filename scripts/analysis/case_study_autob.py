"""Case study on AutoB (partial-credit scores): per task cluster (id prefix) success, mean soft score s and geometric-mean cost ratio against the seed
for the seed, SkillOpt, GEPA and every evaluated BACH front point on the TEST panel (repetition 0), plus the ZO lineage on the selection set.
Soft score = bach.workload.soft; cost c = main_table.task_row; ratio paired per task with the seed evaluation of the same run.
Writes assets/artifacts/attribution/<model>/ours/case-study-autob/{clusters.json,clusters.md,lineage.md,clusters-quality.png,clusters-cost.png}.
Two stages (the slim venv has no matplotlib, the system python has no harness_bench): `bach-harness/.venv/bin/python case_study_autob.py` computes tables,
clusters.json and lineage.md; `python3 case_study_autob.py plot` draws the figures from clusters.json."""
import json, math, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "baselines"), str(ROOT / "baselines/scripts/analysis"), str(ROOT / "bach-harness/src")]
PLOT = sys.argv[1:] == ["plot"]
if PLOT:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
else:
    from bach.workload import soft
from bipa.layout import SKILLS, ARTIFACTS, BASELINE_ROOT, run_path, current_run, relocate
if not PLOT:
    from main_table import evaluation, version

BENCH, MODELS = "automationbench", ("gpt-6-luna", "deepseek-flash")
OURS = json.loads((ROOT / "bach-harness/config/ours_runs.json").read_text())
SEED_VID = None if PLOT else version((SKILLS / "seed/SKILL.md").read_text())
COLORS = ["#000000", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]


def rows(run, vid):
    """{task: (success, soft, cost)} of the newest test evaluation of this skill version in `run`."""
    base = evaluation(run, vid)
    path = max((f for f in (run / "results/evaluations/test").glob("*/evaluation.json") if f.parent.name.startswith(vid[:12])), key=lambda f: f.stat().st_mtime)
    raw = {e["task_id"]: e for e in json.loads(relocate(path).read_text())["episodes"]}
    return {k[0]: (r["success"], None if r["success"] is None else soft(raw[k[0]], r["success"], BENCH), r["c"]) for k, r in base.items()}


def cluster(rs, seed):
    """per cluster and 'all': n, success rate, mean soft, geometric-mean cost ratio against the seed (paired, n_c)."""
    g = defaultdict(list)
    for t, r in rs.items():
        g[t.split("-")[0]].append(t); g["all"].append(t)
    out = {}
    for k, ts in sorted(g.items()):
        known = [t for t in ts if rs[t][0] is not None]
        lr = [math.log(rs[t][2] / seed[t][2]) for t in ts if rs[t][2] and seed[t][2]]
        out[k] = {"n": len(ts), "Q": sum(rs[t][0] is True for t in known) / len(known), "S": sum(rs[t][1] for t in known) / len(known),
                  "rho": math.exp(sum(lr) / len(lr)), "n_rho": len(lr)}
    return out


def table(skills, key, fmt):
    cl = sorted({c for v in skills.values() for c in v})
    cl = [c for c in cl if c != "all"] + ["all"]
    head = "| skill | " + " | ".join(f"{c} (n={next(iter(skills.values()))[c]['n']})" for c in cl) + " |\n|---|" + "---|" * len(cl)
    return head + "\n" + "\n".join(f"| {s} | " + " | ".join(fmt(v[c]) for c in cl) + " |" for s, v in skills.items())


def figures(model, skills, out):
    cl = [c for c in next(iter(skills.values())) if c != "all"] + ["all"]
    w = 0.8 / len(skills)
    for name, panels in (("quality", (("Q", "Success rate"), ("S", "Mean soft score"))), ("cost", (("rho", "Cost ratio to seed (geo. mean)"),))):
        fig, axs = plt.subplots(len(panels), 1, figsize=(8.5, 2.8 * len(panels)), sharex=True, squeeze=False)
        for ax, (key, label) in zip(axs[:, 0], panels):
            for i, (s, v) in enumerate(skills.items()):
                ax.bar([j + (i - (len(skills) - 1) / 2) * w for j in range(len(cl))], [v[c][key] for c in cl], w, label=s, color=COLORS[i % len(COLORS)])
            ax.set_ylabel(label); ax.spines[["top", "right"]].set_visible(False)
            if key == "rho": ax.axhline(1, color="gray", lw=0.8, ls="--")
        axs[-1, 0].set_xticks(range(len(cl)), [f"{c}\n(n={skills[next(iter(skills))][c]['n']})" for c in cl])
        axs[0, 0].legend(ncol=3, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"AutoB test, {model}", y=1.0 if name == "cost" else 1.02, fontsize=9)
        fig.tight_layout(); fig.savefig(out / f"clusters-{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def lineage(model, res, out):
    nodes, md = {n["id"]: n for n in res["nodes"]}, [f"# Lineage of {model} on AutoB (selection set)", "",
        "Per node vs its parent on the tasks both evaluated on sel: dQ = success-rate change, dS = soft change, rho = geometric-mean cost ratio. "
        "Edit delta = bipartite decomposition of the log-cost change by component (top 3 by size; total = their sum over all components). "
        "Nodes with no sel were rejected at the 15-task screen (screen dQ/dS shown).", ""]
    dec = {r["child"]: r["decision"] for r in res["rounds"]}
    for n in res["nodes"]:
        d = dec.get(n["id"], "seed")
        md += [f"## {n['id']} ({n['kind']}, round {n['round']}, parent {n['parent']}): {d}" + (" [returned]" if n["id"] == res["returned"] else ""), ""]
        e = n["edit"]
        if e:
            top = sorted(e["delta"].items(), key=lambda kv: -abs(kv[1]))[:3]
            md.append(f"- edit: op {e['op']}, screen n={e['n']} dQ {e['dQ']:+.3f} dS {e['dS']:+.3f} pass {n['screen']['pass']}; log-cost delta total {sum(e['delta'].values()):+.3f}; "
                      + ", ".join(f"`{k}` {v:+.3f}" for k, v in top))
        if n["sel"]:
            p = nodes[n["parent"]]["sel"]["per_task"] if n["parent"] else None
            pt = n["sel"]["per_task"]
            md.append(f"- sel: n={n['sel']['n']} Q {n['sel']['Q']:.3f} S {n['sel']['S']:.3f} L {n['sel']['L']:+.3f}")
            if p:
                g = defaultdict(list)
                for t in pt:
                    if t in p and pt[t][0] is not None and p[t][0] is not None: g[t.split("-")[0]].append(t)
                md += ["", "| cluster | n | dQ | dS | rho |", "|---|---|---|---|---|"]
                for c, ts in sorted(g.items()):
                    lr = [math.log(pt[t][2] / p[t][2]) for t in ts if pt[t][2] and p[t][2]]
                    md.append(f"| {c} | {len(ts)} | {sum(pt[t][0] - p[t][0] for t in ts) / len(ts):+.2f} | {sum(pt[t][1] - p[t][1] for t in ts) / len(ts):+.2f} | "
                              + (f"{math.exp(sum(lr) / len(lr)):.2f}" if lr else "-") + " |")
        md.append("")
    (out / "lineage.md").write_text("\n".join(md))


def main():
    for model in MODELS:
        out = ARTIFACTS / "attribution" / model / "ours" / "case-study-autob"; out.mkdir(parents=True, exist_ok=True)
        if PLOT:
            figures(model, json.loads((out / "clusters.json").read_text()), out)
            continue
        run = run_path("zo", model, BENCH, OURS[model][BENCH])
        res = json.loads((run / "zo-result.json").read_text())
        nodes = {n["id"]: n for n in res["nodes"]}
        seed = rows(run, SEED_VID)
        skills = {"seed": cluster(seed, seed)}
        for meth in ("skillopt", "gepa"):
            r = current_run(meth, model, BENCH)
            skills[meth] = cluster(rows(r, version(json.loads((r / "summary.json").read_text())["final_skill"])), rows(r, SEED_VID))
        for i in res["front"]:
            skills[f"BACH {i}" + (" (returned)" if i == res["returned"] else "")] = cluster(rows(run, version(nodes[i]["text"])), seed)
        (out / "clusters.json").write_text(json.dumps(skills, indent=1))
        sec = lambda t, k, f: f"### {t}\n\n" + table(skills, k, f) + "\n"
        (out / "clusters.md").write_text(f"# AutoB case study, {model} (TEST panel, 120 tasks, repetition 0)\n\nCluster = task-id prefix. Soft = AutoB passed/non-excluded assertions. "
            "rho = geometric-mean per-task cost ratio (c = N + X/1000) against the seed evaluation of the same run, over tasks with a cost on both sides.\n\n"
            + sec("Success rate", "Q", lambda v: f"{v['Q']:.2f}") + "\n" + sec("Mean soft score", "S", lambda v: f"{v['S']:.3f}") + "\n"
            + sec("Cost ratio to seed (paired n)", "rho", lambda v: f"{v['rho']:.2f} ({v['n_rho']})"))
        lineage(model, res, out)


main()
