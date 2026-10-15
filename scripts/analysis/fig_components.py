"""Figure 'components': where the extra workload goes. Stage 1 (slim venv python): fig_components.py compute -> figures/components.json;
stage 2 (system python3, matplotlib): fig_components.py plot -> figures/components.{pdf,png}.
Each non-seed final skill (SkillOpt, GEPA, SkillAdam, BACH front points) vs the seed's test episodes of the same run, paired per task:
D.split gives per-component deltas that add up to Lambda = mean log(c/c_seed); exp(Lambda) is checked against the table's rho."""
import glob, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = ROOT / "assets/artifacts/results"
OUT = RES / "figures/components"
CELLS = [(m, b) for b in ("automationbench", "tau3-bench") for m in ("deepseek-flash", "gpt-6-luna")]
GROUPS = ["Reads & retrieval", "Bash & other tools", "History", "Skill text", "Fixed context"]
READ = ("tool:read.", "tool:KB_search", "tool:lookup", "tool:discover", "tool:grep", "tool:glob", "tool:bash.read")


def group(g):
    if g.startswith(READ): return GROUPS[0]
    if g.startswith("tool:"): return GROUPS[1]
    return {"history": GROUPS[2], "skill": GROUPS[3]}.get(g, GROUPS[4])


def compute():
    sys.path[:0] = [str(ROOT / "bach-harness/src"), str(HERE), str(HERE.parents[1])]
    import csv, re
    import main_table as MT
    from bach import decompose as D, lineage as LN, workload as M
    from src.layout import SKILLS, current_run
    seed = MT.version((SKILLS / "seed/SKILL.md").read_text())
    ours = json.loads((ROOT / "bach-harness/config/ours_runs.json").read_text())
    table = {(r["model"], r["bench"], r["method"]): float(r["rho"]) for r in csv.DictReader(open(RES / "main-table.csv")) if r["status"] == "done" and r["rho"]}
    front = {}
    for line in open(RES / "ours-front.md"):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 8 and c[0] in ("deepseek-flash", "gpt-6-luna"): front[(c[0], c[1], c[2])] = float(re.match(r"\s*([\d.]+)", c[7]).group(1))
    path = lambda run, vid: max(glob.glob(f"{run}/results/evaluations/test/{vid[:12]}*/evaluation.json"), key=lambda f: Path(f).stat().st_mtime)
    known = lambda eps: [e for e in eps if e.success is not None]
    rows = []
    for m, b in CELLS:
        runs = {"skillopt": current_run("skillopt", m, b), "gepa": current_run("gepa", m, b), "skilladam": current_run("skilladam", m, b)}
        cand = []  # (label, method, run, version, table rho)
        for k, run in runs.items():
            if run and (run / "summary.json").is_file():
                cand.append((k, k, run, MT.version(json.loads((run / "summary.json").read_text())["final_skill"]), table.get((m, b, k))))
        zo = current_run("zo", m, b)
        d = json.loads((zo / "zo-result.json").read_text())
        for n in d["nodes"]:
            if n["id"] in d["front"] and n["parent"] is not None:
                cand.append((f"bach-{n['id']}", "bach", zo, n["version"], front.get((m, b, n["id"]))))
        for label, method, run, vid, ref in cand:
            if not glob.glob(f"{run}/results/evaluations/test/{vid[:12]}*/evaluation.json") or vid == seed: continue
            r1, r0 = MT.evaluation(run, vid), MT.evaluation(run, seed)  # the table's own valid pairs: complete capture, known outcome on both sides
            ok = {k[0] for k, r in r1.items() if k in r0 and None not in (r["c"], r["success"], r0[k]["c"], r0[k]["success"])}
            ref = MT.pair(r1, r0)["rho"] if ref is None else ref
            ids, b0, b1 = LN.paired([e for e in M.load(path(run, vid)) if e.task in ok], [e for e in M.load(path(run, seed)) if e.task in ok])
            s = D.split(b0, b1)
            gd = {g: 0.0 for g in GROUPS}
            fine = {}
            for g, v in s["delta"].items(): gd[group(g)] += float(v); fine[g] = float(v)
            rows.append({"model": m, "bench": b, "method": method, "label": label, "n": len(ids), "Lambda": s["Lambda"], "rho": float(__import__("math").exp(s["Lambda"])),
                         "table_rho": ref, "groups": gd, "fine": fine})
            print(m, b, label, len(ids), f"rho {rows[-1]['rho']:.3f} table {ref}", {k: round(v, 3) for k, v in gd.items()}, flush=True)
    OUT.with_suffix(".json").write_text(json.dumps({"groups": GROUPS, "rows": rows}, indent=1))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    doc = json.loads(OUT.with_suffix(".json").read_text())
    col = dict(zip(GROUPS, ["#0072B2", "#CC79A7", "#E69F00", "#D55E00", "#BBBBBB"]))
    col = {GROUPS[0]: "#0072B2", GROUPS[1]: "#CC79A7", GROUPS[2]: "#F0E442", GROUPS[3]: "#FFFFFF", GROUPS[4]: "#BBBBBB"}
    mc = {"skillopt": "#E69F00", "gepa": "#56B4E9", "skilladam": "#009E73", "bach": "#D55E00"}
    name = {"skillopt": "SkillOpt", "gepa": "GEPA", "skilladam": "SkillAdam", "bach": "BACH"}
    ml = {"deepseek-flash": "Flash", "gpt-6-luna": "Luna"}
    bl = {"automationbench": "AutoBench", "tau3-bench": r"$\tau^3$"}
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "serif"})
    cells = [(m, b) for b in ("automationbench", "tau3-bench") for m in ("deepseek-flash", "gpt-6-luna")]
    heights = [max(1, len([r for r in doc["rows"] if (r["model"], r["bench"]) == c])) for c in cells]
    fig, axes = plt.subplots(len(cells), 1, figsize=(6.8, 0.2 * sum(heights) + 0.45 * len(cells) + 0.6), gridspec_kw={"height_ratios": heights, "hspace": 0.9}, sharex=True)
    for ax, (m, b) in zip(axes, cells):
        R = sorted([r for r in doc["rows"] if (r["model"], r["bench"]) == (m, b)], key=lambda r: -r["Lambda"])
        for y, r in enumerate(R):
            pos = neg = 0
            for g in GROUPS:
                v = r["groups"][g]
                left = pos if v >= 0 else neg
                ax.barh(y, v, left=left, color=col[g], edgecolor="black", linewidth=0.3, height=0.72, hatch="////" if g == GROUPS[3] else None)
                if v >= 0: pos += v
                else: neg += v
            ax.plot([r["Lambda"]], [y], marker="|", color="black", ms=8, mew=1.2)
            ax.text(max(pos, 0) + 0.01, y, f"{r['rho']:.2f}", va="center", fontsize=6)
        ax.set_yticks(range(len(R)))
        ax.set_yticklabels([name[r["method"]] + (" " + r["label"][-2:] if r["method"] == "bach" else "") for r in R])
        for t, r in zip(ax.get_yticklabels(), R): t.set_color(mc[r["method"]])
        ax.axvline(0, color="black", lw=0.6); ax.set_ylim(len(R) - 0.4, -0.6)
        ax.set_title(f"{ml[m]}, {bl[b]}", loc="left", fontsize=7, pad=2)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[-1].set_xlabel(r"change in mean log workload vs. seed, split by component ($\Lambda=\ln\rho$; number = $\rho$, bar tick = $\Lambda$)")
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=col[g], ec="black", lw=0.3, hatch="////" if g == GROUPS[3] else None) for g in GROUPS], labels=GROUPS, ncol=5, fontsize=6, frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, 1.25), columnspacing=1.0, handlelength=1.2)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    {"compute": compute, "plot": plot}[sys.argv[1]]()
