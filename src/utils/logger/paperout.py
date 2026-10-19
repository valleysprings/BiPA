"""Where paper material goes.
Two papers: "mech" (paper 1, paper/draft2mech: measurement + mechanism) and "method" (paper 2, paper/draft2method: BACH). Every spec has pub = "mech" | "method".
Figures -> paper/assets/fig/<pub>/<name>.pdf + editable <name>.svg (text kept as text): savefig(fig, name) looks the pub up in SPECS.
Numbers -> NO data files: paper/assets/data/<pub>/<name>.md, one audit record per draft table / data figure, WRITTEN BY THE SCRIPTS through record(): paper location, producing script and commands,
dates (UTC) of the underlying runs, the numbers as the paper prints them, raw records (absolute paths), caveats. A record can be filled by several scripts (parts); the parts are kept in
assets/cache/analysis/_records/<name>.json and the md is re-rendered from all of them, so a rerun of any script refreshes its part. Intermediates and caches: common.out_dir(module) = assets/cache/analysis/<module>/.
Stdlib only (matplotlib imported on use)."""
import json, re
from datetime import datetime, timezone
from pathlib import Path

H = Path("/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation")
ASSETS = H / "paper/assets"
FIG, DATA = ASSETS / "fig", ASSETS / "data"
DRAFTS = {"mech": H / "paper/draft2mech", "method": H / "paper/draft2method"}
PUBS = tuple(DRAFTS)
SIDE = H / "assets/cache/analysis/_records"
REPO = H / "bipa"

BASELINE_NOTE = "Baseline rows (GEPA, SkillOpt, SkillAdam) come from the https reruns `native-20261008-https` (2026-10-08), each evaluated avg@3 on the test split."
BACH_NOTE = "BACH numbers are from bach-graph-3 (2026-10-10)."

# name -> what the record is about. paper: (label, draft file relative to paper/draft2method, regex of the line to cite, what it shows); files: fig / editable source; notes: static caveats.
SPECS = {
    "sec01_teaser": dict(pub="method", paper=[("fig:teaser (Figure 1)", "sections/01-intro.tex", r"label\{fig:teaser\}", "one AutoBench test task solved by the seed skill, SkillOpt's skill and BACH's skill, each drawn as its real episode graph (turns, tool calls, exposure of tool outputs in later turns) with its input-token split")],
                         files="fig/method/sec01_teaser.pdf, fig/method/sec01_teaser.svg", notes=[BASELINE_NOTE, BACH_NOTE]),
    "sec05_table1": dict(pub="method", paper=[("tab:main (Table 1)", "sections/05-experiments.tex", r"label\{tab:main\}", "test success Q and the ratios of input tokens (rho) and tool calls (rho^calls) to the Seed skill row, per method and benchmark")],
                         files="typed by hand in the tex", notes=[BASELINE_NOTE]),
    "sec05_ablation": dict(pub="method", paper=[("tab:ablation", "sections/05-experiments.tex", r"label\{tab:ablation\}", "component ablations of BACH on tau3 and AutoBench (test Q and rho)")],
                           files="typed by hand in the tex", notes=["Cells without a finished run stay pending (\\tbd in the draft)."]),
    "sec05_evolution": dict(pub="method", paper=[("fig:evolution", "appendix/b-noise.tex", r"label\{fig:evolution\}", "BACH: success (row 1, with the baselines' final skills' test avg@3 success from Table 1 as horizontal reference lines) and input-token ratio rho (row 2) of the incumbent skill after each round against the training tasks consumed, on dev (solid) and test (dashed, where evaluated), per benchmark")],
                            files="fig/method/sec05_evolution.pdf, fig/method/sec05_evolution.svg", notes=["Baseline reference lines are the final skills' test avg@3 from Table 1 (assets/cache/analysis/table1/table1.json).", "Runs without summary.json are in progress; this is noted here only, the plot does not mark it."]),
    "appE_levertest": dict(pub="method", paper=[("fig:appd-rewrite", "appendix/c-analysis.tex", r"label\{fig:appd-rewrite\}", "dev-set prediction of the dominant lever and removable part; measured saving of the economize rewrite on test (per-skill table dropped 2026-10-10; headline numbers in Section 5.4)")],
                           files="typed by hand in the tex", notes=["SkillAdam on TB2 was not run. The caption's seed spreads (0.050, 0.129, 0.037 log input tokens; 0.052, 0.043, 0.054 log tool calls; 4.0, 8.4, 5.6 points) come from `src/analysis/appendix/slopes.py` (`reruns` subcommand, `seed_spread.json`), not from this table's script.",
                                                                    "Draft column mapping: Lever `turns` = `dominant: merge`; Remov. = `predicted_removable`; Saving = `realized_saving`; Turns / Calls = (exp(dlog) - 1) in percent of `turns_dlog` / `actions_dlog`; Words = `words`; Success = parent -> child."]),
}
for _n, _d in (("appC_slopes", "within-task slope of log input tokens / log output tokens / log calls on log turns per benchmark (one row per quantity), with the slope-1 reference"), ("appC_price", "per optimized skill the input-token ratio against the cache-priced cost ratio")):
    SPECS[_n] = dict(pub="mech", paper=[("Appendix C (Additional Analysis)", "appendix/c-analysis.tex", r"label\{app:finding\}", _d)], files=f"fig/mech/{_n}.pdf, fig/mech/{_n}.svg", notes=[BASELINE_NOTE, BACH_NOTE])
SPECS["appC_attribution"] = dict(pub="mech", paper=[("Appendix C (Additional Analysis)", "appendix/c-analysis.tex", r"label\{app:finding\}", "per optimized skill the change in log input tokens split into the four terms and the measured total, drawn as an annotated heatmap grouped by optimizer")],
                                 files="fig/mech/appC_attribution.pdf, fig/mech/appC_attribution.svg",
                                 notes=["Baseline rows come from the https reruns (`native-20261008-https`), pinned in `appc.py` (`ATTR_RUNS`) — the same runs that feed the avg@3 table (`sec05_table1`).",
                                        BACH_NOTE])

# Paper 1 (mech) new records. Location: draft2mech; the sections/appendix file and label are filled in by the agent that writes the tex.
SPECS["mech_table1"] = dict(pub="mech", paper=[("tab:main (Table 1)", "sections/04-setup.tex", r"label\{tab:main\}", "test success Q and the ratios of input tokens (rho) and tool calls to the Seed row for No skill, Seed, GEPA, SkillOpt, SkillAdam on TB2, tau3-bench, AutoBench (avg@3)")],
                            files="typed by hand in the tex", notes=[BASELINE_NOTE])
SPECS["mech_teaser"] = dict(pub="mech", paper=[("fig:teaser (Figure 1)", "sections/01-intro.tex", r"label\{fig:teaser\}", "teaser figure of paper 1")],
                            files="fig/mech/mech_teaser.pdf, fig/mech/mech_teaser.svg", notes=[BASELINE_NOTE])
SPECS["mech_reruns"] = dict(pub="mech", paper=[("fig:reruns", "sections/05-findings.tex", r"label\{fig:reruns\}", "every test run of the seed and of the optimized skills against the seed's run-to-run spread")],
                            files="fig/mech/mech_reruns.pdf, fig/mech/mech_reruns.svg", notes=[BASELINE_NOTE])
SPECS["mech_selection"] = dict(pub="mech", paper=[("fig:selection", "appendix/c-rq1.tex", r"label\{fig:selection\}", "dev-set selection under the null: the gain a success-only selector reports when no candidate helps")],
                               files="fig/mech/mech_selection.pdf, fig/mech/mech_selection.svg", notes=[BASELINE_NOTE])
SPECS["mech_slopes"] = dict(pub="mech", paper=[("fig:slopes", "appendix/d-rq2.tex", r"label\{fig:slopes\}", "within-task slopes of log input tokens, output tokens and calls on log turns")],
                            files="fig/mech/mech_slopes.pdf, fig/mech/mech_slopes.svg", notes=[BASELINE_NOTE])
ORDER = ["sec01_teaser", "sec05_table1", "sec05_ablation", "sec05_evolution", "appE_levertest"]


def pub_of(name):
    """The paper ("mech" or "method") a record / figure belongs to."""
    pub = SPECS[name]["pub"]
    assert pub in PUBS, (name, pub)
    return pub


def savefig(fig, name, **kw):
    """<name>.pdf and <name>.svg (svg.fonttype none: text stays editable) into paper/assets/fig/<pub>/, pub from SPECS[name]["pub"] (register the spec first: SPECS[name] = dict(pub=..., ...))."""
    import matplotlib
    out = FIG / pub_of(name)
    out.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig.savefig(out / f"{name}.svg", **kw)
    fig.savefig(out / f"{name}.pdf", **kw)


def mdtable(head, rows):
    """GitHub markdown table from a header list and rows (lists of cells)."""
    L = ["| " + " | ".join(str(h) for h in head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    L += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(L)


def _utc(ts): return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M")


def _span(ts):
    ts = sorted(ts)
    return "" if not ts else _utc(ts[0]) if _utc(ts[0]) == _utc(ts[-1]) else f"{_utc(ts[0])} .. {_utc(ts[-1])}"


def _when(paths):
    """Markdown table of UTC dates per raw path: a directory (run) gives its launch.log / run.log resume stamps, the first..last evaluation.json written under results/evaluations, and the
    mtime of native_protocol.json / summary.json / result.json / report.json; a file gives its own mtime."""
    rows, allts = [], []
    paths = [Path(p) for p in paths]
    dirs = [p for p in paths if p.is_dir()]
    for p in paths:
        if not p.exists(): rows.append([f"`{p}`", "missing", "", ""]); continue
        if p.is_file():
            allts.append(p.stat().st_mtime)
            if not any(d in p.parents for d in dirs): rows.append([f"`{p}`", "", f"written {_utc(p.stat().st_mtime)}", ""])
            continue
        logs = []
        for n in ("launch.log", "run.log"):
            f = p / n
            if f.is_file():
                logs += [datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() for s in re.findall(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", f.read_text(errors="ignore"))]
        ev = [f.stat().st_mtime for f in p.glob("results/evaluations/*/*/evaluation.json")] + [f.stat().st_mtime for f in p.glob("child_eval.json")]
        listed = [f for f in paths if f.is_file() and p in f.parents]
        meta = [p / n for n in ("native_protocol.json", "summary.json", "result.json", "rewrite.json") if (p / n).is_file()]
        mt = "; ".join(f"{f.name} {_utc(f.stat().st_mtime)}" for f in meta)
        evs = ((f"{len(ev)} files, " + _span(ev)) if ev else "") + ("; listed evaluation " + _span([f.stat().st_mtime for f in listed]) if listed else "")
        rows.append([f"`{p}`", _span(logs) if logs else "", evs, mt])
        allts += logs + ev + [f.stat().st_mtime for f in meta]
    head = ["raw record", "launch.log / run.log stamps", "evaluation.json written", "protocol / summary written"]
    return mdtable(head, rows), allts


def record(name, part, result=None, commands=(), raw=(), notes=(), source=()):
    """Write paper/assets/data/<pub>/<name>.md (pub from SPECS[name]["pub"]). `part` names the piece of the record this call fills (a script may fill one record in several parts, several scripts one record); result is
    markdown (the numbers as the paper prints them), commands the exact shell commands (from bipa/), raw absolute paths of run dirs / evaluation files, notes extra caveats."""
    spec = SPECS[name]
    SIDE.mkdir(parents=True, exist_ok=True); (DATA / pub_of(name)).mkdir(parents=True, exist_ok=True)
    f = SIDE / f"{name}.json"
    parts = json.loads(f.read_text()) if f.is_file() else {}
    parts[part] = dict(result=result, commands=list(commands), raw=sorted({str(p) for p in raw}), notes=list(notes), source=list(source), stamp=_utc(datetime.now(timezone.utc).timestamp()))
    f.write_text(json.dumps(parts, indent=1))
    render(name, parts)


def render(name, parts):
    spec = SPECS[name]
    L = [f"# {name}", "", f"Audit record, written by the producing script(s) below; rerunning them refreshes this file. Do not edit by hand.", ""]
    L += ["## 1. Paper", ""]
    for label, tex, rx, what in spec["paper"]:
        draft = DRAFTS[spec["pub"]]
        f = draft / tex
        lines = f.read_text().splitlines() if f.is_file() else []
        ln = next((i + 1 for i, s in enumerate(lines) if re.search(rx, s)), None)
        L.append(f"- {label}: `{draft / tex}:{ln}`. {what}.")
    L.append(f"- Typeset as: {spec['files']}.")
    L += ["", "## 2. Source", ""]
    srcs = sorted({s for p in parts.values() for s in p["source"]})
    for s in srcs: L.append(f"- `{s}`")
    L += ["", "Regenerate (from `/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation/bipa`):", "", "```"]
    for c in dict.fromkeys(c for p in parts.values() for c in p["commands"]): L.append(c)
    L += ["```", "", "## 3. When", ""]
    raw = sorted({r for p in parts.values() for r in p["raw"]})
    if raw:
        tbl, ts = _when(raw)
        L += [f"Underlying runs / evaluations (UTC): {_span(ts)}.", "", tbl, ""]
    else: L += ["No runs underlie this record.", ""]
    L.append(f"Record generated (UTC): {max(p['stamp'] for p in parts.values())}.")
    L += ["", "## 4. Result", ""]
    for k, p in parts.items():
        if p["result"]: L += [p["result"].rstrip(), ""]
    L += ["## 5. Raw records", ""]
    L += [f"- `{r}`" for r in raw] if raw else ["None."]
    L += ["", "## 6. Notes", ""]
    L += [f"- {n}" for n in dict.fromkeys(list(spec["notes"]) + [n for p in parts.values() for n in p["notes"]])]
    (DATA / spec["pub"] / f"{name}.md").write_text("\n".join(L) + "\n")
