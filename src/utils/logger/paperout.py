"""Audit records filled in parts by one or more scripts: OUT/data/mech/<name>.md, re-rendered from all parts on every write."""
import json, re
from datetime import datetime, timezone
from pathlib import Path

from src import layout
from src.utils.logger import md_table as mdtable

FIG, DATA = layout.OUT / "fig", layout.OUT / "data/mech"
SIDE = layout.ANALYSIS / "_records"

# name -> shows: what the record's table or figure shows, files: the outputs, notes: fixed caveats
SPECS = {
    "mech_table1": dict(shows="test success Q and the ratios of input tokens (rho) and tool calls to the Seed row for No skill, Seed, GEPA, "
                              "SkillOpt, SkillAdam on TB2, tau3-bench, AutoBench (avg@3)", files="the markdown table below (no figure)"),
    "appC_attribution": dict(shows="per optimized skill the change in log input tokens split into the four terms and the measured total, drawn "
                                   "as an annotated heatmap grouped by optimizer",
                             files="fig/mech/appC_attribution.pdf, fig/mech/appC_attribution.svg"),
    "appC_slopes": dict(shows="within-task slope of log input tokens / log output tokens / log calls on log turns per benchmark (one row per "
                              "quantity), with the slope-1 reference", files="fig/mech/appC_slopes.pdf, fig/mech/appC_slopes.svg"),
    "appC_price": dict(shows="per optimized skill the input-token ratio against the cache-priced cost ratio",
                       files="fig/mech/appC_price.pdf, fig/mech/appC_price.svg"),
}


def _utc(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M")


def _span(ts):
    ts = sorted(ts)
    return "" if not ts else _utc(ts[0]) if _utc(ts[0]) == _utc(ts[-1]) else f"{_utc(ts[0])} .. {_utc(ts[-1])}"


def _when(paths):
    """(markdown table of UTC dates per raw path, all timestamps): a run directory gives its log stamps, evaluation
    writes and metadata mtimes; a file gives its own mtime."""
    rows, allts = [], []
    paths = [Path(p) for p in paths]
    dirs = [p for p in paths if p.is_dir()]
    for p in paths:
        if not p.exists():
            rows.append([f"`{p}`", "missing", "", ""])
        elif p.is_file():
            allts.append(p.stat().st_mtime)
            if not any(d in p.parents for d in dirs):
                rows.append([f"`{p}`", "", f"written {_utc(p.stat().st_mtime)}", ""])
        else:
            logs = [datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
                    for n in ("launch.log", "run.log") if (p / n).is_file()
                    for s in re.findall(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", (p / n).read_text(errors="ignore"))]
            ev = [f.stat().st_mtime for f in [*p.glob("results/evaluations/*/*/evaluation.json"), *p.glob("child_eval.json")]]
            listed = [f.stat().st_mtime for f in paths if f.is_file() and p in f.parents]
            meta = [p / n for n in ("native_protocol.json", "summary.json", "result.json", "rewrite.json") if (p / n).is_file()]
            evs = (f"{len(ev)} files, " + _span(ev) if ev else "") + ("; listed evaluation " + _span(listed) if listed else "")
            rows.append([f"`{p}`", _span(logs), evs, "; ".join(f"{f.name} {_utc(f.stat().st_mtime)}" for f in meta)])
            allts += logs + ev + [f.stat().st_mtime for f in meta]
    head = ["raw record", "launch.log / run.log stamps", "evaluation.json written", "protocol / summary written"]
    return mdtable(head, rows), allts


def record(name, part, result=None, commands=(), raw=(), notes=(), source=(), sections=()):
    """Store this script's part of record `name` (markdown result, shell commands, raw run paths, notes, extra
    (title, markdown) sections) and re-render DATA/<name>.md from all parts."""
    SIDE.mkdir(parents=True, exist_ok=True)
    f = SIDE / f"{name}.json"
    parts = json.loads(f.read_text()) if f.is_file() else {}
    parts[part] = dict(result=result, commands=list(commands), raw=sorted({str(p) for p in raw}), notes=list(notes),
                       source=list(source), sections=[list(x) for x in sections], stamp=_utc(datetime.now(timezone.utc).timestamp()))
    f.write_text(json.dumps(parts, indent=1))
    render(name, parts)


def render(name, parts):
    spec = SPECS[name]
    L = [f"# {name}", "", "Audit record, written by the producing script(s) below; rerunning them refreshes this file. "
         "Do not edit by hand.", "", "## 1. What it shows", ""]
    L += [f"- {spec['shows']}.", f"- Output: {spec['files']}.", "", "## 2. Source", ""]
    L += [f"- `{s}`" for s in sorted({s for p in parts.values() for s in p["source"]})]
    L += ["", "Regenerate (from the repository root):", "", "```"]
    L += list(dict.fromkeys(c for p in parts.values() for c in p["commands"]))
    L += ["```", "", "## 3. When", ""]
    raw = sorted({r for p in parts.values() for r in p["raw"]})
    if raw:
        tbl, ts = _when(raw)
        L += [f"Underlying runs / evaluations (UTC): {_span(ts)}.", "", tbl, ""]
    else:
        L += ["No runs underlie this record.", ""]
    L += [f"Record generated (UTC): {max(p['stamp'] for p in parts.values())}.", "", "## 4. Result", ""]
    L += [x for p in parts.values() if p["result"] for x in (p["result"].rstrip(), "")]
    L += ["## 5. Raw records", ""] + ([f"- `{r}`" for r in raw] or ["None."]) + ["", "## 6. Notes", ""]
    L += [f"- {n}" for n in dict.fromkeys([*spec.get("notes", ()), *(n for p in parts.values() for n in p["notes"])])]
    for i, (title, body) in enumerate((x for p in parts.values() for x in p.get("sections", [])), start=7):
        L += ["", f"## {i}. {title}", "", body.rstrip()]
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{name}.md").write_text("\n".join(L) + "\n")
