"""Shared paths and helpers of the paper-1 analyses: the cache directory, the figure writer and the audit-record writer.
Nothing here touches paperout.record() or any existing paper/assets file; every output is named mech_<name>."""
import json
from pathlib import Path

H = Path("/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation")
CACHE = H / "assets/cache/analysis/mech"
FIG = H / "paper/assets/fig/mech"
DATA = H / "paper/assets/data/mech"
# paper/assets was regrouped on 2026-10-12 (fig/mech, fig/method, data/mech, data/method); the existing BACH-paper files are looked up in either layout
OLD_FIG_DIRS = [FIG, FIG.parent]
OLD_DATA_DIRS = [DATA, DATA.parent]
BASELINES = {"GEPA": "gepa", "SkillOpt": "skillopt", "SkillAdam": "skilladam"}


def save_figure(fig, name):
    """Write paper/assets/fig/mech/mech_<name>.pdf and .svg (svg.fonttype none, text stays editable) and a png preview into the mech cache."""
    import matplotlib
    FIG.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    # TrueType (Type 42) fonts in pdf/ps so the PDFs embed without font-type warnings
    with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(FIG / f"mech_{name}.svg")
        fig.savefig(FIG / f"mech_{name}.pdf")
    fig.savefig(CACHE / f"mech_{name}.png", dpi=200)


def write_json(name, obj):
    """Write assets/cache/analysis/mech/<name>.json."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1))
    return path


def write_record(name, what, runs, aggregation, table, commands, extra=""):
    """Write the audit record paper/assets/data/mech/mech_<name>.md: what it shows, runs, aggregation, the numbers, regenerate commands."""
    DATA.mkdir(parents=True, exist_ok=True)
    text = [f"# mech_{name}", "",
            "Audit record of the paper-1 (mechanism paper) analysis; baseline runs only, no BACH. Written by the script named below.", "",
            "## What it shows", "", what, "",
            "## Runs", "", runs, "",
            "## Aggregation", "", aggregation, "",
            "## Numbers", "", table, "",
            extra, "",
            "## Regenerate (from /mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation/bipa, python = /mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation/bach-harness/.venv/bin/python)", "",
            "```", commands, "```", ""]
    (DATA / f"mech_{name}.md").write_text("\n".join(text))


def md_table(head, rows):
    """GitHub markdown table."""
    lines = ["| " + " | ".join(str(h) for h in head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def find_existing(directories, filename):
    """First existing path of filename among the candidate directories (the assets layout moved to mech/ subfolders)."""
    for directory in directories:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(filename)
