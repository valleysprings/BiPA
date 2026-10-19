"""Audit-record output for the analyses: paper/assets/data/mech record writer (write_record, md_table, find_existing), the Report line collector, and paperout (see paperout.py)."""
from pathlib import Path

from src.utils.dataloader import H, out_dir

DATA = H / "paper/assets/data/mech"
# paper/assets was regrouped on 2026-10-12 (fig/mech, fig/method, data/mech, data/method); the existing BACH-paper files are looked up in either layout
OLD_DATA_DIRS = [DATA, DATA.parent]


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


class Report:
    """print and keep the lines of an analysis; save() writes them to assets/cache/analysis/<module>/<name>.txt."""
    def __init__(self): self.lines = []
    def __call__(self, *a):
        s = " ".join(str(x) for x in a); self.lines.append(s); print(s)
    def save(self, module, name):
        (out_dir(module)/name).write_text("\n".join(self.lines) + "\n")
