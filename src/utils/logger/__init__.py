"""Audit records of the analyses (OUT/data/mech) and the Report line collector."""
import importlib, sys

from src import layout
from src.utils.dataloader import out_dir

DATA = layout.OUT / "data/mech"
OLD_DATA_DIRS = [DATA]


def write_record(name, what, runs, aggregation, table, commands, extra=""):
    """Write DATA/mech_<name>.md: what it shows, runs, aggregation, numbers and the commands that regenerate it."""
    DATA.mkdir(parents=True, exist_ok=True)
    text = [f"# mech_{name}", "", "Audit record written by the script named below.", "",
            "## What it shows", "", what, "", "## Runs", "", runs, "", "## Aggregation", "", aggregation, "",
            "## Numbers", "", table, "", extra, "", "## Regenerate (from the repository root)", "", "```", commands, "```", ""]
    (DATA / f"mech_{name}.md").write_text("\n".join(text))


def record(module, name, numbers, runs, what=None):
    """Write DATA/mech_<name>.md, the one record format. module is the writer's module name (__spec__.name). What it shows
    is `what`, else the first paragraph of the module's docstring; Aggregation is the rest of that docstring; Runs is
    settings.runs_text(...); Numbers is what the module computed; Regenerate is the script step that runs the module."""
    doc = (_module(module).__doc__ or "").strip().split("\n\n", 1)
    folder = module.split(".")[1] if module.startswith("aux.") else None
    command = f"scripts/aux.sh {folder} --execute" if folder else "scripts/main.sh table1 --execute"
    text = [f"# mech_{name}", "", "## What it shows", "", what or doc[0], "", "## Runs", "", runs, "",
            "## Aggregation", "", doc[1] if len(doc) > 1 else "", "", "## Numbers", "", numbers, "",
            "## Regenerate (from the repository root)", "", "```", command, "```", ""]
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"mech_{name}.md").write_text("\n".join(text))


def _module(name):
    """The loaded module named name, also when it runs as __main__ (python -m)."""
    for m in list(sys.modules.values()):
        if getattr(getattr(m, "__spec__", None), "name", None) == name:
            return m
    return importlib.import_module(name)


def md_table(head, rows):
    lines = ["| " + " | ".join(str(h) for h in head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def find_existing(directories, filename):
    for directory in directories:
        if (directory / filename).is_file():
            return directory / filename
    raise FileNotFoundError(filename)


class Report:
    """Print lines and keep them; save() writes them under the module's analysis cache."""

    def __init__(self):
        self.lines = []

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        self.lines.append(s)
        print(s)

    def save(self, module, name):
        (out_dir(module) / name).write_text("\n".join(self.lines) + "\n")
