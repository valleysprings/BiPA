"""Audit records of the analyses (OUT/data/mech)."""
import importlib, sys

from src.utils import layout

DATA = layout.OUT / "data/mech"


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

