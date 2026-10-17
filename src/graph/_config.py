"""The `graph` section of the method config, frozen for the offline analyses.

In bach-harness this was `utils.config.load("bach")["graph"]` reading config/bach.yaml; bipa carries only the graph
section (classify vocabulary, context names, allocation weight) in graph.yaml next to this module, read once per process.
"""
import functools
from pathlib import Path

import yaml


@functools.lru_cache(maxsize=None)
def load(name):
    """The parsed graph.yaml ({"graph": ...}); the name argument is kept so callers read unchanged (only "bach" exists)."""
    assert name == "bach", name
    return yaml.safe_load((Path(__file__).resolve().parent / "graph.yaml").read_text("utf-8"))
