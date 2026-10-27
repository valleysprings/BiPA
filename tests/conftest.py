import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from harness_bench.eval.panels import BENCH_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HARNESS_BENCH_ROOT", str(BENCH_ROOT))  # the registry of the installed harness-bench


@pytest.fixture
def stub_run(monkeypatch, tmp_path):
    """src.run offline under tmp_path: small panels, one batch, a scripted adapter and evaluator. Returns what they
    received: the adapter's batches, the evaluated (text, split, cache root) and the recorded rounds."""
    from src import run
    seen = SimpleNamespace(adapted=[], evaluated=[], rounds=[])
    monkeypatch.setattr(run, "run_dir", lambda method, bench, model, tag=None: tmp_path / method / bench / (tag or "t"))
    monkeypatch.setattr(run, "CACHE", tmp_path / "cache")
    monkeypatch.setattr("src.utils.llm.LLM", lambda model, out: SimpleNamespace(usage={"calls": 2}))
    monkeypatch.setattr(run, "panels", lambda bench, scale: {"train": ["a"], "sel": ["b"], "test": ["c", "d"]})
    monkeypatch.setattr(run, "batches", lambda bench, scale, method: [["a"]])
    monkeypatch.setattr(run, "record_round", lambda out, **record: seen.rounds.append(record))
    monkeypatch.setattr(run, "episodes", lambda out: {})
    monkeypatch.setattr(run, "ENV", {})

    def adapter(cell, panels, batches, llm, out):  # no options: a passed keyword fails the call
        seen.adapted.append(batches)
        return {"final_skill": "learned", "episodes": {"train": 1}, "extra": {}}
    load = run.importlib.import_module
    monkeypatch.setattr(run.importlib, "import_module", lambda name, *package: SimpleNamespace(run=adapter)
                        if name.startswith("src.adapters.") else load(name, *package))

    def evaluate(cell, text, ids, split):
        seen.evaluated.append((text, split, cell.cache_root))
        return [{"id": i, "success": 1, "cost": 2.0, "evaluation_path": "e"} for i in ids]
    monkeypatch.setattr(run, "evaluate", evaluate)
    return seen
