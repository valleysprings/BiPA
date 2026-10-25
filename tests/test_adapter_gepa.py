"""GEPA adapter on the vendored upstream engine with a fake executor and LLM."""
import pickle
from collections import Counter
from types import SimpleNamespace

import pytest

from src.adapters import gepa

CELL = SimpleNamespace(benchmark="offline")


def failing_llm():
    def complete(messages, **kwargs):
        raise RuntimeError("offline reflection failure")
    return SimpleNamespace(complete=complete, usage=Counter())


def run_gepa(monkeypatch, out, batches, success=lambda split: 0, on_train=lambda: None, calls=None):
    calls, rounds = [] if calls is None else calls, []

    def evaluate(cell, text, ids, split):
        calls.append(split)
        if split == "train":
            on_train()
        return [dict(id=task, success=success(split), verifier=None, messages=[], evaluation_path=f"{split}.json")
                for task in ids]

    monkeypatch.setattr(gepa, "evaluate", evaluate)
    monkeypatch.setattr(gepa.executor, "record_round", lambda run_root, **kw: rounds.append((kw["step"], kw["role"])))
    result = gepa.run(CELL, {"sel": ["dev-task"]}, batches, failing_llm(), out)
    return result, calls, rounds


def test_failed_reflection_skips_the_iteration(monkeypatch, tmp_path):
    result, _, rounds = run_gepa(monkeypatch, tmp_path, [["t1"], ["t2"]])
    assert result["final_skill"] == gepa.SEED
    assert (result["extra"]["rounds_run"], result["extra"]["T"]) == (2, 2)
    assert rounds == [(0, "full_selection"), (1, "minibatch"), (2, "minibatch")]
    assert "offline reflection failure" in (tmp_path / "gepa/run_log.txt").read_text()


@pytest.mark.parametrize("fault", ["train", "dev"])
def test_unknown_outcome_aborts_and_resume_retries_the_slot(monkeypatch, tmp_path, fault):
    calls = []
    for _ in range(2):
        with pytest.raises(gepa.IncompleteGEPAEvaluation, match="Unknown GEPA outcomes"):
            run_gepa(monkeypatch, tmp_path, [["t1"]], success=lambda split: None if split == fault else 0, calls=calls)
        if fault == "train":
            with (tmp_path / "gepa/gepa_state.bin").open("rb") as stream:
                assert pickle.load(stream)["i"] == -1
    assert calls.count(fault) == 2
    assert not issubclass(gepa.IncompleteGEPAEvaluation, Exception)


def test_stale_stop_file_is_refused(monkeypatch, tmp_path):
    (tmp_path / "gepa").mkdir()
    (tmp_path / "gepa/gepa.stop").touch()
    with pytest.raises(RuntimeError, match="stale stop signal"):
        run_gepa(monkeypatch, tmp_path, [["t1"], ["t2"]])
    assert not (tmp_path / "gepa/gepa_state.bin").exists()


def test_stop_during_the_run_fails(monkeypatch, tmp_path):
    with pytest.raises(gepa.IncompleteGEPAEvaluation, match="^GEPA ran 1 of T=2 iterations: gepa.stop appeared"):
        run_gepa(monkeypatch, tmp_path, [["t1"], ["t2"]], on_train=lambda: (tmp_path / "gepa/gepa.stop").touch())


def test_side_info_keeps_the_whole_conversation():
    long = "evidence" * 20000
    row = {"id": "task", "success": 0, "verifier": "failed",
           "messages": [{"role": "system", "content": "system"},
                        {"role": "user", "content": "task <skill>skill</skill>"},
                        {"role": "assistant", "content": long,
                         "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}]}]}
    score, info = gepa.side_info(row)
    trace = info["Generated Outputs"]["Agent Trace"]
    assert score == 0 and info["Feedback"] == {"Status": "failure", "Test Output": "failed"}
    assert long in trace and "[SYSTEM]\nsystem" in trace and "<skill>skill</skill>" in trace and "\nbash({})" in trace
    assert info["Input"]["Problem"] == "task <skill>skill</skill>"
