"""Repetitions of a frozen skill: coverage, unknown-only recovery and the adjudication of slots without a verdict."""
import json

import pytest
from harness_bench.eval import version_from_files

from src import run

VERSION = version_from_files({"SKILL.md": "seed"})["version_id"]


def evaluation(root, repetition, outcomes, version=VERSION):
    path = root / f"results/evaluations/test/{version[:12]}-{repetition}-{len(list(root.rglob('evaluation.json')))}/evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version_id": version, "split": "test",
                                "panel": [{"task_id": task, "repetition": repetition} for task in outcomes],
                                "episodes": [{"task_id": task, "repetition": repetition, "success": answer}
                                             for task, answer in outcomes.items()]}))
    return path


def test_missing_repetitions_never_claim_avg3(tmp_path):
    source, out = tmp_path / "run", tmp_path / "reps"
    evaluation(source, 0, {"a": True, "b": False})
    result = run.collect(source, out, VERSION, ["a", "b"], 3)
    assert result["avg"] is None and [r["known"] for r in result["runs"]] == [2, 0, 0]
    assert set(result["runs"][1]) == {"repetition", "tasks", "known", "panel_complete", "outcomes_complete", "Q_known",
                                      "evaluation_path", "evaluation_paths", "missing_tasks", "adjudicated_failures", "Q"}
    evaluation(out, 1, {"a": True, "b": True})
    evaluation(out, 2, {"a": False, "b": False}, version="other")
    assert run.collect(source, out, VERSION, ["a", "b"], 3)["avg"] is None
    evaluation(out, 2, {"a": False, "b": False})
    result = run.collect(source, out, VERSION, ["a", "b"], 3)
    assert result["repetitions_complete"] and result["avg"] == 0.5


def test_supplements_fill_unknown_slots_only(tmp_path):
    source, out = tmp_path / "run", tmp_path / "reps"
    original = evaluation(source, 0, {"passed": True, "failed": False, "unknown": None})
    supplement = evaluation(out, 0, {"passed": False, "failed": False, "unknown": True})
    result = run.collect(source, out, VERSION, ["passed", "failed", "unknown"], 1)
    assert result["avg"] == 2 / 3 and result["runs"][0]["evaluation_paths"] == [str(original), str(supplement)]


def test_one_repetition_never_adjudicates(tmp_path):
    source = tmp_path / "run"
    evaluation(source, 0, {"a": True, "b": None})
    result = run.collect(source, tmp_path / "reps", VERSION, ["a", "b"], 1)
    assert result["avg"] is None and result["adjudicated_failures"] == {} and result["runs"][0]["missing_tasks"] == ["b"]


def test_persistent_unknown_is_adjudicated_failure(tmp_path):
    source, out = tmp_path / "run", tmp_path / "reps"
    evaluation(source, 0, {"a": True, "b": None})
    evaluation(out, 1, {"a": True, "b": None})
    evaluation(out, 2, {"a": False, "b": None})
    result = run.collect(source, out, VERSION, ["a", "b"], 3)
    assert result["avg"] == 1 / 3 and not result["repetitions_complete"]
    assert result["adjudicated_failures"]["b"] == {"0": "unresolved", "1": "unresolved", "2": "unresolved"}
    other = tmp_path / "reps2"  # no episode of b in one repetition: it blocks instead
    evaluation(other, 1, {"a": True, "b": None})
    evaluation(other, 2, {"a": False})
    assert run.collect(source, other, VERSION, ["a", "b"], 3)["runs"][2]["missing_tasks"] == ["b"]


@pytest.mark.parametrize("extra", [{"task_id": "a", "repetition": 0, "success": False},
                                   {"task_id": "outside", "repetition": 0, "success": True}])
def test_duplicate_or_unplanned_episodes_are_ignored(tmp_path, extra):
    source = tmp_path / "run"
    path = evaluation(source, 0, {"a": True})
    data = json.loads(path.read_text())
    data["episodes"].append(extra)
    path.write_text(json.dumps(data))
    assert run.collect(source, tmp_path / "reps", VERSION, ["a"], 1)["avg"] is None


def test_repeat_runs_only_missing_slots(monkeypatch, tmp_path):
    source, out = tmp_path / "execution/m/skillopt/b/t", tmp_path / "reps"
    evaluation(source, 0, {"passed": True, "unknown": None})
    (source / "native_protocol.json").write_text(json.dumps(
        {"benchmark": "b", "model": "m", "cache_root": str(tmp_path / "c"), "panels": {"test": ["passed", "unknown"]}}))
    (source / "summary.json").write_text(json.dumps({"final_skill": "seed"}))
    monkeypatch.setattr(run, "run_dir", lambda *a: source)
    calls = []

    def evaluate(cell, text, rows, split):
        calls.append([row["task_id"] for row in rows])
        evaluation(out, rows[0]["repetition"], {row["task_id"]: True for row in rows})
    monkeypatch.setattr(run, "evaluate", evaluate)
    run.repeat(run.argparse.Namespace(method="skillopt", bench="b", model="m", tag="t", slots=1, test_runs=2), out)
    assert calls == [["unknown"], ["passed", "unknown"]]
    assert json.loads((out / "results.json").read_text())["avg"] == 1.0


def test_transfer_source_is_its_delivered_skill(tmp_path):
    source = tmp_path / "gpt-6-luna/gepa/tau3-bench/ood-from-deepseek-flash"
    path = evaluation(source, 0, {"a": True})
    delivery = source / "results/skills/deliveries/evaluations/test" / path.parent.name / "native/SKILL.md"
    delivery.parent.mkdir(parents=True)
    delivery.write_text("seed")
    bench, model, _, text, tasks = run.source(source)
    assert (bench, model, text, tasks) == ("tau3-bench", "gpt-6-luna", "seed", ["a"])
