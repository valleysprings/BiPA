import hashlib
import json

import pytest

from src import repeat, repeat_transfer


def transfer_run(root, outcomes, text="skill"):
    run = root / "gpt-6-luna" / "method" / "bench" / "ood-from-train"
    vid = repeat.version(text)
    evaluation = run / f"results/evaluations/test/{vid[:12]}-0/evaluation.json"
    evaluation.parent.mkdir(parents=True, exist_ok=True)
    evaluation.write_text(json.dumps({"version_id": vid, "split": "test",
        "panel": [{"task_id": task, "repetition": 0} for task in outcomes],
        "episodes": [{"task_id": task, "repetition": 0, "success": answer} for task, answer in outcomes.items()]}))
    delivery = evaluation.parents[3] / "skills/deliveries/evaluations/test" / evaluation.parent.name / "native/SKILL.md"
    delivery.parent.mkdir(parents=True, exist_ok=True)
    delivery.write_text(text)
    (delivery.parent.parent / "identity.json").write_text(json.dumps(
        {"version_id": vid, "files": {"SKILL.md": hashlib.sha256(text.encode()).hexdigest()}}))
    return run, vid


def test_source_recovers_panel_version_and_skill_from_the_evaluation(tmp_path):
    run, vid = transfer_run(tmp_path, {"a": True, "b": None})
    model, bench, found_vid, text, tasks = repeat_transfer.source(run)
    assert (model, bench, found_vid, text, tasks) == ("gpt-6-luna", "bench", vid, "skill", ["a", "b"])
    result = repeat_transfer.collect(run, tmp_path / "reps", vid, tasks, 3)
    assert [r["missing_tasks"] for r in result["runs"]] == [["b"], ["a", "b"], ["a", "b"]]


def test_source_rejects_a_tampered_delivery(tmp_path):
    run, _ = transfer_run(tmp_path, {"a": True})
    delivery = next(run.glob("results/skills/**/native/SKILL.md"))
    delivery.write_text("edited")
    with pytest.raises(ValueError, match="identity record"):
        repeat_transfer.source(run)
    identity = delivery.parent.parent / "identity.json"
    identity.write_text(json.dumps({"files": {"SKILL.md": hashlib.sha256(b"edited").hexdigest()}}))
    with pytest.raises(ValueError, match="evaluated version"):
        repeat_transfer.source(run)


def test_execute_recovers_only_missing_rows_and_keeps_source_unchanged(tmp_path, monkeypatch):
    from src.utils import executor
    run, vid = transfer_run(tmp_path, {"passed": True, "unknown": None})
    out = tmp_path / "reps"
    original = next(run.glob("results/evaluations/test/*/evaluation.json"))
    before = original.read_bytes()
    monkeypatch.setattr(repeat_transfer.config, "repetition_dir", lambda root: out)
    monkeypatch.setattr(repeat_transfer.config, "configure_environment", lambda: None)
    calls = []
    def execute(cell, text, rows, split):
        calls.append((cell.root, text, rows, split))
        evaluation = out / f"results/evaluations/test/{vid[:12]}-{rows[0]['repetition']}/evaluation.json"
        evaluation.parent.mkdir(parents=True, exist_ok=True)
        evaluation.write_text(json.dumps({"version_id": vid, "split": "test",
            "panel": rows, "episodes": [{**row, "success": True} for row in rows]}))
    monkeypatch.setattr(executor, "evaluate", execute)
    repeat_transfer.main(["--run", str(run), "--test-runs", "2", "--execute"])
    assert [[row["task_id"] for row in call[2]] for call in calls] == [["unknown"], ["passed", "unknown"]]
    assert [call[2][0]["repetition"] for call in calls] == [0, 1]
    assert all(call[0] == out.resolve() and call[1] == "skill" and call[3] == "test" for call in calls)
    assert original.read_bytes() == before
    assert json.loads((out / "results.json").read_text())["repetitions_complete"]
