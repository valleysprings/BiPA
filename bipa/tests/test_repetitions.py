import json
from pathlib import Path

import pytest

from bipa import repeat
def evaluation(root, repetition, outcomes, text="seed"):
    path = root / f"results/evaluations/test/{repeat.version(text)[:12]}-{repetition}/evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version_id": repeat.version(text), "split": "test",
        "panel": [{"task_id": task, "repetition": repetition} for task in outcomes],
        "episodes": [{"task_id": task, "repetition": repetition, "success": answer} for task, answer in outcomes.items()]}))
    return path


def test_missing_seed_repetitions_never_claim_avg3(tmp_path):
    run, out = tmp_path / "run", tmp_path / "reps"
    evaluation(run, 0, {"a": True, "b": False})
    result = repeat.collect(run, out, repeat.version("seed"), ["a", "b"], 3)
    assert result["avg"] is None and not result["repetitions_complete"]
    assert [r["known"] for r in result["runs"]] == [2, 0, 0]
    evaluation(out, 1, {"a": True, "b": True})
    evaluation(out, 2, {"a": False, "b": False})
    result = repeat.collect(run, out, repeat.version("seed"), ["a", "b"], 3)
    assert result["repetitions_complete"] and result["avg"] == 0.5


def test_unknown_wrong_skill_and_wrong_panel_do_not_complete_repetitions(tmp_path):
    run, out = tmp_path / "run", tmp_path / "reps"
    evaluation(run, 0, {"a": True})
    evaluation(out, 1, {"a": None})
    evaluation(out, 2, {"a": True}, text="different skill")
    result = repeat.collect(run, out, repeat.version("seed"), ["a"], 3)
    assert result["avg"] is None
    assert not result["runs"][1]["outcomes_complete"]
    assert not result["runs"][2]["panel_complete"]
    evaluation(out, 2, {"other-task": True})
    assert not repeat.collect(run, out, repeat.version("seed"), ["a"], 3)["repetitions_complete"]


def test_repetitions_use_frozen_test_panel_even_if_current_defaults_change(tmp_path):
    protocol = dict(model="m", benchmark="b", arm="skillopt", harness="h", scale="main",
                    config_hash="fixture", protocol="native-v1", panels={"test": ["frozen-task"]})
    (tmp_path / "native_protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "summary.json").write_text(json.dumps({**protocol, "final_skill": "seed"}))
    (tmp_path / "final").mkdir(); (tmp_path / "final/SKILL.md").write_text("seed")
    assert repeat.source(tmp_path)[2] == ["frozen-task"]
    (tmp_path / "final/SKILL.md").write_text("edited")
    with pytest.raises(ValueError, match="Final skill differs"):
        repeat.source(tmp_path)


def test_rep0_supplements_preserve_known_verdicts_and_provenance(tmp_path):
    run, out = tmp_path / "run", tmp_path / "reps"
    original = evaluation(run, 0, {"passed": True, "failed": False, "unknown": None})
    before = original.read_bytes()
    # A supplemental verdict must never override either original known outcome.
    supplement = evaluation(out, 0, {"passed": False, "failed": True, "unknown": True})
    result = repeat.collect(run, out, repeat.version("seed"), ["passed", "failed", "unknown"], 1)
    assert result["avg"] == 2 / 3
    assert result["runs"][0]["missing_tasks"] == []
    assert result["runs"][0]["evaluation_paths"] == [str(original), str(supplement)]
    assert original.read_bytes() == before


def test_rep0_partial_supplements_reject_unrelated_version_and_panel(tmp_path):
    run, out = tmp_path / "run", tmp_path / "reps"
    evaluation(run, 0, {"known": False, "missing": None})
    evaluation(out, 0, {"missing": True}, text="wrong skill")
    evaluation(out, 1, {"missing": True})
    evaluation(out, 0, {"missing": True, "unrelated": True})
    result = repeat.collect(run, out, repeat.version("seed"), ["known", "missing"], 1)
    assert result["avg"] is None
    assert result["runs"][0]["missing_tasks"] == ["missing"]
    # A matching panel with no matching episode is still incomplete.
    path = evaluation(out, 0, {"missing": True})
    data = json.loads(path.read_text()); data["episodes"] = []
    path.write_text(json.dumps(data))
    assert repeat.collect(run, out, repeat.version("seed"), ["known", "missing"], 1)["avg"] is None
    evaluation(out, 0, {"missing": True})
    assert repeat.collect(run, out, repeat.version("seed"), ["known", "missing"], 1)["avg"] == 0.5


def test_execute_recovers_only_missing_rows_and_keeps_source_unchanged(tmp_path, monkeypatch):
    from bipa.utils import executor
    run, out = tmp_path / "run", tmp_path / "reps"
    original = evaluation(run, 0, {"passed": True, "failed": False, "unknown": None})
    before = original.read_bytes()
    protocol = dict(benchmark="fixture", model="gpt-6-luna", harness="fixture", cache_root=str(tmp_path / "cache"))
    monkeypatch.setattr(repeat, "source", lambda root: (protocol, "seed", ["passed", "failed", "unknown"]))
    monkeypatch.setattr(repeat.config, "repetition_dir", lambda root: out)
    monkeypatch.setattr(repeat.config, "configure_environment", lambda: None)
    calls = []
    def execute(cell, text, rows, split):
        calls.append((cell.root, text, rows, split))
        evaluation(out, rows[0]["repetition"], {row["task_id"]: True for row in rows})
    monkeypatch.setattr(executor, "evaluate", execute)
    repeat.main(["--run", str(run), "--test-runs", "2", "--execute"])
    assert [[row["task_id"] for row in call[2]] for call in calls] == [["unknown"], ["passed", "failed", "unknown"]]
    assert [call[2][0]["repetition"] for call in calls] == [0, 1]
    assert all(call[0] == out.resolve() and call[3] == "test" for call in calls)
    assert original.read_bytes() == before
    assert json.loads((out / "results.json").read_text())["repetitions_complete"]


@pytest.mark.parametrize("corruption", ["duplicate", "unplanned"])
@pytest.mark.parametrize("location", ["source", "supplement"])
def test_invalid_episode_slots_cannot_complete_repetitions(tmp_path, corruption, location):
    run, out = tmp_path / "run", tmp_path / "reps"
    original = evaluation(run, 0, {"known": True, "missing": None})
    if location == "source":
        path = original
    else:
        path = evaluation(out, 0, {"missing": True})
    data = json.loads(path.read_text())
    if corruption == "duplicate":
        data["episodes"].append({**data["episodes"][0], "success": False})
    else:
        data["episodes"].append({"task_id": "outside-panel", "repetition": 0, "success": True})
    path.write_text(json.dumps(data))
    result = repeat.collect(run, out, repeat.version("seed"), ["known", "missing"], 1)
    assert result["avg"] is None and not result["repetitions_complete"]
    assert result["runs"][0]["known"] == (0 if location == "source" else 1)
