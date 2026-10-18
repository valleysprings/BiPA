from pathlib import Path
from types import SimpleNamespace

from src.core import launch
def test_suite_schedules_repetitions_for_all_five_methods(tmp_path, monkeypatch):
    monkeypatch.setattr(launch.config, "EXECUTION", tmp_path)
    args = SimpleNamespace(action="suite", model="gpt-6-luna", benches=["tau3"],
                           arms=list(launch.ARMS), tag=None, slots=3, test_runs=3)
    jobs = launch.plan(args)
    assert len(jobs) == 10
    for index, arm in enumerate(launch.ARMS):
        native, repeat = jobs[index * 2:index * 2 + 2]
        assert native["command"][native["command"].index("--arm") + 1] == arm
        assert native["out"] == Path(repeat["command"][repeat["command"].index("--run") + 1])
        assert repeat["command"][-3:] == ["--test-runs", "3", "--execute"]
        assert "bach-harness" not in " ".join(native["command"][2:])


def test_default_preview_does_not_execute_or_create_outputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch.config, "EXECUTION", tmp_path)
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("execution")))
    launch.main(["run", "tau3", "gpt-6-luna", "--arm", "skillopt"])
    assert "--protocol native-v1" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_finished_optimization_skips_but_repetitions_still_run(tmp_path, monkeypatch):
    native, repeats = tmp_path / "native", tmp_path / "repetitions"
    native.mkdir(); (native / "summary.json").write_text("{}")
    calls = []
    monkeypatch.setattr(launch.subprocess, "run", lambda argv, **kw: calls.append(argv))
    monkeypatch.setattr(launch.config, "configure_environment", lambda: None)
    launch.execute([{"out": native, "repeat": False, "command": ["optimizer"]},
                    {"out": repeats, "repeat": True, "command": ["repetitions"]}])
    assert calls == [["repetitions"]]
