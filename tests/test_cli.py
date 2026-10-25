"""SkillAdam's gate on the command line: a run gates on the training batch, records that in its protocol and summary,
takes no gate flags, and a run recorded with the dev-batch gate is not resumed."""
import json

import pytest

from src import run

TRAIN_BATCH = {"gate": "train-batch"}
REFUSED = "was started with the dev-batch gate, which this code does not run; resume it with the original code"


def started(tmp_path, arm="skilladam", **options):
    """A started run on tau3-bench recorded with ``options`` (none: a run from before arm_options were recorded)."""
    out = tmp_path / f"{arm}/tau3-bench/t"
    out.mkdir(parents=True)
    (out / "native_protocol.json").write_text(json.dumps({
        "protocol": "native-v1", "feedback_mode": "full", "arm": arm, "model": "deepseek-flash",
        "benchmark": "tau3-bench", "harness": "opencode", "scale": "main", "cache_root": str(tmp_path / "c"),
        "panels": {"train": ["a"], "sel": ["b"], "test": ["c"]}, "batches": [["a"]], **options}))
    return out


def skilladam(*flags):
    run.main(["run", "skilladam", "tau3", "deepseek-flash", *flags, "--execute"])


def test_a_new_run_gates_on_the_training_batch(stub_run, tmp_path):
    skilladam()
    out = tmp_path / "skilladam/tau3-bench/t"
    assert stub_run.adapted == [[["a"]]]  # the adapter takes no gate options
    assert json.loads((out / "native_protocol.json").read_text())["arm_options"] == TRAIN_BATCH
    assert json.loads((out / "summary.json").read_text())["arm_options"] == TRAIN_BATCH


@pytest.mark.parametrize("flags", [["--skilladam-gate", "train-batch"], ["--skilladam-gate", "dev-batch"],
                                   ["--skilladam-validation-size", "10"]])
def test_the_removed_gate_flags_are_rejected(stub_run, tmp_path, flags):
    with pytest.raises(SystemExit):
        skilladam(*flags)
    with pytest.raises(SystemExit):
        run.main(["suite", "deepseek-flash", "--methods", "skilladam", *flags])
    assert not (tmp_path / "skilladam").exists() and not stub_run.adapted


@pytest.mark.parametrize("options", [{}, {"arm_options": {}}, {"arm_options": {"gate": "dev-batch",
                                                                                "validation_size": 30}}])
def test_a_run_recorded_with_the_dev_batch_gate_is_refused_on_resume(stub_run, tmp_path, options):
    """Recorded SkillAdam runs without arm_options used the dev-batch gate."""
    out = started(tmp_path, **options)
    before = (out / "native_protocol.json").read_text()
    with pytest.raises(SystemExit, match=REFUSED):
        skilladam("--resume")
    assert not stub_run.adapted and not stub_run.evaluated and not (out / "summary.json").exists()
    assert (out / "native_protocol.json").read_text() == before


def test_a_training_batch_run_resumes(stub_run, tmp_path):
    out = started(tmp_path, arm_options=TRAIN_BATCH)
    skilladam("--resume")
    assert stub_run.adapted == [[["a"]]]
    assert json.loads((out / "summary.json").read_text())["arm_options"] == TRAIN_BATCH


def test_other_arms_resume_without_recorded_arm_options(stub_run, tmp_path):
    out = started(tmp_path, arm="gepa")
    run.main(["run", "gepa", "tau3", "deepseek-flash", "--resume", "--execute"])
    assert stub_run.adapted == [[["a"]]] and json.loads((out / "summary.json").read_text())["arm_options"] == {}


def test_a_finished_dev_batch_run_is_left_alone(stub_run, tmp_path, capsys):
    out = started(tmp_path)
    (out / "summary.json").write_text("{}")
    skilladam("--resume")
    assert "already finished" in capsys.readouterr().out and not stub_run.adapted


def test_plans_carry_no_gate_flags(stub_run, tmp_path, capsys):
    started(tmp_path)  # planning a resume does not run it
    run.main(["suite", "deepseek-flash", "--benches", "tau3", "--methods", "skilladam", "gepa", "--tag", "t",
              "--cache-root", str(tmp_path / "c")])
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[4:6] for line in lines] == [["run", "skilladam"], ["repeat", "skilladam"],
                                                    ["run", "gepa"], ["repeat", "gepa"]]
    assert lines[0].endswith("--resume --execute") and all("--skilladam" not in line for line in lines)
    assert all(f"--cache-root {tmp_path / 'c'}" in line for line in lines[::2])
    assert all("--cache-root" not in line for line in lines[1::2])  # repetitions read the run's own cache root
    assert sorted(p.name for p in tmp_path.iterdir()) == ["skilladam"] and not stub_run.adapted
