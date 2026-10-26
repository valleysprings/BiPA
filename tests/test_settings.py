import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src import layout
from src.analysis import settings as S
from src.analysis import table1 as T
from src.utils import dataloader as C

ROOT = Path(__file__).resolve().parents[1]
RECORDED = (layout.EXECUTION / "gpt-6-luna/skilladam/tau3-bench").is_dir()
recorded = pytest.mark.skipif(not RECORDED, reason="needs the recorded runs under BIPA_ASSETS")

# the Luna pins the paper's 10-24 version was computed on: every Luna optimizer run of 12 October
OLD_LUNA = {method: {bench: "native-20261012" for bench in C.BENCHES} for method in S.OPTIMIZERS}


def test_one_experiments_file():
    assert C.PINS is layout.PINS and layout.MODELS is layout._CONFIG["models"]


def test_experiments_file_from_the_environment(tmp_path):
    config = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
    config["pins"]["gpt-6-luna"]["skilladam"] = {"default": "native-20261025", "tau3-bench": "native-test"}
    alt = tmp_path / "experiments.yaml"
    alt.write_text(yaml.safe_dump(config))
    code = "from src.utils import dataloader as C; from src import layout; print(C.PINS['gpt-6-luna']['skilladam']['tau3-bench'], layout.ANALYSIS)"
    env = dict(os.environ, BIPA_EXPERIMENTS=str(alt), BIPA_ANALYSIS_CACHE=str(tmp_path / "analysis"))
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout.split()
    assert out == ["native-test", str(tmp_path / "analysis")]


def test_streams_are_named_independent_and_the_same_in_every_process():
    a = S.rng("pooled", "dq_task").random(4)
    assert np.array_equal(a, S.rng("pooled", "dq_task").random(4))
    assert not np.array_equal(a, S.rng("pooled", "lr_task").random(4))
    code = "from src.analysis import settings as S; print(repr(S.rng('pooled', 'dq_task').random()))"
    env = dict(os.environ, PYTHONHASHSEED="123")
    other = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
    assert float(other) == a[0]
    assert S.boot_index(7, "x").shape == (S.N_BOOT, 7)


def test_intervals():
    draws = np.arange(1, 10001, dtype=float)
    lo, hi = S.interval(draws)
    assert (round(lo, 3), round(hi, 3)) == (250.975, 9750.025)
    assert round(S.upper(draws), 2) == 9500.05 and round(S.lower(draws), 2) == 500.95
    assert tuple(round(x, 4) for x in S.wilson(5, 10)) == (0.2366, 0.7634)


@recorded
def test_learned_skills_follow_the_pins(monkeypatch):
    learned = S.learned()
    assert len(learned) == 16 and sum(m == S.LUNA for m, _, _ in learned) == 7
    assert len(S.learned_pairs()) == 6
    monkeypatch.setitem(C.PINS, "gpt-6-luna", {**C.PINS["gpt-6-luna"], **OLD_LUNA})
    learned = S.learned()
    assert len(learned) == 15 and (S.LUNA, "terminal-bench-2-1", "skilladam") not in learned
    assert len(S.learned_pairs()) == 5


@recorded
def test_dev_gate_is_read_from_the_run():
    assert not S.dev_gated(S.LUNA, "tau3-bench", "skilladam")  # the training-batch gate
    assert S.dev_gated(S.DEEPSEEK, "tau3-bench", "skilladam") and S.dev_gated(S.LUNA, "tau3-bench", "gepa")
    assert not S.dev_gated(S.LUNA, "tau3-bench", "native-seed")


@recorded
def test_cell_files_are_table1s():
    for model, method in ((S.DEEPSEEK, "gepa"), (S.LUNA, "gepa")):
        seed_run, seed_reps = T.seed_files(model, "terminal-bench-2-1")
        ref = T.merged_mean_rows(seed_reps, T.row_reader(model))[1]
        cell = T.cell(model, "terminal-bench-2-1", method, seed_run, seed_reps, ref)
        files = S.cell_files(model, "terminal-bench-2-1", method)
        assert cell["files"] == {k: [str(f) for f in fs] for k, fs in sorted(files.items())}
    assert S.returned_seed(S.LUNA, "terminal-bench-2-1", "gepa")
    assert S.cell_files(S.LUNA, "terminal-bench-2-1", "gepa") == S.cell_files(S.LUNA, "terminal-bench-2-1", "native-seed")
    rows = S.cell_rows(S.LUNA, "terminal-bench-2-1", "native-seed")
    assert sorted(rows) == [0, 1, 2] and all(len(r) == 18 for r in rows.values())
    assert sum(r["success"] is not None for r in rows[0].values()) == 17  # Luna TB2: Q over verified outcomes


def test_settings_do_not_import_table1_at_load():
    code = "import sys; from src.analysis import settings; print('src.analysis.table1' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


def test_runs_come_only_from_the_pins(monkeypatch):
    monkeypatch.setitem(C.PINS, "gpt-6-luna", {k: v for k, v in C.PINS["gpt-6-luna"].items() if k != "gepa"})
    with pytest.raises(LookupError):
        S.run(S.LUNA, "tau3-bench", "gepa")


def test_a_manifest_is_refused(tmp_path):
    env = dict(os.environ, BIPA_MANIFEST=str(tmp_path / "m.json"))
    got = subprocess.run([sys.executable, "-c", "from src.analysis import settings"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert got.returncode != 0 and "BIPA_MANIFEST" in got.stderr


def test_task_helpers():
    assert S.domain_of("tau3-bench", "tau3-airline-12") == "airline"
    assert S.domain_of("automationbench", "zendesk-ticket-3") == "zendesk" and S.domain_of("terminal-bench-2-1", "x") == "all"
    assert S.slot("t") == ("t", 0) and S.slot({"task_id": "t", "repetition": 2}) == ("t", 2)
    assert S.known({"success": False}) and not S.known({"success": None}) and not S.known({"success": True, "outcome_valid": False})
    assert S.Z95 == 1.96 and S.SEED_VERSION.startswith("4fd5ca441c55")


@recorded
def test_named_runs_that_no_cell_pins(monkeypatch):
    parity = yaml.safe_load((ROOT / ".scratch/repin-live/parity/aux_runs.yaml").read_text())
    monkeypatch.setattr(layout, "AUX_RUNS", parity["aux_runs"])
    pool = S.extra("seed_dev_pool", S.LUNA, "terminal-bench-2-1")
    assert [p.parent.parent.name + "/" + p.name for p in pool] == [
        "native-seed/native-20261011", "gepa/native-20261012", "skillopt/native-20261012", "skilladam/native-20261012"]
    assert S.extra("transfer_model", S.LUNA, "automationbench") == []
    with pytest.raises(LookupError):
        S.extra("no such set", S.LUNA, "tau3-bench")


def test_wins_count_runs_over_tasks_verified_in_all_three():
    reps = {k: {"a": {"success": True}, "b": {"success": k == 0}, "c": {"success": None if k == 1 else True}} for k in range(3)}
    assert S.wins(reps) == {"a": 3, "b": 1} and S.wins({0: reps[0]}) == {}
