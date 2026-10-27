import os

import pytest

from src import run
from src.utils import layout
from src.analysis import settings as S, table1
from src.utils import dataloader as C


def finished(path, mtime):
    path.mkdir(parents=True)
    (path / "summary.json").write_text('{"final_skill": "s"}')
    os.utime(path / "summary.json", (mtime, mtime))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.delenv("BIPA_MANIFEST", raising=False)
    monkeypatch.setattr(C, "EX", tmp_path)
    for model in C.MODELS:
        for method in C.METHODS:
            for bench in C.BENCHES:
                finished(tmp_path / model / method / bench / C.PINS[model][method][bench], 1)
                finished(tmp_path / model / method / bench / "native-29991231", 2)
    return tmp_path


def test_every_cell_is_pinned():
    assert {(m, k, b) for m in C.PINS for k in C.PINS[m] for b in C.PINS[m][k]} == \
        {(m, k, b) for m in C.MODELS for k in C.METHODS for b in C.BENCHES}


def test_table1_reads_the_pinned_runs(tree, monkeypatch):
    monkeypatch.setattr(S, "EX", tree)
    assert table1.S.run is S.run  # Table 1 and every analysis resolve a cell's run the same way: the pins only
    for model in C.MODELS:
        for method in C.METHODS:
            for bench in C.BENCHES:
                assert S.run(model, bench, method) == tree / model / method / bench / C.PINS[model][method][bench]


def test_deepseek_seed_is_the_recorded_run():
    assert {C.PINS["deepseek-flash"]["native-seed"][b] for b in C.BENCHES} == {"native-20261003-trace-final"}


def test_unpinned_cell_with_several_runs_raises(tree):
    finished(tree / "deepseek-flash/other/tau3-bench/native-1", 1)
    finished(tree / "deepseek-flash/other/tau3-bench/native-2", 2)
    with pytest.raises(LookupError, match="native-1, native-2"):
        C.current_run("other", "deepseek-flash", "tau3-bench")


def test_unpinned_cell_with_one_run(tree):
    finished(tree / "deepseek-flash/other/tau3-bench/native-1", 1)
    (tree / "deepseek-flash/other/tau3-bench/native-2").mkdir()
    (tree / "deepseek-flash/other/automationbench/native-1").mkdir(parents=True)
    assert C.current_run("other", "deepseek-flash", "tau3-bench").name == "native-1"
    assert C.current_run("other", "deepseek-flash", "automationbench") is None
    assert C.current_run("other", "deepseek-flash", "terminal-bench-2-1") is None


def test_missing_pinned_run_raises(tree, monkeypatch):
    monkeypatch.setitem(C.PINS["deepseek-flash"]["gepa"], "tau3-bench", "native-missing")
    with pytest.raises(FileNotFoundError):
        C.current_run("gepa", "deepseek-flash", "tau3-bench")
    (tree / "deepseek-flash/gepa/tau3-bench/native-missing").mkdir()
    with pytest.raises(FileNotFoundError, match="summary.json"):
        C.current_run("gepa", "deepseek-flash", "tau3-bench")


@pytest.fixture
def transfer_plan(tree, monkeypatch, capsys):
    monkeypatch.setattr(layout, "EXECUTION", tree)
    monkeypatch.setattr(run, "EXECUTION", tree / "out")

    def plan(*flags):
        run.main(["transfer", "skilladam", "tau3", "gpt-6-luna", "--to", "deepseek-flash", *flags])
        return capsys.readouterr().out.split(" -> ")[0]
    return plan


def test_transfer_reads_the_pinned_run(transfer_plan, tree):
    cell = tree / "gpt-6-luna/skilladam/tau3-bench"
    assert transfer_plan() == str(cell / C.PINS["gpt-6-luna"]["skilladam"]["tau3-bench"])
    assert transfer_plan("--tag", "native-29991231") == str(cell / "native-29991231")


def test_transfer_of_an_unpinned_cell_with_several_runs_raises(transfer_plan, monkeypatch):
    monkeypatch.delitem(C.PINS["gpt-6-luna"]["skilladam"], "tau3-bench")
    with pytest.raises(LookupError, match="pin one"):
        transfer_plan()
