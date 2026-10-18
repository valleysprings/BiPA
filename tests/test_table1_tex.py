"""table1.tex: an incomplete avg@3 cell is withheld (tbd), never printed as a number."""
from src.analysis import table1


def test_paper_avg3_table_withholds_incomplete_repetitions(monkeypatch):
    monkeypatch.setattr(table1, "ROWS", [("Seed skill", "native-seed")])
    monkeypatch.setattr(table1, "TEX_BENCHES", ["bench"])
    monkeypatch.setitem(table1.MODEL_TEX, "offline", "Offline")
    cell = {"avg3_complete": False, "Q": 100.0}
    rendered = table1.tex({"offline|bench": {"Seed skill": cell}}, models=["offline"])
    assert "100.0" not in rendered and rendered.count(r"\tbd{}") == 4
