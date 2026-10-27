"""aux/success and aux/three_word_seed: the small rules they rest on, and the three-word-seed coverage audit on the recorded runs."""
from pathlib import Path

import pytest
import yaml

from src.utils import layout

ROOT = Path(__file__).resolve().parents[1]
reruns = pytest.importorskip("aux.success.reruns")
domains = pytest.importorskip("aux.success.domains")
collect = pytest.importorskip("aux.three_word_seed.collect")
recorded = pytest.mark.skipif(not (layout.EXECUTION / "gpt-6-luna/skillopt/automationbench/fromscratch").is_dir(),
                              reason="needs the recorded runs under BIPA_ASSETS")


def test_tokens_count_each_request_once_and_skip_failed_ones():
    reqs = [{"request_id": "a", "input_tokens": 5}, {"request_id": "a", "input_tokens": 5}, {"request_id": "b", "input_tokens": None},
            {"request_id": "c", "input_tokens": 7}]
    assert reruns.tokens({"requests": reqs}) == 12
    assert reruns.tokens({"requests": [{"request_id": "b", "input_tokens": None}]}) is None


def test_domain_bound_is_the_second_largest_gap_rounded_up():
    cell = lambda q: {k: {"q": q} for k in [*domains.DOMAINS["tau3-bench"], *domains.DOMAINS["automationbench"]]}  # noqa: E731
    res = {b: {"native-seed": cell(50.0), "gepa": cell(50.0), "skillopt": cell(50.0), "skilladam": cell(50.0)} for b in domains.DOMAINS}
    res["automationbench"]["skillopt"]["sales"]["q"] = 36.7
    res["tau3-bench"]["skilladam"]["retail"]["q"] = 42.76
    gaps, bound = domains.gaps(res)
    assert gaps[0][:3] == ("automationbench", "sales", "skillopt") and bound == pytest.approx(7.3)


def test_sentences_skip_front_matter_headings_and_fragments():
    text = "---\nname: x\n---\n# Title\n- Read the file first. Then act; check twice.\n1. Go.\n"
    assert collect.sentences(text) == ["Read the file first.", "Then act; check twice."]


@recorded
def test_every_sentence_of_the_returned_skills_has_a_label(monkeypatch):
    monkeypatch.setattr(layout, "AUX_RUNS", yaml.safe_load((ROOT / ".scratch/repin-live/parity/aux_runs.yaml").read_text())["aux_runs"])
    counts = {(o, b): len(collect.labelled(b, o)) for b in collect.BENCHES for o in collect.OPTIMIZERS}
    assert counts == {("skillopt", "automationbench"): 19, ("gepa", "automationbench"): 28, ("skillopt", "tau3-bench"): 18,
                      ("gepa", "tau3-bench"): 28}


@recorded
def test_coverage_audit(monkeypatch):
    """The numbers of the c-scratch coverage paragraph that do not depend on the Luna SkillAdam pin."""
    from aux.three_word_seed import analysis
    from src.analysis import settings as S
    monkeypatch.setattr(layout, "AUX_RUNS", yaml.safe_load((ROOT / ".scratch/repin-live/parity/aux_runs.yaml").read_text())["aux_runs"])
    agents = collect.agents("automationbench")
    rows, prose = analysis.coverage({name: S.wins(S.rows(fs, S.LUNA)) for name, fs in agents.items()})
    got = {r[0].split(" (")[0]: r[1:] for r in rows}
    assert got["covers / loses against the three-word seed"] == ["16 / 4", "18 / 3"]
    assert got["of those it covers, solved by no skill or the hand-written seed"] == [12, 12]
    assert got["points from the 63 tasks no skill or the hand-written seed solves / the other 57"] == ["16.7 / 1.7", "16.7 / 2.8"]
    assert got["of the 35 tasks the three-word seed solves in one or two runs, solved in all three"] == [20, 22]
    assert got["from the hand-written seed: covers / loses against it"] == ["13 / 4", "11 / 6"]
    assert got["tasks covered by its run from one seed only"] == [14, 11]
    assert "25.0 against 27.5, covers / loses 9 / 11" in prose and "seeds: 10;" in prose
