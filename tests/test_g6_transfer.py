"""aux/transfer: the statistics of the coverage record equal the old modules', and the small helpers do what they say."""
import math

import numpy as np
import pytest

coverage = pytest.importorskip("aux.transfer.coverage")
contamination = pytest.importorskip("aux.transfer.contamination")


def triples(n, seed):
    """Random (reference successes, own successes, agrees more) of n tasks."""
    g = np.random.default_rng(seed)
    return [(int(r), int(o), bool(a)) for r, o, a in zip(g.integers(0, 4, n), g.integers(0, 4, n), g.integers(0, 2, n))]


def test_similarity():
    assert coverage.similarity([], []) == 1.0
    assert coverage.similarity(["a"], []) == 0.0
    assert coverage.similarity(list("kitten"), list("sitting")) == pytest.approx(1 - 3 / 7)


def test_sign_test():
    assert coverage.sign_p(5, 5) == 1.0
    assert coverage.sign_p(10, 0) == pytest.approx(2 / 2 ** 10)


def test_odds_ratio_of_identical_strata_is_one():
    same = [(1, 2, True), (1, 2, False), (1, 1, True), (1, 1, False)] * 5
    assert coverage.odds_ratio(same)[0] == pytest.approx(1.0)


def test_moves():
    got = coverage.moves([(0, 1, True), (0, 2, False), (1, 1, True), (2, 0, False)])
    assert got["up"][:2] == (1, 2) and got["same"][:2] == (1, 1) and got["down"][:2] == (0, 1)


def test_same_statistics_as_the_old_modules():
    old_es = pytest.importorskip("aux.transfer_and_trajectories.explore_strat", reason="the old modules are gone after integration")
    old_dyn = pytest.importorskip("aux.headline_numbers.dynamics")
    old_fig = pytest.importorskip("aux.transfer_and_trajectories.gains_fig")
    for seed in range(5):
        t = triples(300, seed)
        strata = []
        for level in range(4):
            up = [a for r, o, a in t if r == level and o > r]
            same = [a for r, o, a in t if r == level and o == r]
            strata.append((sum(up), len(up) - sum(up), sum(same), len(same) - sum(same)))
        old = old_es.mh(strata)
        ratio, se = coverage.odds_ratio(t)
        assert ratio == pytest.approx(old["OR"])
        assert ratio * math.exp(1.96 * se) == pytest.approx(old["ci"][1])
    g = np.random.default_rng(0)
    for _ in range(50):
        a, b = list(g.integers(0, 4, g.integers(0, 12))), list(g.integers(0, 4, g.integers(0, 12)))
        assert coverage.similarity(a, b) == pytest.approx(old_dyn.editsim(a, b))
    for a, b in ((35, 12), (58, 65), (0, 0), (3, 9)):
        assert coverage.sign_p(a, b) == pytest.approx(old_fig.sign_p(a, b))


def test_contamination_helpers():
    assert contamination.names("see tau3-retail-12 first", "tau3-retail-12")
    assert not contamination.names("see tau3-retail-123", "tau3-retail-12")
    assert contamination.grams("a b c d".split(), 3) == {("a", "b", "c"), ("b", "c", "d")}
    assert contamination.VALUE.findall("run 12 then 1234") == ["1234"]
