import math

import numpy as np

from aux.tokens import accounting, attribution, cumulative
from aux.tokens.split import four_terms
from src.graph.episode import Episode


def episode(prefix, history):
    """An episode whose requests read `prefix` fixed tokens and `history` tokens of earlier turns."""
    req = np.array(prefix, float) + np.array(history, float)
    return Episode("t", True, len(req), req, ["history", "system"], np.column_stack([history, prefix]).astype(float), [])


def test_compaction_rebills_only_the_history():
    e = episode([10, 10, 10, 10], [0, 5, 12, 20])
    assert accounting.billed(e) == e.X
    assert accounting.billed(e, window=1) == 40 + 0 + 5 + 7 + 8  # each request keeps what the previous turn added
    assert accounting.billed(e, cap=6) == 40 + 0 + 5 + 6 + 6
    assert accounting.billed(e, window=10) == e.X


def test_four_terms_add_up_when_content_and_prefix_move_alone():
    seed = dict(R=10, N=10, X=10 * 100 + 45 * 20, F=100, h=20, sF=1000 / 1900)
    skill = dict(seed, h=30, X=10 * 100 + 45 * 30, sF=1000 / 2350)
    terms, dX = four_terms([(seed, skill)])
    assert abs(terms["more_actions"][0]) < 1e-12 and abs(terms["turn_splitting"][0]) < 1e-12 and abs(terms["skill_text"][0]) < 1e-12
    assert abs(terms["content"][0] - dX[0]) < 0.01


def test_shares_of_the_summed_increase():
    cell = dict(model="m", bench="b", tasks=["a", "b"], A=dict(more_actions=0.2, turn_splitting=0.1, content=-0.1, skill_text=0.1, total=0.3),
                terms={k: np.array([v, v]) for k, v in dict(more_actions=0.2, turn_splitting=0.1, content=-0.1, skill_text=0.1).items()},
                dX=np.array([0.3, 0.3]))
    s = attribution.shares([cell])
    assert math.isclose(s["share_turns"], 1.0) and math.isclose(s["share_text"], 1 / 3) and math.isclose(s["gross_turns"], 0.6)
    draws = attribution.draw_counts([cell])
    b = attribution.shares([cell], draws)
    assert np.allclose(b["share_turns"], 1.0) and b["p_total_le_0"] == 0.0  # every resample of identical tasks gives the same shares


def test_thousands_labels():
    assert [cumulative.kfmt(v) for v in (175.4, -21.2, 0.3, 1100)] == ["+175k", "−21k", "0", "+1.10M"]
    assert cumulative.kfmt(707, sign=False) == "707k" and cumulative.pfmt(-5.5) == "−6"
