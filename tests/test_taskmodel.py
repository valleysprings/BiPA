"""aux.selection.taskmodel is the old selection library with an explicit generator: the same draws give the same numbers,
and the seed's executions read through the aux_runs registry are the ones the old date and tag filters picked."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src import layout
from src.analysis import settings as S
from src.utils import dataloader as C

ROOT = Path(__file__).resolve().parents[1]
recorded = pytest.mark.skipif(not (layout.EXECUTION / "gpt-6-luna/skilladam/tau3-bench").is_dir(),
                              reason="needs the recorded runs under BIPA_ASSETS")
old_library = pytest.importorskip("aux.selection_replay.selection", reason="the old library is gone after integration")
from aux.selection import taskmodel as TM  # noqa: E402

OLD_LUNA = {m: {b: "native-20261012" for b in C.BENCHES} for m in S.OPTIMIZERS}


def same_draws(old_call, new_call, seed=7):
    old_library.RNG = np.random.default_rng(seed)
    a = old_call()
    b = new_call(np.random.default_rng(seed))
    return a, b


def equal(x, y):
    if isinstance(x, (tuple, list)):
        return len(x) == len(y) and all(equal(u, v) for u, v in zip(x, y))
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
    return np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y or (x != x and y != y)  # nan: no gain found


def test_rules_draw_exactly_as_the_old_library():
    from aux.edit_gate_and_design import design
    from aux.selection_replay import luna
    P = old_library
    dev = {f"t{i}": [i % 2, 1, (i // 2) % 2] for i in range(12)}
    prior = (1.3, 0.9)
    a, b = same_draws(lambda: P.task_probabilities(dev, prior, 9, 500), lambda r: TM.task_probabilities(dev, prior, 9, 500, r))
    assert np.array_equal(a, b)
    a, b = same_draws(lambda: P.task_probabilities(dev, prior, 15, 500), lambda r: TM.task_probabilities(dev, prior, 15, 500, r))
    assert np.array_equal(a, b)
    p = b
    for old, new in [(lambda: P.outcomes(p), lambda r: TM.outcomes(p, r)),
                     (lambda: P.best_of(p, 4, 7), lambda r: TM.best_of(p, 4, 7, r)),
                     (lambda: P.paired(p, P.raised(p, 2)), lambda r: TM.paired(p, TM.raised(p, 2), r)),
                     (lambda: P.critical_value(p), lambda r: TM.critical_value(p, r)),
                     (lambda: P.mde80(p), lambda r: TM.mde80(p, r)),
                     (lambda: P.batch_gain_tail(p, 5, 1), lambda r: TM.batch_gain_tail(p, 5, 1, r)),
                     (lambda: P.skilladam_accepts(p, 5, [2, 3, 1]), lambda r: TM.skilladam_accepts(p, 5, [2, 3, 1], r)),
                     (lambda: design.acceptance(p, P.raised(p, 2), 3), lambda r: TM.acceptance(p, TM.raised(p, 2), 3, r))]:
        x, y = same_draws(old, new)
        assert equal(x, y), (x, y)
    sa = {"batch": 5, "budgets": [2, 3], "reported_gain_points": 20.0}
    assert P.skilladam_null(p, sa, np.random.default_rng(3)) == TM.skilladam_null(p, sa, np.random.default_rng(3))
    gain = TM.best_of(p, 4, 7, np.random.default_rng(1))
    assert P.replay_power(p, 4, 7, gain, np.random.default_rng(5)) == TM.replay_power(p, 4, 7, gain, np.random.default_rng(5))
    tasks = sorted(dev)
    for old_kind, kind in (("null plug-in", "plug-in"), ("null Bayesian bootstrap", "bayes-boot")):
        assert np.array_equal(luna.null_probabilities(old_kind, dev, tasks, 300, np.random.default_rng(2)),
                              TM.prior_free(kind, dev, tasks, 300, np.random.default_rng(2)))
    assert luna.rank_test(5, [4, 6, 5], 1) == TM.rank_test(5, [4, 6, 5], 1)
    assert P.fit_prior([(1, 3), (2, 3), (0, 3)]) == TM.fit_prior([(1, 3), (2, 3), (0, 3)])


@recorded
def test_seed_executions_through_the_registry(monkeypatch):
    from aux.selection_replay import luna
    registry = yaml.safe_load((ROOT / ".scratch/repin-live/parity/aux_runs.yaml").read_text())["aux_runs"]
    monkeypatch.setattr(layout, "AUX_RUNS", registry)
    monkeypatch.setitem(C.PINS, "gpt-6-luna", {**C.PINS["gpt-6-luna"], **OLD_LUNA})
    for bench in C.BENCHES:
        dev, test, files = old_library.seed_executions(bench)
        new_dev, new_test, new_files = TM.seed_executions(S.DEEPSEEK, bench)
        assert (new_dev, new_test, new_files) == (dev, test, files)
        assert TM.dev_draws(S.DEEPSEEK, bench, new_files, len(dev)) == old_library.dev_draws(bench, files, len(dev))
        obs, new_obs = old_library.observed(bench, len(dev)), TM.observed(S.DEEPSEEK, bench, len(dev))
        assert all(new_obs[r][k] == obs[r][k] for r in ("SkillOpt", "GEPA") for k in ("K", "baseline", "gain"))
        assert new_obs["SkillAdam"] == obs["SkillAdam"]
        l_dev_files, l_test_files = luna.luna_seed_files(bench)
        l_dev, l_test, _ = TM.seed_executions(S.LUNA, bench)
        per_task = lambda e: {t: sorted(v) for t, v in e.items()}  # the same executions; file order differs
        assert (per_task(l_dev), per_task(l_test)) == (per_task(old_library.executions(l_dev_files)),
                                                       per_task(old_library.executions(l_test_files)))
        assert per_task(TM.null_executions(S.LUNA, bench)) == per_task(luna.luna_null_executions(bench))
        seed_solved = TM.dev_draws(S.LUNA, bench, TM.dev_files(S.LUNA, bench), len(l_dev))[0]["solved"]
        old_l, new_l = luna.luna_observed(bench, len(l_dev), seed_solved), TM.observed(S.LUNA, bench, len(l_dev), seed_solved)
        assert all(new_l[r][k] == old_l[r][k] for r in ("SkillOpt", "GEPA") for k in ("K", "baseline", "gain", "candidates"))
        assert new_l["SkillAdam"] == old_l["SkillAdam"]
