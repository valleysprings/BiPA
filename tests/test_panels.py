"""Panels and the 160-slot training stream every optimizer reads."""
from collections import Counter

import pytest

from src.panels import batches, chunk_passes, panels, plan_passes, slice_stream

BENCHES = ("tau3-bench", "automationbench", "terminal-bench-2-1")
T = {"gepa": (20, 20, 20), "skillopt": (4, 4, 6), "skilladam": (16, 16, 16)}


@pytest.mark.parametrize("method", sorted(T))
def test_every_optimizer_reads_160_train_episodes(method):
    for bench, steps in zip(BENCHES, T[method]):
        planned = batches(bench, "main", method)
        assert len(planned) == steps and sum(map(len, planned)) == 160
        assert all(len(set(batch)) == len(batch) for batch in planned)


def test_panel_sizes():
    sizes = {bench: {split: len(ids) for split, ids in panels(bench).items()} for bench in BENCHES}
    assert sizes == {"tau3-bench": {"train": 160, "sel": 30, "test": 75},
                     "automationbench": {"train": 160, "sel": 30, "test": 120},
                     "terminal-bench-2-1": {"train": 62, "sel": 9, "test": 18}}
    assert all(len({t for b in batches(bench, "main", "gepa") for t in b}) == 160 for bench in BENCHES[:2])


def test_tb2_is_one_stream_shared_by_every_optimizer():
    streams = {m: [t for b in batches("terminal-bench-2-1", "main", m) for t in b] for m in ("gepa", "skillopt", "skilladam", "seed")}
    assert len({tuple(stream) for stream in streams.values()}) == 1
    stream = streams["gepa"]
    assert set(stream[:62]) == set(stream[62:124]) and len(set(stream[:62])) == 62 and len(set(stream[124:])) == 36
    assert Counter(Counter(stream).values()) == {3: 36, 2: 26}
    assert all(len(set(stream[i:i + 16])) == 16 for i in range(len(stream) - 15))
    assert [len(b) for b in batches("terminal-bench-2-1", "main", "skillopt")] == [40, 22, 40, 22, 18, 18]


def test_stream_functions():
    tasks = [f"t{i}" for i in range(10)]
    passes = plan_passes(tasks, 3, [1, 2, 3], 25, 4)
    assert [len(p) for p in passes] == [10, 10, 5] and plan_passes(tasks, 3, [4, 5, 6], 25, 4) != passes
    stream = sum(passes, [])
    assert all(len(set(stream[i:i + 4])) == 4 for i in range(len(stream) - 3))
    assert slice_stream(passes, 4) == [stream[i:i + 4] for i in range(0, 24, 4)]
    assert [len(b) for b in chunk_passes(passes, 4)] == [4, 4, 2, 4, 4, 2, 1, 2, 2]


def test_smoke_scale():
    assert {split: len(ids) for split, ids in panels("tau3-bench", "smoke").items()} == {"train": 4, "sel": 2, "test": 2}
    assert [len(b) for b in batches("tau3-bench", "smoke", "gepa")] == [2, 2]
