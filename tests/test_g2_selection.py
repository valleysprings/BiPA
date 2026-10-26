"""aux.selection: the kept seed dev executions carry neutral run labels, and the release-shift fit of the path test (the
marginal model the paper reports) finds no shift when the update reproduces the pinned runs' counts."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_kept_seed_dev_executions_are_labelled_neutrally():
    kept = json.loads((ROOT / "aux/data/seed_dev_extra.json").read_text())
    text = json.dumps(kept)
    assert "bach" not in text and "ours/" not in text
    assert len(kept["executions"]) == 59
    assert Counter(e[0] for e in kept["executions"]) == {"tau3-bench": 30, "automationbench": 29}
    for _, _, outcome, run, _ in kept["executions"]:
        assert outcome in (0, 1) and run in kept["runs"] and run in kept["runs"][run]


def test_release_shift_is_zero_when_nothing_changed():
    from aux.selection.pathtest import shift
    counts = {"dev": {f"t{i}": (i % 4, 3) for i in range(12)}, "test": {f"u{i}": ((i * 7) % 4, 3) for i in range(20)}}
    fit = shift(counts, counts, (0.6, 0.3))
    for (model, split), (_, d, points, lr, p) in fit.items():
        if model != "marginal":
            continue  # the paired model scores the counts against posteriors shrunk toward the prior mean
        assert abs(d) < 0.05 and lr < 0.01 and p > 0.9, (model, split)
        assert points is None or abs(points) < 1
