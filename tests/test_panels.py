"""Panel composition: the LODO scale drops the held-out domain before sampling; main is unchanged."""
from src.core.panels import batches, panels


def test_lodo_banking_excludes_domain_everywhere_but_test():
    sample = panels("tau3-bench", "lodo-banking", ())
    assert not any(t.startswith("tau3-banking_knowledge-") for t in sample["train"] + sample["sel"])
    assert sum(t.startswith("tau3-banking_knowledge-") for t in sample["test"]) == 19
    assert len(sample["train"]) == 160 and len(sample["test"]) == 75
    # the dev pool without banking_knowledge has 27 tasks, below the sel cap of 30
    assert len(sample["sel"]) == 27


def test_lodo_banking_keeps_main_budget_and_batches():
    planned = batches("tau3-bench", "lodo-banking", "skillopt", ())
    assert sum(map(len, planned)) == 160
    assert all(not t.startswith("tau3-banking_knowledge-") for batch in planned for t in batch)


def test_main_scale_unaffected_by_lodo():
    sample = panels("tau3-bench", "main", ())
    assert any(t.startswith("tau3-banking_knowledge-") for t in sample["train"])
    assert any(t.startswith("tau3-banking_knowledge-") for t in sample["sel"])
    assert len(sample["test"]) == 75 and len(sample["sel"]) == 30
