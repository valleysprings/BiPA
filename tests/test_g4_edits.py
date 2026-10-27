"""aux.edits: SkillOpt parents follow the skill each step started from, and every rule unit of a learned skill has a label."""
import json

import pytest

from src.utils import layout
from src.analysis import settings as S

lineage = pytest.importorskip("aux.edits.lineage")
rules = pytest.importorskip("aux.edits.rules")
RECORDED = (layout.EXECUTION / "gpt-6-luna/skillopt/terminal-bench-2-1").is_dir()
recorded = pytest.mark.skipif(not RECORDED, reason="needs the recorded runs under BIPA_ASSETS")


@recorded
def test_skillopt_parent_is_the_skill_the_step_started_from():
    for model in (S.DEEPSEEK, S.LUNA):
        for bench in lineage.BENCHES:
            run = S.run(model, bench, "skillopt")
            nodes, edges = lineage.skillopt_edges(run)
            history = json.loads((run / "skillopt/history.json").read_text())
            for step, edge in zip(history, edges):
                started = (run / f"skillopt/skills/skill_v{step['step'] - 1:04d}.md").read_text()
                assert nodes[edge["parent"]] == started
                single_epoch = all(h.get("epoch", 1) == 1 for h in history)
                if single_epoch:  # one pass: the parent is the seed or the last accepted step
                    assert edge["parent"] == "seed" or edge["parent"].startswith("s")


@recorded
def test_every_unit_of_a_learned_skill_has_a_label():
    labels = json.loads(rules.LABELS.read_text())
    for model, bench, method in S.learned():
        coded = rules.skill(model, bench, method, labels[model])
        assert coded["units"] and all(u["category"] in rules.CATS for u in coded["units"])


def test_proposal_fixes():
    assert rules.propose("Perform every consequential write once") != "serialize"
    assert rules.propose("Parallelize only read-only discovery") == "parallelize"
    assert rules.propose("Run the steps sequentially, one at a time") == "serialize"
