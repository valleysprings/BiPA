# edit_gate_and_design

Kept against rejected edits and the evidence behind the design principles. All read the edge table of `python -m aux.rule_coding.edges build`.
`python -m aux.edit_gate_and_design.edges_contrast [holm]` writes `mech_edges.md` (kept and rejected edits on the four-term split and on dev success) or `mech_edges_holm.md` (the rule-category contrasts under Holm's correction).
`python -m aux.edit_gate_and_design.design batching|designev|selrules` writes `mech_batching.md` (cost at the seed's calls per turn), `mech_designev.md` (what explains an edit's token change, the cost-aware gate) and `mech_selrules.md` (four acceptance rules on the task model); `designev` reads `compaction.json` of `aux.cost_appendix.compaction compute`.
`python -m aux.edit_gate_and_design.tiebreak`, then `tiebreak_record`, writes `mech_tiebreak.md` (whether success or cost separates a candidate from its parent, and the cost tie-break replayed on the optimizers' candidates).
