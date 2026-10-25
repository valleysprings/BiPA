# edit-gate-and-design

Kept vs rejected edits and the Holm-adjusted rule-category contrasts with the mech_edges figure (`edges_contrast`), the evidence behind the four design principles and the implications-figure inputs (`design`: batching, designev, selrules, draw), and the tie-break-by-cost replay (`tiebreak`, then `tiebreak_record`).
Run from the repo root, e.g. `python -m aux.edit_gate_and_design.edges_contrast [holm|draw]`, `python -m aux.edit_gate_and_design.design batching|designev|selrules|draw`, `python -m aux.edit_gate_and_design.tiebreak`. All read the edge table of `python -m src.analysis.edges build`.
