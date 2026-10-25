# rule_coding

The edge table, the rule coding of the returned skills and its reliability, and the release copies of the skills.
`python -m aux.rule_coding.edges build` writes `edges_table.json` (parent-to-child edits of every optimizer run); then `python -m aux.rule_coding.skilltext compute|record|luna` codes the added text into eight rule categories (`skilltext.json`, `mech_skilltext.md`; `luna` writes `rules_luna.json` and `mech_rules_luna.md` for GPT-6-Luna). `compute` reads `attr.json` from `python -m src.analysis.attribution attr` (a step of `scripts/main.sh table1`).
`python -m aux.rule_coding.coder2` compares the second coder's labels with the final labels (`mech_coder2.md`); no step writes those labels, so they are kept in `aux/data/coder2_labels.json`; `python -m aux.rule_coding.release_skills` writes the seed and the returned skills with a README table to `$BIPA_OUT/skills/`.
