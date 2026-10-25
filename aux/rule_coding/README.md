# rule_coding

Figures and reliability of the rule coding of `src/analysis/skilltext.py`: the rules and skills figures, the second-coder agreement (`mech_coder2.md`) and the release copies of the returned skills.
`python -m aux.rule_coding.figures rules|skills|review` draws `mech_rules` / `mech_skills` from `skilltext.json` and `attribution.json`, or prints every coded unit.
`python -m aux.rule_coding.coder2` compares `coder2_labels.json` with the final labels; `python -m aux.rule_coding.release_skills` writes the seed and the returned skills with a README table to `$BIPA_OUT/skills/`.
