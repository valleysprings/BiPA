# contamination_and_loading

The answer-leak check of the learned skills and on-demand skill loading.
`python -m aux.contamination_and_loading.cheatsheet` writes `mech_cheatsheet.md`: word n-gram overlap of every learned skill with the dev tasks' solutions and checkers, dev task ids it names, and the rule-level check of the coded rules (reads `skilltext.json` of `aux.rule_coding.skilltext`).
`python -m aux.contamination_and_loading.trigger_fig` draws `mech_trigger` and writes `mech_trigger_fig.md` from the recorded delivery and trigger runs (always resident against on demand; loading without an activation instruction).
`run_native` and `run_trigger` produced those runs (they call the model and need the SkillZip runtime, which is not part of this repository).
