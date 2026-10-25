# contamination_and_loading

Answer-leak audit of the learned skills, the shortcut audit of convergent gains, on-demand skill loading, the trigger rate and the guidance-prefix ablation, with the delivery table and the trigger figure.
`python -m aux.contamination_and_loading.cheatsheet` writes `mech_cheatsheet.md`; `shortcut_load` then `shortcut_audit judge|summary` run the shortcut audit (judge calls the model; `spotcheck.json` holds the manual verdicts).
`delivery_check`, `trigger_rate` and `prefix_ablation` read the recorded delivery, trigger and ablation runs; `trigger_fig` draws `mech_trigger`.
`run_native` and `run_trigger` produced those runs (they call the model and need the SkillZip runtime, which is not part of this repository).
