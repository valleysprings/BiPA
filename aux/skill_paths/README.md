# skill_paths

Per-run optimization paths (every candidate's text, gate decision, dev score and cost) for GEPA, SkillOpt and SkillAdam, the shared-vs-unique analysis of the added text, independent re-derivations (`verify/`) and the seed-path comparison (`compare/`). Records `mech_paths.md` and `mech_shared.md`.
Outputs go to `$BIPA_OUT/skill_paths/`. Order: `python -m aux.skill_paths.extract_gepa` (and `extract_skillopt`, `extract_skilladam`), then `prep_inputs`, `analyze_paths`, `analyze_shared`; checks: `python -m aux.skill_paths.verify.v1_runs_scores` ... `v13_firststep`, `python -m aux.skill_paths.verify.indep_check.cmp_gepa <run dir> <path json>`.
Comparison: `compare.prep_sentences`, `compare.build_app_vocab`, `compare.objective`; `compare.analyze_levels` also needs the two judge outputs `judge_J1.jsonl`, `judge_J2.jsonl` in `$BIPA_OUT/skill_paths/compare/`.
The extractors and analyses also read the near-empty seed of the seed-path runs, `$BIPA_ASSETS/skills/seed-minimal/SKILL.md`.
`soft_labels.json` holds the hand labels that set the paraphrase thresholds of `analyze_shared`.
