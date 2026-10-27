# Auxiliary analyses

Every figure and number of the paper outside Table 1, one folder per question. Each module reads the recorded runs through `src/analysis/settings.py` (the pinned run of every cell, the learned skills, the test episodes, resampling), draws with the one figure style of `src/utils/fig.py` and writes its record with `record()` in `src/utils/logger.py`. Run a module as `python -m aux.<folder>.<module>` with any stage argument printed by the scripts, or a whole folder with `scripts/aux.sh <folder>`; `scripts/aux.sh all` runs `scripts/main.sh table1` first, then the folders in the order below.

| Folder | What it answers | Records and figures |
|---|---|---|
| `tokens` | Where the extra input tokens come from and what they cost: the four-term split under both models, within-task turn slopes, turn elasticity and prefix shares, the cost of failed tasks, cached pricing, a compacting harness, optimization cost, the teaser and the per-turn cumulative tokens | mech_slopes, mech_attribution3, mech_attribution_luna, mech_failcost, mech_price3, mech_compaction, mech_optcost, mech_bipcum, mech_teaser; mech_slopes_input2, teaser2, method, bipcum3, bipcum_main2 |
| `calls` | What the skills change in tool calls per turn, and the serialized writes | mech_callsperturn, mech_calls_rw, mech_batching; bipwrites3 |
| `edits` | What the editors write and what the gate keeps: edit lineage, rule categories, kept against rejected edits, acceptance rules, the cost tie-break, the released skills | mech_skilltext, mech_rules_luna, mech_coder2, mech_edges, mech_edges_holm, mech_designev, mech_selrules, mech_tiebreak; implications4, skills2 |
| `success` | Learned skills against the seed: Table 1's paired statistics, pooled success and input-token ratio, run-to-run spread, per-domain success | mech_stats, mech_pooled, mech_reruns, mech_domains; reruns3, domains_wide |
| `selection` | The dev gain that selection alone reports (replay of each dev-gated run), dev-set size, task-model checks, dev against test, the path test | mech_selection, mech_selection_luna, mech_devsize, mech_assumptions, mech_devdyn, mech_priorfit, mech_pathtest; dev2test4, selection, devpertask, priorfit, pathtest |
| `transfer` | The skills under another model or on another benchmark, coverage and agreement, the contamination check, skill loading | mech_transfer, mech_ood, mech_gains, mech_cheatsheet, mech_trigger; transfer3, gains2, trigger |
| `three_word_seed` | The runs started from the three-word seed | mech_fromscratch; scratch2 |

The analysis-specific `data/` directories hold versioned hand-curated/frozen analysis inputs, not run-generated artifacts:

- `edits/data/`: `rule_labels.json` gives the category of every added rule unit, `coder2_labels.json` holds the second coder's labels, and `edges_frozen_rows.json` preserves the edge rows of panels whose request records are missing.
- `selection/data/`: `seed_dev_extra.json` holds seed dev executions of earlier runs, pooled by the selection task model.
- `three_word_seed/data/`: `sentence_labels.json` holds the three-word-seed sentence labels.

The steps that call the model (the transfer evaluations and the path-test passes) run only with `scripts/aux.sh ... --execute` and without `--offline`. The delivery and trigger runs of `transfer/loading.py` used the SkillZip runtime, which is not in this repository; their commands are in that module's docstring.
