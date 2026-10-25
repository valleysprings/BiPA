# Auxiliary analyses

The extractors and analyses here read the RECORDED runs under `$BIPA_ASSETS/artifacts/execution`, which were written before the current runtime in `src/`. Runs written by the current runtime carry fewer provenance fields (no per-round UTC stamps, skill or evaluation hashes, completion audits, resume logs or `llm/attempts.jsonl`), so scripts that read those fields work on the recorded runs only; `cost_appendix.optcost` prints n/a for the failed-attempt count when `llm/attempts.jsonl` is absent.

Each folder holds the scripts behind one group of the paper's figures and numbers, with a README of its commands. Run them from the repository root as `python -m aux.<folder>.<script>`, after `python -m src.analysis` (Table 1 and the attribution caches they read). They import the experiment and analysis code in `src/`; nothing in `src/` imports them. There are no tests for `aux/`.

`scripts/aux.sh FOLDER` takes one folder's steps in its README's order; `scripts/aux.sh all` takes the steps of `scripts/main.sh table1` (`python -m src.analysis`), then the folders in the order of this table, since a folder reads the caches of the folders above it. What its flags do is in the [README](../README.md#reproduction).

| Folder | What it supports in the paper |
|---|---|
| [rule_coding](rule_coding/README.md) | The edge table, the rule coding of the added text, the second-coder agreement and the released skills |
| [cost_appendix](cost_appendix/README.md) | Cost under cache pricing, re-billing under a compacted history, optimization cost and the editor prompt lengths |
| [token_attribution](token_attribution/README.md) | The four-term split (DeepSeek record, GPT-6-Luna version), the cost of failed tasks and the data of the cumulative-tokens figure |
| [selection_replay](selection_replay/README.md) | The selection replay of both models, the dev/test figure, the selection figure, the dev-set size, the task-by-task flips and the fitted prior |
| [edit_gate_and_design](edit_gate_and_design/README.md) | Kept against rejected edits, the acceptance rules and the cost tie-break: the numbers of the design principles and their figure |
| [calls_and_writes](calls_and_writes/README.md) | Tool calls per turn and the serialized writes: the writes sharing a turn, the packing counterfactual, the data of the serialized-writes figure |
| [noise_and_domains](noise_and_domains/README.md) | Run-to-run noise (the rerun figure), the task-model checks quoted with the selection replay, and per-domain success and its figure |
| [headline_numbers](headline_numbers/README.md) | The transfer evaluations (they call the model); pooled success and token ratio over all skills, task coverage, and the skills tested on another model or benchmark |
| [transfer_and_trajectories](transfer_and_trajectories/README.md) | The transfer figure and the gains record and figure (coverage and agreement of a task's runs) |
| [three_word_seed](three_word_seed/README.md) | The runs started from a three-word seed and their figure |
| [figures](figures/README.md) | The teaser figure, the method figure, the main-text figure of input tokens read per turn by prompt component, and the figures of the learned skills' split and rules, their per-component token gaps and the serialized writes |
| [contamination_and_loading](contamination_and_loading/README.md) | The contamination check of the learned skills and the skill-loading figure (always resident against on demand, loading without an instruction) |
| [path_test](path_test/README.md) | The path test along the tau3 optimization paths on the updated release (its passes call the model), and the release shift |

`data/edges_frozen_rows.json` holds edge rows that `python -m aux.rule_coding.edges build` uses for panels whose request records are missing.
`data/coder2_labels.json` holds the second coder's label of every added rule unit, which `python -m aux.rule_coding.coder2` compares with the final labels.
`data/seed_dev_extra.json` holds the seed's dev outcomes and input tokens of runs the assets no longer hold (one per tau3 and AutoBench dev task, with the runs they came from), which the selection task model (`aux.selection_replay.selection.seed_executions`) and the seed-vs-seed token noise floor (`aux.edit_gate_and_design.tiebreak`) add to the recorded ones when those runs are absent.
