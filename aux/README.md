# Auxiliary analyses

The extractors and analyses here read the RECORDED runs under `$BIPA_ASSETS/artifacts/execution`, which were written before the current runtime in `src/`. Runs written by the current runtime carry fewer provenance fields (no per-round UTC stamps, skill or evaluation hashes, completion audits, resume logs or `llm/attempts.jsonl`), so scripts that read those fields work on the recorded runs only; `cost_appendix.optcost` prints n/a for the failed-attempt count when `llm/attempts.jsonl` is absent.

Each folder holds the scripts behind one group of figures, tables and numbers, with a README of its commands. Run them from the repository root as `python -m aux.<folder>.<script>`. They import the main-text code in `src/`; nothing in `src/` imports them.

| Folder | Topic |
|---|---|
| [calls_and_writes](calls_and_writes/README.md) | Tool calls per turn, read and write calls, serialized writes |
| [contamination_and_loading](contamination_and_loading/README.md) | Answer-leak and shortcut audits, on-demand skill loading, trigger rate, guidance-prefix ablation |
| [cost_appendix](cost_appendix/README.md) | Cost under cache pricing, compacted history, optimization cost, slope tables |
| [edit_gate_and_design](edit_gate_and_design/README.md) | Kept against rejected edits, design-principle evidence, tie-break by cost |
| [noise_and_domains](noise_and_domains/README.md) | Run-to-run noise, checks of the replay's task model, per-domain success |
| [path_test](path_test/README.md) | Test runs along the optimization paths on an updated model release |
| [rule_coding](rule_coding/README.md) | Rule-coding figures, second-coder agreement, release copies of the skills |
| [selection_replay](selection_replay/README.md) | Selection-rule replays: GPT-6-Luna, dev-set size, dev/test flips |
| [skill_paths](skill_paths/README.md) | Every candidate along each optimization path, shared and unique added text |
| [three_word_seed](three_word_seed/README.md) | Runs started from a three-word seed |
| [token_attribution](token_attribution/README.md) | Four-term split under GPT-6-Luna, cost of failed tasks, bipartite figures |
| [transfer_and_trajectories](transfer_and_trajectories/README.md) | Transfer across models and benchmarks, trajectory dynamics, convergence of gains |

`data/edges_frozen_rows.json` holds edge rows that `python -m src.analysis.edges build` uses for panels whose request records are missing.
