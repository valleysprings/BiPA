# headline_numbers

Numbers quoted next to Table 1: learned skills against the seed pooled over all cells, task coverage, and the skills tested unchanged on another model or benchmark.
`dynamics` is the shared reader: `load_cell(model, arm, bench)` gives the per-episode fields of the avg@3 test runs of both models, which `pooled`, `transfer` and the `transfer_and_trajectories` scripts read.
`python -m aux.headline_numbers.pooled [pooled|gains]` writes `pooled.json` (pooled success difference and token ratio), and `pass3_ci.json` and `gains_tb2.json` (pass@3 coverage) under `$BIPA_ASSETS/cache/analysis/main/`; with no argument it runs both.
`python -m aux.headline_numbers.transfer [luna|ood]` writes `luna_transfer.json` (GPT-6-Luna avg@3 cells on tau3-bench) and prints the DeepSeek cross-benchmark runs (Terminal-Bench 2.1 and AutomationBench); with no argument it runs both.
