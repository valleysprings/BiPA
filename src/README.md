# Code map

| Path | What it does |
|---|---|
| `run.py` | Entry point: `run` optimizes one cell and tests its final skill, `repeat` completes the avg@3 test repetitions, `transfer` tests a returned skill on another model or benchmark, `suite` runs and repeats every cell of a model |
| `panels.py` | Task panels and training batches from harness-bench's registry and `configs/tasks.yaml` |
| `utils/executor.py` | A skill text on a list of tasks -> per-task rows, through `harness_bench.eval.run`; `record_round` writes `round-evidence/` |
| `utils/llm.py` | The optimizer's model client: streamed chat completions, retries, prompt cap, call records under `<run>/llm/` |
| `utils/layout.py` | Paths from `BIPA_ASSETS`, `HARNESS_BENCH_ROOT`, `BIPA_OUT`, `BIPA_SEED_FILE`, `BIPA_LEGACY_ROOTS`; the pins of `configs/experiments.yaml`; run directories; `relocate` for paths in older records |
| `utils/skill_frame.py` | Namespaced skill-frame matching, including the optional `path` attribute |
| `adapters/` | `gepa.py`, `skillopt.py`, `skilladam.py`: each upstream loop on our batches, executor and model client |
| `prompts/skilladam/` | SkillAdam's benchmark prompt files (vendor role, metric, failure and success types, iteration example) |
| `graph/episode.py` | The per-episode bipartite graph: turns, context components, tool calls |
| `graph/metrics.py` | Per-episode metrics and within-task slope fits shared by the analyses |
| `analysis/` | `settings.py` (what every analysis reads: the pinned run of each cell, the learned skills, the test episodes, resampling), `table1.py` (Table 1 for both models, including `rho_turns`) and `common.py` (shared test-repetition readers and per-task cost ratios); run Table 1 as `python -m src.analysis.table1`. Paired statistics, turn slopes and attribution are in `aux/` |
| `utils/dataloader.py` | Evaluation readers for the analyses; analysis cache under `$BIPA_ASSETS/cache/analysis` (or `BIPA_ANALYSIS_CACHE`) |
| `utils/logger.py` | `record()`: the one format of every analysis record, under `$BIPA_OUT/data/mech` |
| `utils/fig.py` | The one figure style (fonts, sizes, colours) and `save_figure`, under `$BIPA_OUT/fig/mech` |

Versioned hand-curated/frozen inputs for auxiliary analyses are archived by function under `aux/<analysis>/data/`; see `aux/README.md`.
