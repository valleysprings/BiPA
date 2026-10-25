# Code map

| Path | What it does |
|---|---|
| `run.py` | Entry point: `run` optimizes one cell and tests its final skill, `repeat` completes the avg@3 test repetitions, `transfer` tests a returned skill on another model or benchmark, `suite` runs and repeats every cell of a model |
| `panels.py` | Task panels and training batches from harness-bench's registry and `configs/tasks.yaml` |
| `executor.py` | A skill text on a list of tasks -> per-task rows, through `harness_bench.eval.run`; `record_round` writes `round-evidence/` |
| `llm.py` | The optimizer's model client: streamed chat completions, retries, prompt cap, call records under `<run>/llm/` |
| `layout.py` | Paths from `BIPA_ASSETS`, `HARNESS_BENCH_ROOT`, `BIPA_OUT`, `BIPA_SEED_FILE`, `BIPA_LEGACY_ROOTS`; run directories; `relocate` for paths in older records |
| `adapters/` | `gepa.py`, `skillopt.py`, `skilladam.py`: each upstream loop on our batches, executor and model client |
| `prompts/skilladam/` | SkillAdam's benchmark prompt files (vendor role, metric, failure and success types, iteration example) |
| `graph/episode.py` | The per-episode bipartite graph: turns, context components, tool calls |
| `analysis/` | `table1.py` and `stats.py` (Table 1 and its paired statistics), `slopes.py` and `attribution.py` (turn slopes and the four-term split on the episode graph), `episodes.py` and `common.py` (their shared readers); `python -m src.analysis` runs them in order. Every other figure and number is in `aux/` |
| `utils/dataloader/` | The pinned run of each cell (`configs/experiments.yaml`) and evaluation readers for the analyses; analysis cache under `$BIPA_ASSETS/cache/analysis` |
| `utils/logger/` | Analysis records under `$BIPA_OUT/data/mech` |
| `utils/fig/` | Figures under `$BIPA_OUT/fig/mech` |
