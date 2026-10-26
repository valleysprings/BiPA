# Does Your Generalist Skill Optimizer Really Incentivize Agentic Tasks? Rethinking Skill Optimization via Bipartite Attribution

This is the code for the paper. It trains three skill optimizers (GEPA, SkillOpt, SkillAdam) and evaluates them against Seed and No-skill controls on three benchmarks (Terminal-Bench 2.1, τ3-bench, AutomationBench) with two models (DeepSeek-V4.1-Flash, GPT-6-Luna). It regenerates every table and figure from the recorded runs.

## Motivation

Skill optimizers report only success on a small dev set. That score is noisy, keeping the best of several noisy comparisons inflates it, and it says nothing about what the skill makes the agent do or spend. The paper asks:

- **RQ1, success.** Do learned skills succeed more often than the seed, and do the reported dev gains hold on test?
- **RQ2, behavior and cost.** What do learned skills change in the agent's turns, tool calls and input tokens?
- **RQ3, the edits.** What in the edits produces these changes?

## How BiPA works

A harness without compaction re-sends the prefix and the whole conversation at every turn. With prefix $F$ and a history growing $h$ tokens per turn, an episode of $R$ turns reads

$$X = R\,F + \frac{R(R-1)}{2}\,h$$

input tokens, so input tokens grow faster than turns: the turn elasticity $e_R = 1 + s_H\,R/(R-1)$, with $s_H$ the history share of $X$, is always above 1 and approaches 2 in long episodes.

BiPA records each episode as a bipartite graph between its turns and a fixed set of context components (the skill text, the rest of the prompt, each tool's returned content), with two streams: what the model reads (input tokens) and what the harness executes (tool calls). Cost is compared with the seed task by task, as the mean paired log-ratio of input tokens and its ratio $\rho$:

$$L(S)=\frac{1}{|\mathcal I|}\sum_{i\in\mathcal I}\log\frac{\bar X_i(S)}{\bar X_i(S_0)},\qquad \rho=e^{L(S)}\quad(\rho>1:\ \text{a token tax})$$

With $m=N/R$ calls per turn, the change splits into four behaviors, one term each:

$$\Delta\log X \approx \underbrace{e_R\,\Delta\log N}_{\text{more actions}}\;\underbrace{-\,e_R\,\Delta\log m}_{\text{turn-splitting}}+\underbrace{s_H\,\Delta\log h}_{\text{content per turn}}+\underbrace{s_F\,\Delta\log F}_{\text{skill text}}$$

Success is judged separately: avg@3 test repetitions against the seed, and a replay of each optimizer's acceptance rule on candidates identical to the seed, which shows the dev gain selection alone reports.

## Layout

```
bipa/
├── src/          # experiments and main-text analyses (code map: src/README.md)
│   ├── run.py        # entry point: run, repeat, transfer, suite
│   ├── panels.py     # task panels and training batches
│   ├── executor.py   # a skill on a list of tasks -> per-task results, through harness-bench
│   ├── llm.py        # the optimizer's model client
│   ├── layout.py     # paths, all taken from environment variables
│   ├── adapters/     # GEPA, SkillOpt, SkillAdam: upstream loops on these batches, executor and model client
│   ├── analysis/     # Table 1, its statistics, the turn slopes, the four-term attribution; settings.py: what every analysis reads
│   └── graph/        # the per-episode bipartite graph
├── aux/          # every other figure and number, one folder per question (aux/README.md)
├── optimizers/   # vendored upstream sources at pinned commits (optimizers/README.md)
├── skills/       # the seed and every returned skill (skills/README.md)
├── configs/      # tasks.yaml: panels, batches, executor; experiments.yaml: the run of every cell (pins), the other runs the analyses read (aux_runs) and the caches
├── scripts/      # main.sh: the main experiment and Table 1; aux.sh: every other analysis (both source host.env)
└── tests/        # offline tests with a fake executor and model
```

The seed is `skills/seed/SKILL.md` (`verification-before-completion`, 580 words). Recorded runs live outside the repository, at `$BIPA_ASSETS/artifacts/execution/<model>/<method>/<benchmark>/<run>/`. A run folder holds its settings (panels, batches, cache root, gate), `summary.json`, `final/SKILL.md`, harness-bench's `results/evaluations/<split>/<version>-<panel>/evaluation.json`, and `round-evidence/` (the skill each optimizer round evaluated). Test repetitions go to `execution/<model>/test-reps/<benchmark>__<method>__<run>/`.

## Setup

Python 3.12, with this repository next to a `harness-bench` checkout. From the repository root:

```bash
pip install -e ../harness-bench -e optimizers/skillopt -e 'optimizers/skilladam[backend]' \
    numpy scipy scikit-learn matplotlib openai httpx python-dotenv pyyaml pytest
pip install --no-deps -e optimizers/gepa
cp host.env.example host.env    # then edit it
python -m pytest tests -q       # offline
```

Paths come from environment variables (`src/layout.py`), the optimizer's endpoint from `src/llm.py`:

| Variable | Default | What |
|---|---|---|
| `BIPA_ASSETS` | `./assets` | `artifacts/` (recorded runs) and `cache/` (episode and analysis caches) |
| `HARNESS_BENCH_ROOT` | `../harness-bench` | the harness-bench checkout: task registry, benchmark data, `.env` |
| `BIPA_OUT` | `./out` | analysis records in `data/mech/`, figures in `fig/mech/` |
| `BIPA_SEED_FILE` | `skills/seed/SKILL.md` | the seed, byte-identical to the one the runs used |
| `BIPA_LEGACY_ROOTS` | none | `:`-separated roots of the paths in older records; `layout.relocate` maps them into `$BIPA_ASSETS` and `optimizers/` |
| `OPENAI_API_KEY_2` | harness-bench's `.env` | the optimizer's API key |
| `OPENAI_BASE_URL_2` | harness-bench's `.env` | the optimizer's endpoint (`https` only) |
| `OPENAI_BASE_URL_2_OPTIMIZER` | unset | replaces the endpoint for optimizer calls only |
| `PYTHON` | `python` | the interpreter of `scripts/main.sh` and `scripts/aux.sh` |

The agent's own model calls are configured in harness-bench. Host settings such as proxies go in `host.env`, which both scripts source.

## Execution backend

BiPA does not run agents itself. Every episode (one task, one skill text, one repetition) goes through `src/executor.py`, which calls `harness_bench.eval.run` and reads back, per task, a verdict (true, false, or unknown for provider and infrastructure failures, which never count as failures), every model request with its exact provider input-token count, its tool calls, and a record that the skill text reached the model input. Unknown outcomes are re-run at most twice.

Our runs used harness-bench, a wrapper around Harbor that runs each episode in a container on a sandbox service. It is not released, so the commands that call the model (`--execute` of `run`, `repeat`, `transfer`, `suite`) do not work from a fresh clone. The analyses read only recorded episodes and need no executor, model or network.

To run elsewhere, pass an `executor` keyword to `harness_bench.eval.run` from `settings()` in `src/executor.py`: an object with `run(version, panel, split, output_dir, attempt)`, which returns the evaluation record, and `cached_episodes(version, rows, split)`. Harbor's `docker` switch is rejected for these benchmarks (AutomationBench needs a network-less grading container, Terminal-Bench 2.1 89 prebuilt images, τ3-bench a tool-server sidecar, a grading service and model routes for a user simulator and a judge). Episodes from another executor are a different measurement, so give them their own `--cache-root`.

## Reproduction

Run everything from the repository root. Every target of `scripts/main.sh` and `scripts/aux.sh` prints its commands and runs nothing; `--execute` runs them, the model calls included, and `--execute --offline` runs only the steps that read the recorded runs. Records go to `$BIPA_OUT/data/mech/`, figures to `$BIPA_OUT/fig/mech/`.

**Main experiment: `scripts/main.sh`.**

```bash
scripts/main.sh train deepseek-flash --execute                                  # every method and control on every benchmark, then three test repetitions
scripts/main.sh run skillopt tau3 gpt-6-luna --tag my-run --execute            # one cell
scripts/main.sh repeat skillopt tau3 deepseek-flash --execute                  # avg@3 test repetitions of one run
scripts/main.sh transfer skillopt tau3 deepseek-flash --to gpt-6-luna --execute  # a returned skill under another model or on another benchmark
scripts/main.sh table1 --execute                                                # Table 1, its statistics, the turn slopes, the attribution
```

Benchmarks are `tau3`, `autob`, `tb2`; methods are `seed`, `none` (No skill), `gepa`, `skillopt`, `skilladam`. Each cell reads the run pinned for it in `configs/experiments.yaml`; `--tag NAME` picks another. A started run continues only with `--resume`.

**Other analyses: `scripts/aux.sh`.** One target per folder of `aux/` ([aux/README.md](aux/README.md) lists what each reproduces), and `all`.

```bash
scripts/aux.sh all                                   # print every step
scripts/aux.sh all --execute --offline               # run every step that needs no model call
scripts/aux.sh selection --execute --offline         # one folder
```

With `--execute` a failing step ends its folder, the other folders still run, and the run exits non-zero if any step failed. The steps that call the model (the transfer evaluations in `transfer`, the path-test passes in `selection`) run only without `--offline`. The delivery and trigger runs behind `transfer/loading.py` used the SkillZip runtime, which is not in this repository; their commands are in that module's docstring.
