# BiPA — Bipartite Attribution for Learned Skills

Code for the BiPA paper. BiPA verifies a learned skill by what it changes in the agent: it compares the skill with its seed over repeated runs, replays the optimizer's own selection with every candidate identical to the seed (a control that isolates the dev gain selection alone reports), and records each episode as a bipartite graph between turns and the context they read, splitting extra input tokens into four behaviors: more actions, the same actions spread over more turns, more content per turn, and longer skill text.

## Layout

| Path | Contents |
|---|---|
| `bipa/` | Experiment harness: scheduling (`run.sh` → `launch.py`), single-cell runs (`run.py`), repeated test evaluation (`repeat.py`, `repeat_transfer.py`), episode tracing (`native_protocol.py`, `native_trace.py`), fixed panels (`panels.py`), and adapters (`adapters/`) to the upstream optimizers |
| `bipa/optimizers/` | Canonical upstream sources: SkillOpt, SkillAdam, GEPA, EvoHarness |
| `scripts/analysis/` | Offline audits, run statistics, and the table/figure generators |
| `scripts/jobs/` | Transfer evaluation jobs |
| `assets/skills/seed/SKILL.md` | The common seed skill |

The adapters convert tasks, trajectories, and verifier scores into each upstream interface; the search, edit, acceptance, and return rules remain the upstream implementations. Adaptations relative to the original papers (benchmarks, budgets, starting skill, execution interface, caching, transport) are recorded in each adapter's `DEVIATIONS`.

## Requirements

- Python 3.12 with `harness-bench` (task execution, caching, retry, and raw traces) installed
- The dependencies of each upstream optimizer, see `bipa/optimizers/<name>/`

## Usage

From the repository root, everything previews by default (no model calls):

```bash
bash bipa/run.sh run tau3 <model> --arm skillopt      # one optimization cell
bash bipa/run.sh repeat tau3 <model> --arm skillopt   # repeated test evaluation
bash bipa/run.sh suite <model>                        # full suite for one model
```

Add `--execute` to actually run, or `--detach` to run the schedule in the background. `run.sh` sources an optional `host.env` in the repository root for host-specific settings (e.g. proxy); `BASELINE_PYTHON` selects the interpreter.

Raw execution artifacts and caches are not part of this repository.
