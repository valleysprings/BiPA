# Does Your Generalist Skill Optimizer Really Incentivize Agentic Tasks? Rethinking Skill Optimization via Bipartite Attribution

Code of the paper: it trains and evaluates the three skill optimizers the paper studies (GEPA, SkillOpt, SkillAdam) plus Seed and No-skill controls on three benchmarks (Terminal-Bench 2.1, τ3-bench, AutomationBench) with two models (DeepSeek-V4.1-Flash, GPT-6-Luna), and reproduces every table and figure from the recorded experiments.

## Motivation

Skill optimizers report success alone, and a success score on a small dev set cannot settle whether a learned skill fits its task family better:

- **Evaluation noise.** A single test evaluation can rank two skills either way; a paired comparison of three repetitions detects only differences of 7.1 / 5.2 / 7.3 points on τ3-bench, AutoBench and TB2.
- **Selection inflates gains.** Keeping the best of several noisy dev comparisons is a winner's curse: replaying each optimizer's own acceptance rule on candidates identical to the seed reports 1.2 to 5.1 dev tasks of expected gain against the observed 1 to 5.
- **Success hides behavior and cost.** A skill acts through the agent's turns and what they read; whether a gain survives repeated evaluation, and what the skill makes the agent do and spend, go unchecked.

## Findings

- **The success gains do not hold (RQ1).** Dev gains of 3 to 40 points end at test differences of −11.1 to +2.7 points over three repetitions, eight of the nine negative (Fig. 5).
- **Every learned skill levies a token tax (RQ2).** All nine DeepSeek learned skills read 1.08–1.43× the seed's input tokens per task and take 1.03–1.22× its turns, on 59–87% of tasks per skill (Table 1, Fig. 4). Extra turns carry 73% of the summed log-token increase, the skill text 21%; within a task, input tokens rise with turns at slopes 1.35–1.58 while outputs and tool calls rise at ≤1.03 (Fig. 3, 6a).
- **The optimizers accumulate procedure (RQ3).** 58 of 59 traceable edits lengthen the skill (654–2,447 words against the seed's 580); of 337 added rules, 52% verify and none asks the agent to read less; kept and rejected edits cost alike — the cost arrives with the proposals (Fig. 6b, 7d).
- **The findings generalize (RQ4).** On GPT-6-Luna the pattern repeats (on TB2 all three optimizers return the seed unchanged); frozen skills carry their tax to an unseen benchmark (five of six transfers read 1.23–1.47× the seed's tokens) and to the second model (Table 1 lower panel, Tables 2–3, Fig. 16).

## Implications for skill optimizer design

- **Keep an edit only when its gain exceeds selection noise.** Keeping whatever wins one dev comparison keeps a candidate identical to its parent 24–42% of the time; confirming the win on a fresh draw cuts this to 6–17% while keeping 55–94% of edits that truly gain.
- **Charge each edit for the extra turns it causes.** Skill length explains 10% of the variance in an edit's token change against 93% for turns, so compressing text saves at most the skill-text term of the split; objectives that charge the extra turns save more.
- **Let the editor remove rules and write cheaper ones.** 58 of 59 traced edits only append; re-billing recorded episodes at the seed's calls per turn halves the tax — issue independent calls in one turn, and return from each call only what the next turn needs.
- **Show the editor what its earlier rules cost.** 44 of 59 edits raise input tokens whether kept or rejected, and only 5 keep dev success at no extra tokens — the cost arrives with the proposals, so it has to reach the editor that writes them.

## How BiPA works

A harness without compaction re-sends the prefix and the whole conversation at every turn. With prefix $F$ and a history growing $h$ tokens per turn, an episode of $R$ turns reads

$$X = R\,F + \frac{R(R-1)}{2}\,h$$

input tokens — each extra turn re-reads the whole history, so input tokens grow faster than turns: the turn elasticity $e_R = 1 + s_H\,R/(R-1)$ ($s_H$ the history share of $X$) is always above 1 and approaches 2 in long episodes.

BiPA records each episode as a bipartite graph between its turns and a fixed set of context components (skill text, rest of the prompt, each tool's returned content), with two streams: what the model reads (input tokens) and what the harness executes (tool calls). Cost is judged against the seed task by task, as the mean paired log-ratio of input tokens and its ratio $\rho$:

$$L(S)=\frac{1}{|\mathcal I|}\sum_{i\in\mathcal I}\log\frac{\bar X_i(S)}{\bar X_i(S_0)},\qquad \rho=e^{L(S)}\quad(\rho>1:\ \text{a token tax})$$

and with $m=N/R$ calls per turn, the change splits into four behaviors, one term each:

$$\Delta\log X \approx \underbrace{e_R\,\Delta\log N}_{\text{more actions}}\;\underbrace{-\,e_R\,\Delta\log m}_{\text{turn-splitting}}+\underbrace{s_H\,\Delta\log h}_{\text{content per turn}}+\underbrace{s_F\,\Delta\log F}_{\text{skill text}}$$

Success is judged separately: avg@k test repetitions against the seed, plus a seed-replay control — each dev task's success probability is drawn from its Beta posterior $p_v\,|\,\text{seed}\sim\mathrm{Beta}(a+s_v,\;b+n_v-s_v)$ fitted to the seed's executions, candidates identical to the seed go through the optimizer's own acceptance rule, and the gain $G_0$ this keeps is what selection alone reports.

## Layout

```
bipa/
├── src/
│   ├── run.py        # one optimization cell (the external entry)
│   ├── core/         # launch, repeat, panels, traces, layout, config
│   ├── adapters/     # per-optimizer interfaces (skillopt, skilladam, gepa)
│   ├── analysis/     # main.py (Table 1); rq1-rq4/ (per-question analyses); mainfigs/ (main-text figures);
│   │                 # appendix/ (appendix + Section-3 numbers); submission/ (frozen manifest + audit)
│   ├── dynamics/     # transfer drivers, execution-evidence audits, status/refresh
│   ├── graph/        # per-episode bipartite-graph metrics
│   ├── utils/        # executor, llm, fig/, dataloader/, logger/ (audit records)
│   └── prompts/      # skilladam optimizer prompt fragments (vendor role, metric, failure/success types, iteration example)
├── scripts/          # baselines.sh (train + repeat) and analysis.sh (everything else)
├── optimizers/       # vendored upstream sources, pinned commits
├── configs/          # tasks / experiments / graph yaml
├── tests/            # 66 tests, fake model and tasks, no API calls
└── docs/             # investigation notes
```

The seed skill (`verification-before-completion`, 580 words) is read from the shared artifacts tree at `../assets/skills/seed/SKILL.md`. Recorded episodes live outside the repo at `../assets/artifacts/execution/<model>/<method>/<bench>/<run-id>/`; the offline steps only read them.

## Setup

Python 3.12 with `harness-bench`; the upstream optimizers' dependencies (`pip install -e` the pinned checkouts, see [optimizers/README.md](optimizers/README.md)). Interpreter via `BASELINE_PYTHON` (defaults to the harness-mitigation `bach-harness/.venv`; only the Python environment is reused, no BACH code is imported). `baselines.sh` sources an optional `host.env` for host settings. Tests: `"$BASELINE_PYTHON" -m pytest tests -q`.

## Reproduction

Run from the repo root; `python -m` resolves `src.*`. The scripts are thin shell entries — the Python lives in `src/`. Everything previews by default — `--execute` executes, `--detach` backgrounds. Benchmark aliases: `tau3`, `autob`, `tb2`. Only training, repetitions and transfer call the model; all analysis is offline.

**Main experiments** (the paper's study):

```bash
# 1. train the three optimizers and the two controls, three benchmarks x two models
bash scripts/baselines.sh suite deepseek-flash --execute
#    one cell instead of the suite: baselines.sh run tau3 gpt-6-luna --method skillopt
# 2. avg@3 repeated test evaluation of each condition's frozen final skill
bash scripts/baselines.sh repeat tau3 deepseek-flash --method skillopt --execute
# 3. Table 1 (both models) -> ../paper/assets/data/mech/mech_table1.md
bash scripts/analysis.sh table1
```

A test slot that ran in all three repetitions but never got a verifier verdict scores 0 (`adjudicated_failures`, never retried); a slot with no episode evidence blocks the average (`src/core/repeat.py`).

**Secondary experiments**:

```bash
# RQ1-RQ4 analyses and the main-text figures (offline; per target, compute first then figures)
bash scripts/analysis.sh rq1        # rq2 rq3 rq4 mainfigs table1 refresh, or "all"
# transfer: a returned skill evaluated unchanged under another model / benchmark
# (dry-run by default, the evaluation cache resumes; --execute calls the model)
bash scripts/analysis.sh transfer model|bench|tb2auto [--execute]
# execution-evidence audits: completion certificate, trace integrity, skill history
bash scripts/analysis.sh audit native-run|trace|skill-history ...
```

`analysis.sh all` runs `rq1 rq2 rq4 rq3 mainfigs` — the order the cross-RQ cache dependencies require (rq2's paired figure reuses rq1's statistics, rq3 reuses rq2's attribution cache, the cheatsheet audit reuses rq4's). Everything runs under the one `$BASELINE_PYTHON` interpreter. What each target runs (`src/analysis/`; figures are `draw`-family subcommands of the same module, drawing from the cache its compute step wrote):

- **RQ1** — do the dev gains survive the test split: paired statistics, winner's-curse replay, per-domain success, per-repetition reproducibility. `rq1/{stats,selection,domains,reruns}.py` + `appendix/{nullmodel,stability,appb}.py`
- **RQ2** — the token tax: the four-term attribution, within-task slopes, solved-vs-failed tasks, cached-input pricing, the compaction counterfactual, the optimizer's own training bill. `rq2/{attribution,slopes,failcost,price,compaction,optcost}.py` + `appendix/{appc,slopes,pricing,optcost,facts}.py`
- **RQ3** — what the optimizers write into the skill: rule coding and its second-coder check, calls per turn, accepted vs rejected edits, the answer-leakage audit. `rq3/{skilltext,coder2,callsperturn,edges,cheatsheet}.py`
- **RQ4** — generality: skills frozen and re-evaluated on another model / benchmark, and the trajectory-dynamics hypotheses. `rq4/{transfer,dynamics}.py` + the `analysis.sh transfer` drivers
- **Main-text figures** (`mainfigs`) — Fig. 1, the Sec.-4.3 per-skill figure, the Sec.-5 design-principle evidence. `mainfigs/{teaser,design}.py` + `rq3/skilltext.py skills`

Figures are `draw`-family subcommands of the same module, drawing from the cache its compute step wrote; each module's docstring lists its subcommands. Each module writes its audit record to `../paper/assets/data/mech/`.
