# analysis (package `src.analysis` at `bipa/src/analysis/`)

Offline analysis of finished runs: read-only on `assets/artifacts/execution`, no model calls. It turns runs into paper 1's (draft2mech) tables, figures and numbers: the five reported arms are native-noseed, native-seed, gepa, skillopt, skilladam, on the models deepseek-flash and gpt-6-luna.

Commands run from `bipa/`. The interpreter is the bach-harness venv (`BASELINE_PYTHON`, default `/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation/bach-harness/.venv/bin/python`); running `python -m` from the bipa root puts the root on sys.path, so `src.analysis` resolves.

Data steps (the venv has the scientific stack but no matplotlib):
`<python> -m src.analysis.<part>.<module> <cmd>` from `bipa/`.
Figure steps (system `python3` has matplotlib; run it from `bipa/` so `src` resolves):
`python3 -m src.analysis.<part>.<module> draw`.

## Layout

- **Root (`src/analysis/*.py`) — shared library plus the Table 1 pipeline.** `common.py` and `paperout.py` are import-only libraries (no `__main__`): paths, loaders, skill version ids, the audit-record writer. `report.py` is the stats/ratios helper used by `mech/transfer.py`. `table1.py`, `submission_protocol.py`, `submission_audit.py` are entry points (`python -m src.analysis.<module>`).
- **`mech/` — paper-1 (BiPA) analyses** feeding the draft2mech figures and records (attribution, domains, dynamics, edges, slopes, transfer, ...; each module has its own entry point and docstring).
- **`mechanism/` — where the optimized skills' tokens come from, and whether the gains hold.** Two experiment threads: (a) the Sec 3 cost mechanism (elasticity slopes, gap decomposition, pricing) with the Appendix C figures; (b) the Sec 5 selection-noise / reproducibility checks (reruns, pooled, null model, paired, power) with the Appendix B figures. (The BACH lever-test figure `appd.py` stayed in bach-harness; it is paper-2-only.)

Paper output: figure modules write the draft figures (appB_*, appC_* to `paper/assets/fig/mech/` as pdf + svg with svg.fonttype none) and, together with the table modules, one audit record per draft table / data figure to `paper/assets/data/mech/<name>.md` through `paperout.record()`: table1 -> mech_table1, mechanism/appb -> appB_{reruns,nullsim}, mechanism/appc -> appC_{slopes,attribution,price}. No .json/.tex/.txt data files in `paper/assets/data`. Intermediates and caches (table1.json, cache.json, previews, ...) live in `assets/cache/analysis/<module>/` (`common.out_dir`) and are rebuilt by the modules.

## Shared (`src/analysis/`)

| module | role | produces | paper element | output path | command |
|---|---|---|---|---|---|
| common.py | library (imported) | paths (H, EX, DS), loaders (current_run, eval_file, final_skill, returned_skills), skill version ids (vid), out_dir(module), Report | all | `assets/cache/analysis/<module>/` | imported |
| paperout.py | library (imported) | savefig (the data figures), mdtable, record(name, part, result, commands, raw, notes) -> md with the sections Paper / Source / When (UTC, from launch.log stamps and evaluation.json, summary.json, native_protocol.json mtimes of the raw records) / Result / Raw records / Notes; several scripts can fill one record (parts kept in assets/cache/analysis/_records/) | paper/assets/data/mech/<name>.md | imported |
| report.py | library (imported) | stats (success/known/tokens of one run) and ratios (rho, rho^calls vs the Seed row) over plain episode records | mech_transfer | none | imported |
| table1.py | entry | test Q (avg@3) and the cost ratios (rho, calls, input tokens) of each method vs the Seed row on a common task set, both models; ranks and dagger. Adjudication: a task whose episodes ran in all three repetitions but never produced a verdict counts 0 | Table 1 (tab:mech-main); record paper/assets/data/mech/mech_table1.md; table1.json is the cache read by fig1, reruns, facts | assets/cache/analysis/table1/{table1.json,table1.tex} | `<python> -m src.analysis.table1` |
| submission_protocol.py | entry | freezes the submission protocol: protocol.json + PROTOCOL.md (which run of every row/benchmark is scored, with hashes) | Table 1 and every later analysis (read through env BACH_SUBMISSION_MANIFEST) | assets/cache/analysis/submission-20261008/ | `<python> -m src.analysis.submission_protocol [--out DIR]` |
| submission_audit.py | entry | read-only re-hash of everything the manifest names; lists pending or inconsistent cells (exit 0 only when all complete and consistent) | check before the final Table 1 | stdout | `<python> -m src.analysis.submission_audit --manifest FILE` |

## mechanism (`src/analysis/mechanism/`)

### (a) Where the optimized skills' tokens come from (Sec 3, Appendix C)

| module | produces | paper element | output path | command |
|---|---|---|---|---|
| mechanism/episodes.py | shared: per-episode rows (turns, input, cache, output, calls) of the DeepSeek test evaluations, within-task fixed-effect slope | none (imported) | none | none |
| mechanism/slopes.py | slope of log X on log R by sample/skill, gap decomposition slope = e_R + s_H b_h + s_F b_F, last-quarter input share, turns vs success, extra-turn mechanism | Sec 3 elasticity claims, appendix slope tables | slopes/slopes.txt | `<python> -m src.analysis.mechanism.slopes` |
| mechanism/slopes_checks.py | per-skill decomposition dlogX = act + turns + content + text (limit vs exact e_R, R^2), within-task slopes (main, seed x3, h, F), 1+mean s_H, exact e_R; per-skill fit.json (the four terms t_N/t_m/t_h/t_F, pooled fits) | Sec 3 numbers, appendix slopes and costsplit text, eq:split | assets/cache/analysis/slopes_checks/{checks,fit}.json | `<python> -m src.analysis.mechanism.slopes_checks [--model deepseek-flash]` |
| mechanism/pricing.py | price slope vs cache price c, cross-over c* where the slope reaches 1, price shares, marginal turn price | Sec 3 price-elasticity claim | pricing/pricing.txt | `<python> -m src.analysis.mechanism.pricing` |
| mechanism/fig1.py | data stage of the Appendix C figures: within-task slopes and the per-skill four-term split (draw only writes a preview) | input of appc.py | assets/cache/analysis/fig1/{cache.json,numbers.json,fig1.png} | `<python> -m src.analysis.mechanism.fig1 compute` |
| mechanism/appc.py | Appendix C figures: slopes, attribution, price | paper/assets/fig/mech/appC_{slopes,attribution,price}.{pdf,svg}, data/mech/appC_*.md | assets/cache/analysis/appc/ | `<python> -m src.analysis.mechanism.appc compute; python3 -m src.analysis.mechanism.appc draw` |
| mechanism/facts.py | output tokens and price shares, within-task price slopes (f1), compaction scan (f4), filter counts on V (f6) and test (f6b), facts-v9.md | Sec. 3.4/5 price numbers, appendix compaction and sample sizes (facts-v9 Sec. 1, 4, 6) | assets/cache/analysis/facts/{f1,f4,f6,f6b}.json, facts-v9.md | `<python> -m src.analysis.mechanism.facts` |
| mechanism/optcost.py | episodes consumed/executed on train and V, agent tokens, optimizer calls/tokens, largest prompt, cut prompts (tab:optcost rows in optcost.tex); training-cost cells, returned-skill words/blocks/diffs, test cost ratios and USD (cost.json) | tab:optcost (appendix C), facts Sec. 2, appendix returned skills stats | assets/cache/analysis/optcost/{optcost.tex,llm.json,cost.json,cost_table.tex,training-cost*.{csv,json,tex},skills*.{csv,json,tex},skills-diff-excerpts.txt} | `<python> -m src.analysis.mechanism.optcost` |

### (b) Why their success gains do not replicate — selection noise (Sec 5, Appendix B)

| module | produces | paper element | output path | command |
|---|---|---|---|---|
| mechanism/reproducibility.py | rho_X/rho_R of reruns vs seed, seed noise floor, flip fraction, log-X/log-R correlation across reruns (also the run loaders) | "Do the gains hold?" (Sec 5), appendix seed spread (no record) | reproducibility/reproducibility.txt | `<python> -m src.analysis.mechanism.reproducibility` |
| mechanism/pooled.py | pooled success over 3 runs and pooled X/P/R/N/O ratios to the seed per optimizer x bench, seed floor | "Do the gains hold?" pooled numbers (Sec 5; no record) | pooled/pooled.json, pooled.txt | `<python> -m src.analysis.mechanism.pooled` |
| mechanism/nullmodel.py | winner's-curse null model (fitted / conc 4 / homogeneous): P(gain >= observed) per optimizer protocol, z of the changed cells; needs scipy | Sec 5 selection paragraph, appendix nullsim text (no record) | nullmodel/{obs_test,out,out_c4,out_c1000000}.json, summary.txt, zcells.txt | `<python> -m src.analysis.mechanism.nullmodel` |
| mechanism/paired.py | task-level paired summary of every complete row vs the seed's three test executions | selection-noise appendix | `--out DIR` | `<python> -m src.analysis.mechanism.paired --manifest FILE --out DIR` |
| mechanism/power.py | power / minimum detectable effect (MDE80) of the submission protocol | selection-noise appendix | `--out DIR` (power.json, power.md) | `<python> -m src.analysis.mechanism.power --manifest FILE --out DIR` |
| mechanism/reruns.py | per-episode e_R range, task / skill fixed-effect slopes, run-by-run and pooled success; seed's three-run spread of mean log input | rerun paragraph of Sec. 5 | assets/cache/analysis/reruns/{check.json,seed_spread.json} | `<python> -m src.analysis.mechanism.reruns` |
| mechanism/appb.py | Appendix B figures: appB_reruns (per test run of seed and returned skills: success and input-token ratio vs the pooled seed; the B.2 prose values are asserted) and appB_nullsim (zero-effect dev-selection replay distributions, 200k draws, frozen from the nullmodel run behind B.3) | paper/assets/fig/mech/appB_{reruns,nullsim}.{pdf,svg}, data/mech/appB_*.md | assets/cache/analysis/appb/{reruns,nullsim}.json | `<python> -m src.analysis.mechanism.appb compute; python3 -m src.analysis.mechanism.appb draw` |

## All records, in this order

Writes paper/assets/data/mech/*.md and paper/assets/fig/mech/*.{pdf,svg} (from `bipa/`):

`<python> -m src.analysis.table1; <python> -m src.analysis.mechanism.fig1 compute; <python> -m src.analysis.mechanism.appc compute; python3 -m src.analysis.mechanism.appc draw; <python> -m src.analysis.mechanism.appb compute; python3 -m src.analysis.mechanism.appb draw`
