# figures

Two figures drawn from the per-episode graph of `src/graph/episode.py`, on the Table 1 runs of DeepSeek; they read the recorded runs directly.
`python -m aux.figures.teaser compute` writes `teaser.json` (one AutomationBench test task that the seed and SkillOpt's skill both solve, and the four-term split of the pair's change in log input tokens); `python -m aux.figures.teaser draw` draws `mech_teaser2.{pdf,svg}`.
`python -m aux.figures.cumulative` writes `bipcum_main.json` and draws `mech_bipcum_main2.{pdf,svg}`: input tokens read up to each turn, by prompt component, seed against SkillOpt (DeepSeek).
Caches go to `$BIPA_ASSETS/cache/analysis/main/`, figures to `$BIPA_OUT/fig/mech/`.
