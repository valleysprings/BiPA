# figures

The teaser and the cumulative-tokens figure are drawn from the per-episode graph of `src/graph/episode.py`, on the Table 1 runs of DeepSeek, and read the recorded runs directly; the other figures read the caches of the data steps named with them.
`python -m aux.figures.teaser compute` writes `teaser.json` (one AutomationBench test task that the seed and SkillOpt's skill both solve, and the four-term split of the pair's change in log input tokens); `python -m aux.figures.teaser draw` draws `mech_teaser2.{pdf,svg}`.
`python -m aux.figures.cumulative` writes `bipcum_main.json` and draws `mech_bipcum_main2.{pdf,svg}`: input tokens read up to each turn, by prompt component, seed against SkillOpt (DeepSeek).
`python -m aux.figures.method` builds `mech_method.pdf` from the TikZ source `method.tex` with pdflatex: the method figure, whose values are illustrative.
`python -m aux.figures.skills` writes `skills2.json` and draws `mech_skills2.{pdf,svg}` from `attribution.json` and `skilltext.json`: per learned skill (DeepSeek), the four-term split of the change in log input tokens and the rules added by category.
`python -m aux.figures.bipcum` writes `bipcum3.json` and draws `mech_bipcum3.{pdf,svg}` from `bipdata.pkl` (`python -m aux.token_attribution.bipdata`): each learned skill's gap to the seed in the input tokens read so far, by context component, both models.
`python -m aux.figures.writes` writes `bipwrites3.json` and draws `mech_bipwrites3.{pdf,svg}` from `calls_rw_data.pkl` and `calls_rw.json` (`python -m aux.calls_and_writes.calls_rw extract`, then `stats`): AutoBench writes one per turn while reads stay grouped.
`layoutcheck.check(fig)` lists texts below 6.5 pt, outside the canvas or overlapping; `aux.edit_gate_and_design.implications_fig` and `aux.three_word_seed.scratch_fig` refuse to save a figure that fails it.
Caches go to `$BIPA_ASSETS/cache/analysis/main/`, figures to `$BIPA_OUT/fig/mech/`.
