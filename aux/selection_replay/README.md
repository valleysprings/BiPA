# selection_replay

Replays of the optimizers' own selection rules on copies of the seed. `python -m aux.selection_replay.selection compute` writes `selection.json`, `dev2test.json` and `mech_selection.md` (the dev gain each rule reports when no candidate differs from the seed, DeepSeek Table 1 runs); `draw` draws `mech_dev2test4.{pdf,svg}` and reads `table1.json` from `python -m src.analysis table1`.
The other scripts import its task model: `luna` (the GPT-6-Luna replay, `mech_selection_luna.md`), `devsize` (the dev-set-size sweep, `mech_devsize.md`), `selection_fig` (the `mech_selection` figure from `selection.json`), and `collect`, then `fig_devpertask` (task-by-task dev and test flips, the `mech_devpertask` figure).
Run from the repo root, e.g. `python -m aux.selection_replay.luna`, `python -m aux.selection_replay.devsize`. Outputs go to `$BIPA_OUT/selection_replay/`, records and figures to `$BIPA_OUT/{data,fig}/mech/`, caches to `$BIPA_ASSETS/cache/analysis/main/`.
