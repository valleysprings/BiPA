# selection-replay

Replays of the optimizers' own selection rules on copies of the seed: the GPT-6-Luna replay (`luna`), the dev-set-size sweep (`devsize`), task-by-task dev/test flips (`collect`, then `fig_devpertask`, `fig_devmove`, `fig_devtest`), the dev-gain figures (`fig_devdyn`, `fig_chance`, `fig_dev2test`), the earlier mech_selection figure (`selection_fig`) and a check of the task model against the seed's reruns (`replay_check`).
Run from the repo root, e.g. `python -m aux.selection_replay.luna`, `python -m aux.selection_replay.devsize compute|draw`, `python -m aux.selection_replay.collect`. Outputs go to `$BIPA_OUT/selection_replay/`, records and figures to `$BIPA_OUT/{data,fig}/mech/`, caches to `$BIPA_ASSETS/cache/analysis/main/`.
