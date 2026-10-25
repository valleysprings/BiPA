# transfer_and_trajectories

Transfer of the skills across models and benchmarks, and whether a learned skill makes a task's runs agree.
`python -m aux.transfer_and_trajectories.transfer_dq` writes `transfer_dq.json` (success difference to the seed in the own setting and after transfer, the data of the transfer figure); `transfer_ci` prints the Luna tau3 differences, pooled over the transferred and over the Luna-learned skills.
`python -m aux.transfer_and_trajectories.transfer_fig` draws `mech_transfer3.{pdf,svg}` from `transfer_dq.json`, Table 1 and `luna_transfer.json` (`python -m aux.headline_numbers.transfer luna`): success difference and input-token ratio to the seed, in the own setting and transferred.
`python -m aux.transfer_and_trajectories.explore_strat` writes `explore_strat.json`; then `python -m aux.transfer_and_trajectories.gains_fig` draws `mech_gains2.{pdf,svg}` and writes `mech_gains.md` (tasks covered by one side only and the agreement odds ratio; it also reads `pass3_ci.json` and `gains_tb2.json` from `python -m aux.headline_numbers.pooled gains`).
`plot_style` holds the colors and the matplotlib settings shared with `aux.contamination_and_loading.trigger_fig`.
