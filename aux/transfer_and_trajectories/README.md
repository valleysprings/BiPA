# transfer_and_trajectories

Transfer of skills across models and benchmarks, trajectory dynamics (H1 / H2 / H4), exploration and convergence of gains (the transfer and gains figures, mech_dynamics, mech_explore).
`python -m aux.transfer_and_trajectories.trajectories [draw]` writes `dynamics.json` and `mech_dynamics.md`; `transfer [draw]` the single-run cross-model table; `luna_avg3` the Luna avg@3 cells.
`explore`, `explore_strat`, `convergent_gains` and `transfer_dq` write their JSON under `$BIPA_OUT/transfer_and_trajectories`; `rq4_figs` and `gains_fig [explore]` draw the figures (gains_fig also reads `pass3_ci.json` and `gains_tb2.json` from `python -m src.analysis.pooled gains`).
`transfer_ci` and `squeeze_base` print the Luna transfer intervals and the white-squeeze base rate.
