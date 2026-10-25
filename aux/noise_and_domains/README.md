# noise-and-domains

Run-to-run noise of the test runs (`appb`, `reruns`, `stability`, `reruns_check`, `distinct_runs`), checks of the task model behind the replays (`load` caches the inputs; `a1_runshock`, `a2_exch`, `a2_priorll`, `a34_boot_mde`, `assumption_checks`, `replay_models`, `replay_shock`, `indep_replay`, `replay_power`), per-domain success (`domains`) and the devtest / paired figures (`stats_figs`).
`appb` draws every test run of the seed and of the returned skills (success and input-token ratio) against the seed's three-run spread.
Run from the repo root, e.g. `python -m aux.noise_and_domains.reruns compute|draw`, `python -m aux.noise_and_domains.domains [draw]`, `python -m aux.noise_and_domains.stats_figs devtest|paired`, `python -m aux.noise_and_domains.a1_runshock`. Script outputs go to `$BIPA_OUT/noise_and_domains/`.
