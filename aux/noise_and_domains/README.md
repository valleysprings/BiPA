# noise_and_domains

Run-to-run noise of the test runs, checks of the task model behind the selection replay, and per-domain success.
`python -m aux.noise_and_domains.appb` writes `reruns.json` (success and input-token ratio of every test run of the seed and of the returned skills against the seed's three runs), the numbers of the rerun figure.
`python -m aux.noise_and_domains.load` caches the seed's dev and test executions (`data.pkl`); then `python -m aux.noise_and_domains.a2_exch` tests whether the dev and test tasks share one Beta prior, and `python -m aux.noise_and_domains.replay_models` reruns the selection replay under task models without the pooled prior, including the dev-size curve with tasks drawn from the observed ones. Their outputs go to `$BIPA_OUT/noise_and_domains/`.
`python -m aux.noise_and_domains.domains` writes `domains.json` and `mech_domains.md` (test success per task domain).
`python -m aux.noise_and_domains.reruns_fig` adds the GPT-6-Luna runs to `appb`'s numbers (`reruns_models.json`) and draws `mech_reruns3.{pdf,svg}`: success and input tokens against the seed for every test run, both models.
`python -m aux.noise_and_domains.domains_fig` (after `domains`) draws `mech_domains_wide.{pdf,svg}`: test success per domain of tau3-bench and AutoBench, DeepSeek rows.
