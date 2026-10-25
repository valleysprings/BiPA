# token_attribution

The four-term split of the input-token change under Luna and with intervals, the cost of failed tasks, and the bipartite figures (bip_cum and the others below), with the records mech_attribution_luna and mech_failcost.
`python -m aux.token_attribution.attribution_draw` draws `mech_attribution3` from `attribution.json`; `python -m aux.token_attribution.failcost compute|record` gives `mech_failcost.md`.
`python -m aux.token_attribution.attribution_luna` writes `attribution_luna.json` and `mech_attribution_luna.md`; then `python -m aux.token_attribution.bipdata` collects the episode graphs into `bipdata.pkl`.
`fig_split`, `fig_view`, `fig_delta`, `fig_cumdelta`, `fig_cumabs`, `fig_cum_main` draw the bipartite figures from those files, and `write_records` writes their notes (all under `$BIPA_OUT/token_attribution`).
