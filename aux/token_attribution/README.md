# token_attribution

The four-term split of the input-token change: its record for DeepSeek, its GPT-6-Luna version, the cost of failed tasks, and the episode graphs of the cumulative-tokens figure.
`python -m aux.token_attribution.attribution_draw` writes `mech_attribution3.md` from `attribution.json` (`python -m src.analysis.attribution compute`); `python -m aux.token_attribution.failcost compute|record` gives `mech_failcost.md`.
`python -m aux.token_attribution.attribution_luna` writes `attribution_luna.json` and `mech_attribution_luna.md`; then `python -m aux.token_attribution.bipdata` collects the episode graphs into `bipdata.pkl` (under `$BIPA_OUT/token_attribution`), the data of the cumulative-tokens figure of every learned skill.
