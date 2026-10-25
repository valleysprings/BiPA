# cost_appendix

Cost under cache pricing, re-billing under compacted history, optimization cost, the slope tables and the facts scan, including the optimizer-cost tables and the check that OpenCode does not compact the history.
`python -m aux.cost_appendix.price [draw]`, `compaction compute|draw` and `optcost` write `mech_price3`, `mech_compaction` and `mech_optcost`; `training_cost` writes the optimizer-cost tables.
`python -m aux.cost_appendix.slope_checks [checks|reruns]` prints the slope tables and checks; `pricing` the price slope against the cache price; `facts` the facts scan (f1, f4, f6).
`python -m aux.cost_appendix.appc_draw fig1|compute|draw` gives the appC figures; `python -m aux.cost_appendix.slopes_draw` draws `mech_slopes` from `slopes.json`.
