# cost_appendix

The cost of the learned skills under cache pricing and under a compacted history, and the cost of the optimization itself.
`python -m aux.cost_appendix.price` writes `mech_price3.md` (price ratio to the seed with cached input at 0.1 and output at 4 times an uncached input token, and over nine weightings).
`python -m aux.cost_appendix.compaction compute`, then `record`, writes `mech_compaction.md` (the recorded episodes re-billed with only the last w turns of history).
`python -m aux.cost_appendix.optcost` writes `mech_optcost.md` (each optimizer run's bill in seed test episodes and the break-even number of episodes); `python -m aux.cost_appendix.training_cost` writes the optimizer-cost table `optcost.tex` (episodes, tokens, optimizer calls, the largest optimizer prompt and the prompts cut to the prompt limit).
