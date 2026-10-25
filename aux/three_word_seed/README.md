# three_word_seed

GPT-6-Luna SkillOpt and GEPA runs started from the three-word seed (`fromscratch` runs): dev scores and the selection replay (`collect_dev`, then `selection_replay`), test success and cost (`rep_stats`, `rho`, `tau3_seed_overlap`), word counts (`wordcounts`), the sentence labels of the returned skills (`sentences`, then `rules_audit`), write and read-back fingerprints (`fingerprints`), and `make_numbers`, which assembles `numbers.json`, the data of the three-word-seed figure.
Run from the repo root in that order, e.g. `python -m aux.three_word_seed.collect_dev`, then `python -m aux.three_word_seed.selection_replay`; outputs go to `$BIPA_OUT/three_word_seed/`.
`python -m aux.three_word_seed.scratch_fig` then draws `mech_scratch2.{pdf,svg}` from `numbers.json`, the tau3 repetitions of the runs from the hand-written seed and the runs' candidate logs: test success from each seed, and every candidate's dev score against its length.
