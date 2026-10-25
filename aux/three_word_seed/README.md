# three_word_seed

GPT-6-Luna SkillOpt and GEPA runs started from the three-word seed (`fromscratch` runs): dev scores and the selection replay (`collect_dev`, then `selection_replay`), test success and cost (`rep_stats`, `rho`, `tau3_seed_overlap`), word counts (`wordcounts`), the sentence labels of the returned skills (`sentences`, then `rules_audit`), write and read-back fingerprints (`fingerprints`), and `make_numbers`, which assembles `numbers.json`, the data of the three-word-seed figure.
Run from the repo root in that order, e.g. `python -m aux.three_word_seed.collect_dev`, then `python -m aux.three_word_seed.selection_replay`; outputs go to `$BIPA_OUT/three_word_seed/`.
