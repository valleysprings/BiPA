# three-word-seed

GPT-6-Luna SkillOpt and GEPA runs started from the three-word seed (`fromscratch` runs): returned skills and word counts (`identify`, `wordcounts`, `sentences`), dev scores and the selection replay (`collect_dev`, `selection_replay`, `candidate_dist`), test success and cost (`rep_stats`, `rho`, `cost_claim`, `tau3_seed_overlap`), call fingerprints (`fingerprints`, `fingerprints_variants`), the rule audit (`rules_audit`) and `make_numbers`, which assembles numbers.json.
Run from the repo root in that order, e.g. `python -m aux.three_word_seed.collect_dev`, then `python -m aux.three_word_seed.selection_replay`; outputs go to `$BIPA_OUT/three_word_seed/`.
