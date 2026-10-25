"""Assemble numbers.json from the component json files written by the other scripts (no recomputation).
Run: python -m aux.three_word_seed.make_numbers"""
import json
from src import layout

O = layout.OUT / "three_word_seed"


def main():
    ld = lambda n: json.load(open(O / n))
    dev = ld("dev_collect.json")
    rules = ld("rules_audit.json")
    for v in rules.values():
        v.pop("verbatim_examples", None)
    out = dict(
        a_test=ld("test_stats.json"),
        b_wordcounts=ld("wordcounts.json"),
        c_rho=ld("rho.json"), c_tau3_seed_overlap=ld("tau3_seed_overlap.json"),
        d_dev=dict(collect={k: {a: b for a, b in v.items() if a not in ("dev",)} if isinstance(v, dict) else v for k, v in dev.items()},
                   replay=ld("selection_replay.json")),
        e_fingerprints=ld("fingerprints.json"),
        f_rules=rules,
    )
    json.dump(out, open(O / "numbers.json", "w"), indent=1, ensure_ascii=False)
    print("numbers.json", (O / "numbers.json").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
