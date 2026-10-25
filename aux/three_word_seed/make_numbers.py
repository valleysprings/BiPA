"""Assemble numbers.json from the component json files written by the other scripts (no recomputation).
Run: python -m aux.three_word_seed.make_numbers"""
import json
from src import layout

O = layout.OUT / "three_word_seed"


def main():
    ld = lambda n: json.load(open(O / n))
    sel, dev, fp, ts = ld("selection_replay.json"), ld("dev_collect.json"), ld("fingerprints.json"), ld("test_stats.json")
    rules = ld("rules_audit.json")
    for v in rules.values():
        v.pop("verbatim_examples", None)
        for j in v.get("blind_judges", {}).values():
            j.pop("per_sentence", None)
    out = dict(
        returned_skills=ld("identify.json"),
        a_test=ts,
        b_wordcounts=ld("wordcounts.json"),
        c_rho=ld("rho.json"), c_cost_claim_origin=ld("cost_claim.json"), c_tau3_seed_overlap=ld("tau3_seed_overlap.json"),
        d_dev=dict(collect={k: {a: b for a, b in v.items() if a not in ("dev",)} if isinstance(v, dict) else v for k, v in dev.items()}, replay=sel, candidate_distribution=ld("candidate_dist.json")),
        e_fingerprints=fp, e_variants=ld("fingerprints_variants.json"),
        f_rules=rules,
    )
    out["verdicts"] = {
        "pre_returned_skill": "CORRECTED: SkillOpt tau3 returned dfd18dc47eb0 = 0.600 (summary.test.final.success; extra.test_hard); d0f2b28fe27d = 0.667 is the post-slow-update last iterate (extra.final_test_hard), not returned",
        "a_hand_seed": "CONFIRMED: AutoBench 0.3722 (0.3417/0.4/0.375), tau3 0.6933; from-scratch +6.1 (SkillOpt) / +7.2 (GEPA) points on AutoBench; no gain on tau3",
        "b_wordcounts": "CONFIRMED: 347/330/388/438 (file incl. front matter), seed 580",
        "c_tau3_rho": "CORRECTED: rho 1.112 (SkillOpt) / 1.136 (GEPA) vs own seed eval; -5%/+45% = summary.json test.cost ratios (N + X/1000 over unmatched capture-complete episodes)",
        "d_dev": "CONFIRMED +20.0/+20.0 AutoBench dev gain; '13-17 threshold' internally inconsistent: paper fitted conditional model gives chance mean 1.8/4.0 (P(>=+20)=0.001/0.005); the null models estimated from all candidates give 14.2-17.0 (13-17) but then P(>=+20)=0.066-0.244, i.e. not far above; tau3 dev gains (+10.0/+6.7) are at chance (P=0.28/0.75)",
        "e1_writes_share": "CORRECTED: 3-word seed 58.7% (59.9% main), SkillOpt-fs 4.3% (3.75% main), GEPA-fs 1.5% (1.3% main); pooled learned 3.0%; 62% not reproduced",
        "e2_readback": "NOT VERIFIABLE as stated: last write followed by any read 53.8% -> 97.2%/97.8%; same-service writes 56.3% -> 93.1%/93.8%",
        "f_rules": "NOT VERIFIABLE (62-76%); '0 task-specific' does not survive: S2+S3 sentences 0/1/6/4 (SkillOpt-AB, SkillOpt-tau3, GEPA-AB, GEPA-tau3); GEPA-tau3 has MMS/APN and rewards sentences",
    }
    json.dump(out, open(O / "numbers.json", "w"), indent=1, ensure_ascii=False)
    print("numbers.json", (O / "numbers.json").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
