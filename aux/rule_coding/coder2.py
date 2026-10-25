"""Second-coder agreement on the eight-category coding of the added rule units (skilltext.json).

A second coder (a language model working from the category definitions only) labelled every unit
(CACHE/analysis/main/coder2_labels.json); this compares those labels with the final labels and reports
percent agreement, Cohen's kappa (overall and per relied-on group), the confusion matrix and the counts.
Unit id = "<optimizer>|<benchmark>#<idx>".  Run:  python -m aux.rule_coding.coder2
"""
import json

import numpy as np

from src.utils import logger, dataloader
CATS = ["serialize", "parallelize", "limit_read", "ask_stop", "verify", "domain", "format", "other"]
# Groups the paper relies on; each becomes a binary "in group / not in group" coding.
GROUPS = {"verify": ["verify"], "serialize": ["serialize"],
          "limit_read + parallelize": ["limit_read", "parallelize"]}
# The first skilltext cell (gepa|tau3-bench) had its labels shown to coder 2 before coding, so it is not blind.
EXPOSED = "gepa|tau3-bench#"


def kappa(a, b):
    """Cohen's kappa of two label sequences; (agreement, kappa)."""
    cats = sorted(set(a) | set(b))
    table = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        table[cats.index(x), cats.index(y)] += 1
    n = table.sum()
    observed = np.trace(table) / n
    expected = (table.sum(0) * table.sum(1)).sum() / n ** 2
    value = 1.0 if expected == 1 else (observed - expected) / (1 - expected)
    return observed, value


def final_labels():
    """Final label of every unit from skilltext.json, keyed by unit id."""
    coded = json.loads((dataloader.CACHE / "skilltext.json").read_text())["coded_units"]
    return {f"{cell}#{u['idx']}": u["category"] for cell, rows in coded.items() for u in rows}


def compare(ids, first, second):
    """Overall and per-group agreement/kappa over the unit ids."""
    a = [first[i] for i in ids]
    b = [second[i] for i in ids]
    agree, k = kappa(a, b)
    out = {"n": len(ids), "agreement": agree, "kappa": k, "groups": {}}
    for name, members in GROUPS.items():
        ga = [x in members for x in a]
        gb = [x in members for x in b]
        out["groups"][name] = dict(zip(("agreement", "kappa"), kappa(ga, gb)),
                                   final=sum(ga), coder2=sum(gb))
    return out


def compute():
    """All numbers: full set, set without the exposed cell, confusion matrix, counts."""
    final = final_labels()
    second = json.loads((dataloader.CACHE / "coder2_labels.json").read_text())
    assert set(final) == set(second)
    ids = sorted(final)
    blind = [i for i in ids if not i.startswith(EXPOSED)]
    confusion = {f: {s: 0 for s in CATS} for f in CATS}
    for i in ids:
        confusion[final[i]][second[i]] += 1
    count = lambda lab: {c: sum(1 for i in ids if lab[i] == c) for c in CATS}
    return {"all": compare(ids, final, second), "blind": compare(blind, final, second),
            "confusion": confusion, "counts_final": count(final), "counts_coder2": count(second),
            "n_disagree": sum(final[i] != second[i] for i in ids)}


def record(r):
    """Write the JSON and the markdown audit record."""
    dataloader.write_json("coder2", r)
    n = r["all"]["n"]
    rows = []
    for tag, key in ((f"all {n} units", "all"), (f"{r['blind']['n']} blind units (without gepa tau3)", "blind")):
        d = r[key]
        rows.append([tag, f"{100 * d['agreement']:.1f}%", f"{d['kappa']:.3f}"]
                    + [f"{100 * d['groups'][g]['agreement']:.1f}% / {d['groups'][g]['kappa']:.3f}" for g in GROUPS])
    agree = logger.md_table(["Units", "agreement", "kappa (8 categories)"]
                       + [f"{g}: agreement / kappa (binary)" for g in GROUPS], rows)
    cnt = logger.md_table(["category", "final", "final %", "coder 2", "coder 2 %"],
                     [[c, r["counts_final"][c], f"{100 * r['counts_final'][c] / n:.1f}",
                       r["counts_coder2"][c], f"{100 * r['counts_coder2'][c] / n:.1f}"] for c in CATS])
    conf = logger.md_table(["final \\ coder 2"] + CATS,
                      [[f] + [r["confusion"][f][s] for s in CATS] for f in CATS])
    grp = logger.md_table(["group", "units in final", "units in coder 2"],
                     [[g, r["all"]["groups"][g]["final"], r["all"]["groups"][g]["coder2"]] for g in GROUPS])
    extra = "\n".join([
        "### Counts per category", "", cnt, "", "### Units per relied-on group", "", grp, "",
        "### Confusion matrix (rows: final labels, columns: coder 2)", "", conf, "",
        "### Protocol", "",
        f"Blind second coder. The {n} unit texts were exported without labels and coder 2 (a language model) "
        "labelled each from the category definitions only (appendix D and section 4.2 of mech_skilltext.md), "
        "without opening the final labels. Exception: the labels of the first cell (gepa tau3-bench, "
        f"{n - r['blind']['n']} units) were shown to coder 2 before coding, so that cell is not blind; "
        f"agreement is reported on all {n} units and on the {r['blind']['n']} blind units. Labels: "
        "`assets/cache/analysis/main/coder2_labels.json`. Coder 2 is a model, not a human.",
        f"Units with different labels: {r['n_disagree']} of {n}."])
    logger.write_record("coder2", "Inter-coder agreement for the eight-category coding of the 337 added rule units.",
                   "Nine learned skills (GEPA, SkillOpt, SkillAdam on tau3-bench, AutoBench, TB2); units from skilltext.json.",
                   "Percent agreement and Cohen's kappa between coder 2 and the final labels; binary kappa for each relied-on group.",
                   agree, "python -m aux.rule_coding.coder2", extra)


if __name__ == "__main__":
    record(compute())
