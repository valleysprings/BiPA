"""Independent recomputation of the section-3 counts (units by category, domain/generic, order statistic T).  Uses the paper's skilltext.py definitions
(units, classify, added_line_set) but own aggregation code."""
import json, re, difflib, math, collections
from pathlib import Path
import numpy as np
from src.layout import OUT


if __name__ == "__main__":
    SC = OUT / "skill_paths"
    from src.analysis.skilltext import added_line_set, classify, units
    runs = {}
    for p in sorted((SC / "paths").glob("*.json")):
        d = json.load(open(p)); runs[(d["model"], d["optimizer"], d["benchmark"], d["run_id"])] = d
    def cands(d): return [s for s in d["steps"] if s["event"] == "candidate"]
    GEN = {"verify", "serialize", "parallelize", "limit_read", "ask_stop"}
    def run_units(d):
        """list over candidates (produced order) of list of categories of added units"""
        byid = {s["candidate_id"]: s for s in d["steps"]}
        out = []
        for s in cands(d):
            par = byid.get(s["parent_id"])
            if par is None: out.append(None); continue
            a = par["skill_text"].splitlines(); b = s["skill_text"].splitlines()
            add = added_line_set(a, b)
            cats = [classify(u["text"]) for u in units(s["skill_text"]) if u["kind"] == "rule" and any(i in add for i in u["lines"])]
            out.append(cats)
        return out

    for opt in ("gepa","skillopt","skilladam"):
        for seed in ("hand-written","near-empty"):
            ks=[k for k in runs if k[1]==opt and (k[3]=="fromscratch")==(seed=="near-empty")]
            if not ks: continue
            first=collections.Counter(); allc=collections.Counter()
            for k in sorted(ks):
                u=run_units(runs[k])
                if u and u[0]: first.update(u[0])
                for c in u:
                    if c: allc.update(c)
            f=lambda c: (100*c["domain"]/sum(c.values()) if sum(c.values()) else float("nan"), sum(c.values()), 100*c["verify"]/max(1,sum(c.values())))
            print(opt, seed, "first-step domain share %.0f%% of %d units (verify %.0f%%) | all-steps domain share %.0f%% of %d units (verify %.0f%%)" % (f(first)[0], f(first)[1], f(first)[2], f(allc)[0], f(allc)[1], f(allc)[2]))
    print("record: GEPA 19/18, SkillOpt 16/16, SkillAdam 0/4; near-empty GEPA first candidate 60% verify vs 46% overall")
