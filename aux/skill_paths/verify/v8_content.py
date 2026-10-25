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
    res = {}
    for opt in ("gepa", "skillopt", "skilladam"):
        for seed in ("hand-written", "near-empty"):
            ks = [k for k in runs if k[1] == opt and (k[3] == "fromscratch") == (seed == "near-empty")]
            if not ks: continue
            cnt = collections.Counter(); ncand = 0; ru = {}
            for k in sorted(ks):
                u = run_units(runs[k]); ru[k] = u
                ncand += len(u)
                for c in u:
                    if c: cnt.update(c)
            tot = sum(cnt.values()); gen = sum(cnt[c] for c in GEN)
            print(f"{opt:9s} {seed:12s} runs {len(ks)} candidates {ncand} units {tot} " + " ".join(f"{c}={cnt[c]}" for c in ("verify", "serialize", "parallelize", "limit_read", "ask_stop", "domain", "format", "other")) + f" | domain/generic {cnt['domain']}/{gen} = {gen/max(cnt['domain'],1):.1f}:1")
            # halves: first half of each run's candidates vs second half
            fh = collections.Counter(); sh = collections.Counter()
            for k, u in ru.items():
                n = len(u)
                for i, c in enumerate(u):
                    if not c: continue
                    (fh if i < n / 2 else sh).update(c)
            f_ = lambda c: 100 * c["domain"] / max(1, sum(c.values()))
            # observed T and permutation p
            def Tstat(order_perm):
                dom, gen_ = [], []
                for k, u in ru.items():
                    n = len(u); perm = order_perm(k, n)
                    for i, c in enumerate(u):
                        if not c: continue
                        pos = perm[i] / (n - 1) if n > 1 else 0.0
                        for cat in c:
                            if cat == "domain": dom.append(pos)
                            elif cat in GEN: gen_.append(pos)
                return (np.mean(dom) - np.mean(gen_)) if dom and gen_ else float("nan")
            rng = np.random.default_rng(3)
            T0 = Tstat(lambda k, n: np.arange(n)); NP = 5000; ex = 0
            for _ in range(NP):
                if abs(Tstat(lambda k, n: rng.permutation(n))) >= abs(T0) - 1e-12: ex += 1
            print(f"      domain share first half {f_(fh):.0f}% second half {f_(sh):.0f}% | T {T0:+.3f} two-sided perm p {(ex+1)/(NP+1):.2f}")
    print("record: GEPA hw 110 cand 3315 units (1751 49 18 26 230 608 82 551) 19%/18% T -0.010 p 0.52; SkillOpt hw 26 cand 348 (191 14 2 5 23 55 8 50) 16/15 T +0.002 p 0.96; SkillAdam hw 100 cand 170 (123 3 2 1 12 6 4 19) 4/2 T -0.132 p 0.35; GEPA ne 20 cand 296; SkillOpt ne 4 cand 12")
