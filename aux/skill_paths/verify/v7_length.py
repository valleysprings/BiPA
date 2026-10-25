import json, math
from pathlib import Path
import numpy as np
from math import comb
from src.layout import OUT


if __name__ == "__main__":
    SC = OUT / "skill_paths"
    runs = {}
    for p in sorted((SC / "paths").glob("*.json")):
        d = json.load(open(p)); runs[(d["model"], d["optimizer"], d["benchmark"], d["run_id"])] = d
    def cands(d): return [s for s in d["steps"] if s["event"] == "candidate"]
    def scored(d): return [s for s in cands(d) if s.get("dev")]
    def norm(t): return [l.strip() for l in t.splitlines() if l.strip()]
    def sign_p(k, n):
        return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)
    HW = sorted(k for k in runs if k[3] != "fromscratch")
    res = {}
    for opt in ("gepa", "skillopt", "skilladam"):
        per_run = []; firsts = []; allr = []
        for k in [k for k in HW if k[1] == opt]:
            d = runs[k]; seed_words = d["steps"][0]["skill_words"]
            if opt == "gepa":
                sc = scored(d); seq = [s["dev"]["n_success"] for s in sc]; base = d["seed_dev"]["n_success"]
                best = max(seq); kb = seq.index(best) if best > base else -1
                best_words = sc[kb]["skill_words"] if kb >= 0 else seed_words
                later = sc[kb + 1:]
            elif opt == "skillopt":
                sc = cands(d); seq = [s["dev"]["n_success"] for s in sc]; base = d["seed_dev"]["n_success"]
                best = max(seq); kb = seq.index(best) if best > base else -1
                best_words = sc[kb]["skill_words"] if kb >= 0 else seed_words
                later = sc[kb + 1:]
            else:
                sc = cands(d); last = max([i for i, s in enumerate(sc) if s["kept"]], default=-1)
                best_words = sc[last]["skill_words"] if last >= 0 else seed_words
                later = []
                for s in sc[last + 1:]:
                    par = next((x for x in d["steps"] if x["candidate_id"] == s["parent_id"]), None)
                    if par is not None and norm(par["skill_text"]) != norm(s["skill_text"]): later.append(s)
            if not later:
                print(opt, k[0][:6], k[2][:6], "no later candidates"); continue
            ratios = [s["skill_words"] / best_words for s in later]
            n_longer = sum(r > 1 for r in ratios)
            g = math.exp(np.mean(np.log(ratios)))
            per_run.append(g); firsts.append(ratios[0] > 1); allr += ratios
            print(f"{opt:9s} {k[0][:6]} {k[2][:6]:7s} later {n_longer}/{len(ratios)} gm {g:.2f} max {max(ratios):.2f} first later longer: {ratios[0] > 1}")
        ab = sum(g > 1 for g in per_run)
        print(f"  => {opt}: runs with later {len(per_run)}, gm>1 in {ab} (sign p {sign_p(ab, len(per_run)):.3f}), gm of per-run gm {math.exp(np.mean(np.log(per_run))):.3f}, gm over all later candidates {math.exp(np.mean(np.log(allr))):.3f}, first later longer in {sum(firsts)}/{len(firsts)}")
    print("record: GEPA 5/6 pooled 1.15 sign p 0.22, first-later-longer 3/6; SkillOpt 5/5 1.27 p 0.06 first 5/5; SkillAdam 5/5 1.06 p 0.06 first 5/5")
