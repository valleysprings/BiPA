"""Independent recomputation, part 2: SkillAdam gates, kept rules, 59 edges, returned text, cost rho, length, pooled means, sign tests."""
import json, math, re, itertools, sys
from pathlib import Path
from math import comb, exp, log
from src.layout import OUT


if __name__ == "__main__":
    SC = OUT / "skill_paths"
    runs = {}
    for p in sorted((SC / "paths").glob("*.json")):
        d = json.load(open(p)); runs[(d["model"], d["optimizer"], d["benchmark"], d["run_id"])] = d
    def cands(d): return [s for s in d["steps"] if s["event"] == "candidate"]
    def scored(d): return [s for s in cands(d) if s.get("dev")]
    def norm(t): return [l.strip() for l in t.splitlines() if l.strip()]
    def gm(x): return exp(sum(log(v) for v in x) / len(x))
    def sign_p(k, n):
        m = min(k, n - k)
        return min(1.0, 2 * sum(comb(n, i) for i in range(0, m + 1)) / 2 ** n)
    HW = [k for k in runs if k[3] != "fromscratch"]
    DS = [k for k in HW if k[0] == "deepseek-flash"]; LU = [k for k in HW if k[0] == "gpt-6-luna"]
    print("hand-written runs:", len(HW), "deepseek", len(DS), "luna", len(LU))

    # ---------- A. SkillAdam gate differences, summed gain, live counts
    print("\n== A. SkillAdam")
    adam_gain = {}
    for k in sorted(runs):
        if k[1] != "skilladam": continue
        d = runs[k]; N = None; acc = []; live = 0; nit = 0
        for s in cands(d):
            nit += 1
            g = s["gate"]
            if not g: continue
            det = g["detail"]
            if det.get("candidate") and det.get("baseline"):
                diff = det["candidate"]["n_success"] - det["baseline"]["n_success"]
                nn = det["candidate"]["n"]
            else:
                diff = None; nn = None
            if s["kept"]:
                acc.append((s["step"], diff, nn))
            # live = text differs from parent text
            par = next((x for x in d["steps"] if x["candidate_id"] == s["parent_id"]), None)
            if par is not None and norm(par["skill_text"]) != norm(s["skill_text"]) and s.get("skill_text"):
                live += 1
        gain_pts = sum(100 * df / nn for _, df, nn in acc)
        adam_gain[k] = gain_pts
        kept_n = sum(1 for s in cands(d) if s["kept"])
        print(k[0][:6], k[2][:6], k[3][:10], "iterations", nit, "dev-scored", len(scored(d)), "accepted", [(a, b) for a, b, _ in acc], "gain pts %.1f" % gain_pts, "live(text differs from parent)", live, "kept flag n", kept_n)
    # check accept <=> gate diff > 0 for every SkillAdam step
    viol = 0; tot = 0
    for k in runs:
        if k[1] != "skilladam": continue
        for s in cands(runs[k]):
            g = s["gate"]
            if not g or not g["detail"].get("candidate"): continue
            diff = g["detail"]["candidate"]["n_success"] - g["detail"]["baseline"]["n_success"]
            tot += 1
            if (diff > 0) != bool(s["kept"]): viol += 1
    print("SkillAdam kept <=> candidate-baseline > 0:", tot - viol, "/", tot, "violations", viol)

    # ---------- B. kept rules
    print("\n== B. kept rules")
    # SkillOpt: kept iff dev > stored current (current = best kept so far; seed initially)
    v = 0; t = 0
    for k in runs:
        if k[1] != "skillopt": continue
        d = runs[k]; cur = d["seed_dev"]["n_success"]
        for s in cands(d):
            t += 1
            kept = s["dev"]["n_success"] > cur
            if kept != bool(s["kept"]): v += 1
            if s["kept"]: cur = s["dev"]["n_success"]
    print("SkillOpt kept <=> strictly > stored current:", t - v, "/", t, "violations", v)
    v = 0; t = 0
    for k in runs:
        if k[1] != "gepa": continue
        for s in cands(runs[k]):
            t += 1
            if bool(s["kept"]) != (s["dev"] is not None): v += 1
    print("GEPA kept <=> dev-scored (pool entrant):", t - v, "/", t, "violations", v)
    # GEPA returned = argmax dev, ties to seed then lowest pool index
    for k in sorted(runs):
        if k[1] != "gepa": continue
        d = runs[k]; best = d["seed_dev"]["n_success"]; bid = "c0"
        for s in scored(d):
            if s["dev"]["n_success"] > best: best = s["dev"]["n_success"]; bid = s["candidate_id"]
        assert bid == d["returned_skill_id"], (k, bid, d["returned_skill_id"])
    print("GEPA returned = argmax dev (ties seed/lowest idx): all 9+2 runs OK")
    for k in sorted(runs):
        if k[1] != "skillopt": continue
        d = runs[k]; best = d["seed_dev"]["n_success"]; bid = "seed"
        for s in cands(d):
            if s["dev"]["n_success"] > best: best = s["dev"]["n_success"]; bid = s["candidate_id"]
        assert bid == d["returned_skill_id"], (k, bid, d["returned_skill_id"])
    print("SkillOpt returned = running max: all 9+2 runs OK")

    # ---------- C. 59 dev edges (DeepSeek, excl SkillAdam TB2)
    print("\n== C. dev edges")
    cnt = {}
    for k in DS:
        d = runs[k]
        if k[1] == "skilladam" and k[2] == "terminal-bench-2-1": continue
        for s in scored(d):
            par = next((x for x in d["steps"] if x["candidate_id"] == s["parent_id"]), None)
            if par is None: continue
            if norm(par["skill_text"]) == norm(s["skill_text"]): continue
            cnt[(k[1], "accepted" if s["kept"] else "rejected")] = cnt.get((k[1], "accepted" if s["kept"] else "rejected"), 0) + 1
    print(sorted(cnt.items()), "total", sum(cnt.values()))

    # ---------- D. returned text = tested skill
    print("\n== D. returned text")
    for k in sorted(runs):
        d = runs[k]; rid = d["returned_skill_id"]
        st = next(x for x in d["steps"] if x["candidate_id"] == rid or (rid == "seed" and x["step"] == 0))
        vid = st.get("version_id") or st.get("extra", {}).get("version_id")
        ft = d["final_test"]
        seed_txt = d["steps"][0]["skill_text"]
        isseed = norm(st["skill_text"]) == norm(seed_txt)
        m = re.search(r"evaluations/test/([0-9a-f]{12})-", ft["source"])
        tv12 = m.group(1) if m else None
        print(k[0][:6], k[1][:8], k[2][:6], k[3][:10], "ret", rid, "step vid12", (vid or "")[:12], "test-eval vid12", tv12, "MATCH" if (vid or "")[:12] == tv12 else "DIFF", "| text==seed:", isseed, "words", st["skill_words"])

    # ---------- E. cost rho at first scored candidate
    print("\n== E. rho")
    def rho(s, d): return s["dev"]["mean_input_tokens_per_episode"] / d["seed_dev"]["mean_input_tokens_per_episode"]
    for grp, ks in (("DeepSeek", DS), ("Luna", LU)):
        first = []
        for k in sorted(ks):
            if k[1] == "skilladam": continue
            d = runs[k]; first.append(rho(scored(d)[0], d))
        print(grp, "first-scored rho", [round(x, 2) for x in first], "gm %.3f" % gm(first), "above 1:", sum(x > 1 for x in first), "/", len(first))
    # SkillOpt kept vs rejected (hand-written), gm of rho
    kept = []; rej = []
    for k in HW:
        if k[1] != "skillopt": continue
        d = runs[k]
        for s in cands(d): (kept if s["kept"] else rej).append(rho(s, d))
    print("SkillOpt hand-written kept n", len(kept), "gm %.3f" % gm(kept), "rejected n", len(rej), "gm %.3f" % gm(rej), "ratio %.3f" % (gm(kept) / gm(rej)))
    import random
    random.seed(1)
    bs = []
    lk = [log(x) for x in kept]; lr = [log(x) for x in rej]
    for _ in range(20000):
        a = [random.choice(lk) for _ in lk]; b = [random.choice(lr) for _ in lr]
        bs.append(exp(sum(a) / len(a) - sum(b) / len(b)))
    bs.sort(); print("bootstrap ratio 95%% [%.2f, %.2f] (record: 0.98 [0.83, 1.16])" % (bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]))
    # GEPA gm rho kept
    g = []
    for k in HW:
        if k[1] != "gepa": continue
        d = runs[k]
        for s in scored(d): g.append(rho(s, d))
    print("GEPA hand-written scored n", len(g), "gm rho %.3f (record 1.12)" % gm(g))
    # later-than-best / first for GEPA/SkillOpt
    for opt in ("gepa", "skillopt"):
        ratios = []; ratios2 = []
        for k in HW:
            if k[1] != opt: continue
            d = runs[k]; sc = scored(d) if opt == "gepa" else cands(d)
            seq = [s["dev"]["n_success"] for s in sc]; base = d["seed_dev"]["n_success"]
            best = max(seq)
            if best > base:
                kb = seq.index(best)
                later = [rho(s, d) for s in sc[kb + 1:]]
                if later: ratios.append(gm(later) / rho(sc[kb], d))
            if len(sc) > 1:
                ratios2.append(gm([rho(s, d) for s in sc[1:]]) / rho(sc[0], d))
        print(opt, "later/best rho per run", [round(x, 2) for x in ratios], "gm %.3f" % gm(ratios), "above1", sum(x > 1 for x in ratios), "/", len(ratios), "sign p %.3f" % sign_p(sum(x > 1 for x in ratios), len(ratios)),
              "| later/first", [round(x, 2) for x in ratios2], "gm %.3f" % gm(ratios2), "above1", sum(x > 1 for x in ratios2), "/", len(ratios2), "sign p %.3f" % sign_p(sum(x > 1 for x in ratios2), len(ratios2)))

    # ---------- F. length
    print("\n== F. length")
    for opt in ("gepa", "skillopt", "skilladam"):
        r = []
        for k in HW:
            if k[1] != opt: continue
            d = runs[k]; st = next(x for x in d["steps"] if x["candidate_id"] == d["returned_skill_id"] or (d["returned_skill_id"] == "seed" and x["step"] == 0))
            r.append(st["skill_words"] / d["steps"][0]["skill_words"])
        print(opt, "returned/seed words per run", [round(x, 2) for x in r], "gm %.3f" % gm(r))
    for k in sorted(runs):
        if k[3] == "fromscratch":
            d = runs[k]; st = next(x for x in d["steps"] if x["candidate_id"] == d["returned_skill_id"])
            print(k[1], "near-empty: seed words", d["steps"][0]["skill_words"], "returned", st["skill_words"], "x %.2f" % (st["skill_words"] / d["steps"][0]["skill_words"]))

    # ---------- G. pooled mean final gain
    print("\n== G. pooled mean gains (pts)")
    def final_gain_pts(k):
        d = runs[k]
        if k[1] == "skilladam": return adam_gain[k]
        seq = [s["dev"]["n_success"] for s in (scored(d) if k[1] == "gepa" else cands(d))]
        return 100 * max(0, max(seq) - d["seed_dev"]["n_success"]) / d["dev_panel_size"]
    for opt in ("skillopt", "gepa", "skilladam"):
        for lab, ks in (("DeepSeek", DS), ("Luna", LU), ("both", HW)):
            v = [final_gain_pts(k) for k in ks if k[1] == opt]
            print(opt, lab, "n", len(v), "mean %.1f" % (sum(v) / len(v)), [round(x, 1) for x in v])

    # ---------- H. sign tests
    print("\n== H. sign tests (two-sided exact)")
    for k_, n_ in ((5, 6), (5, 5), (3, 4), (3, 5), (3, 6), (4, 5)):
        print(f"{k_}/{n_}: p = {sign_p(k_, n_):.4f}")
