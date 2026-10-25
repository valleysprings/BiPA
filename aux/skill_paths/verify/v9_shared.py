"""Independent recomputation of mech_shared.md headline numbers from the stored sentence texts (shared/shared_sentences.json = the 638 added sentences with run + text),
with a separate implementation of L (4-gram Jaccard >= 0.05), S (3-gram containment >= 0.5, both >= 6 words), P2/P3 (tf-idf cosine; spec as in the record: stemmed
content unigrams+bigrams, stop words out, idf over the sentences).  The seed's 53 sentences are not in the JSON, so the idf here is over the 638 added sentences only."""
import json, re, math, collections, itertools
from pathlib import Path
from math import comb
import numpy as np
from src.layout import OUT


if __name__ == "__main__":
    SC = OUT / "skill_paths"
    S = json.load(open(SC / "shared/shared_sentences.json"))["sentences"]
    STOP = set("a an the and or of to in on for with by at as is are be been being this that these those it its from into than then so if when while before after not no nor do does did done should must can may will would could also any each every all only more most such other their there which who whom what how why where one two up out about over under again further once".split())
    def stem(w):
        for suf in ("ingly", "edly", "ing", "ies", "ied", "es", "ed", "ly", "s"):
            if len(w) > len(suf) + 3 and w.endswith(suf): return w[:-len(suf)] + ("y" if suf in ("ies", "ied") else "")
        return w
    toks = [re.findall(r"[a-z0-9]+", x["text"].lower()) for x in S]
    n = len(S)
    print("sentences", n, "| stored ntok equals the recomputed token count for", sum(len(t) == x["ntok"] for t, x in zip(toks, S)), "/", n)
    def sh(t, k): return {tuple(t[i:i + k]) for i in range(len(t) - k + 1)}
    s4 = [sh(t, 4) for t in toks]; s3 = [sh(t, 3) for t in toks]
    J = np.zeros((n, n)); C = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = s4[i], s4[j]
            if a and b:
                c = len(a & b)
                if c: J[i, j] = J[j, i] = c / (len(a) + len(b) - c)
            if len(toks[i]) >= 6 and len(toks[j]) >= 6:
                a, b = s3[i], s3[j]
                if a and b:
                    c = len(a & b)
                    if c: C[i, j] = C[j, i] = c / min(len(a), len(b))
    def feats(t):
        c = [stem(w) for w in t if w not in STOP and not w.isdigit()]
        return c + [c[i] + "_" + c[i + 1] for i in range(len(c) - 1)]
    F = [feats(t) for t in toks]; df = collections.Counter()
    for f in F: df.update(set(f))
    vocab = {w: k for k, w in enumerate(sorted(df))}
    M = np.zeros((n, len(vocab)))
    for r, f in enumerate(F):
        for w, c in collections.Counter(f).items(): M[r, vocab[w]] = (1 + math.log(c)) * math.log((1 + n) / (1 + df[w]))
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    P = M @ M.T; np.fill_diagonal(P, 0)
    runs = sorted({x["run"] for x in S}); ridx = {r: i for i, r in enumerate(runs)}
    rid = np.array([ridx[x["run"]] for x in S]); w = np.array([x["ntok"] for x in S], float)
    bench = lambda r: r.split("__")[2]
    rb = {r: bench(r) for r in runs}
    meas = {"L": J >= 0.05, "S": C >= 0.5, "P2": P >= 0.2, "P3": P >= 0.3}
    print("learned runs", len(runs))
    # per-sentence: matches with which runs (other runs, same benchmark)
    def share(mask, which_runs_of=None):
        res = {}
        out = {}
        for k, Mm in meas.items():
            flags = np.zeros(n, bool)
            for i in range(n):
                for r in runs:
                    if r == S[i]["run"] or rb[r] != rb[S[i]["run"]]: continue
                    if Mm[i, rid == ridx[r]].any(): flags[i] = True; break
            out[k] = flags
        return out
    flags = share(None)
    # agreement with stored flags
    for k in meas:
        stored = np.array([x["shared"][k] for x in S])
        print(f"measure {k}: recomputed vs stored flags agree on {int((flags[k] == stored).sum())}/{n} sentences (recomputed shared {int(flags[k].sum())}, stored {int(stored.sum())})")
    def pooled(sel):
        o = {}
        for k in meas: o[k] = 100 * w[sel & flags[k]].sum() / w[sel].sum()
        return o
    allsel = np.ones(n, bool)
    frag = [x for x in S if x["ntok"] < 4]
    print("fragments (<4 words) among stored sentences:", len(frag))
    print("pooled all:", {k: round(v, 1) for k, v in pooled(allsel).items()}, "(record L 5.9 S 0.5 P2 14.2 P3 4.7)")
    for b, rec in (("tau3-bench", (3.9, 0.6, 14.3, 3.4)), ("automationbench", (9.3, 0.6, 19.7, 9.0)), ("terminal-bench-2-1", (5.1, 0.0, 2.8, 0.0))):
        sel = np.array([rb[x["run"]] == b for x in S])
        print(b, {k: round(v, 1) for k, v in pooled(sel).items()}, "record", rec, "| sentences", int(sel.sum()), "words", int(w[sel].sum()))
    # arithmetic of the headline fractions
    print("words total", int(w.sum()), "| 1359/9590 = %.1f%%  619/3147 = %.1f%%" % (100 * 1359 / 9590, 100 * 619 / 3147))
    # chance floor (exact over all partner subsets from other benchmarks)
    def floor(k):
        Mm = meas[k]; num = {b: 0.0 for b in set(rb.values())}; den = {b: 0.0 for b in set(rb.values())}
        nl = collections.Counter(rb.values())
        for i in range(n):
            b = rb[S[i]["run"]]; npart = nl[b] - 1
            others = [r for r in runs if rb[r] != b]; N = len(others)
            m = sum(Mm[i, rid == ridx[r]].any() for r in others)
            pr = 1 - comb(N - m, npart) / comb(N, npart)
            num[b] += w[i] * pr; den[b] += w[i]
        return {b: round(100 * num[b] / den[b], 1) for b in num}, round(100 * sum(num.values()) / sum(den.values()), 1)
    for k in ("L", "P2", "P3"):
        print("chance floor", k, floor(k), "| record all: L 0.7, P2 4.3, P3 1.5; P2 per benchmark tau3 3.4 AutoB 7.7 TB2 0.4")
    # per-skill P2 share (table 1a), a few rows
    rowsrec = {"deepseek-flash__gepa__tau3-bench__native-20261008-https": 19.7, "gpt-6-luna__skilladam__tau3-bench__native-20261012": 42.3, "deepseek-flash__skilladam__automationbench__native-20261008-https": 39.1,
               "gpt-6-luna__gepa__automationbench__fromscratch": 34.8, "deepseek-flash__skillopt__terminal-bench-2-1__native-20261008-https": 2.9, "gpt-6-luna__skilladam__automationbench__native-20261012": 0.0}
    for r, rec in rowsrec.items():
        sel = np.array([x["run"] == r for x in S])
        print("1a P2 share", r[:62], "mine %.1f" % (100 * w[sel & flags["P2"]].sum() / w[sel].sum()), "record", rec)
    # AutoB native-only pooled P2 (drop the two fromscratch skills from partner pools and from the numerators)
    nat = np.array([("fromscratch" not in x["run"]) for x in S])
    def share_native(k, b):
        Mm = meas[k]; num = den = 0.0
        for i in range(n):
            if not nat[i] or rb[S[i]["run"]] != b: continue
            den += w[i]; ok = False
            for r in runs:
                if r == S[i]["run"] or rb[r] != b or "fromscratch" in r: continue
                if Mm[i, rid == ridx[r]].any(): ok = True; break
            num += w[i] * ok
        return 100 * num / den
    print("AutoB native only (6 skills): L %.1f S %.1f P2 %.1f P3 %.1f (record 5.8 / 0.0 / 11.3 / 3.6)" % tuple(share_native(k, "automationbench") for k in ("L", "S", "P2", "P3")))
    # contribution of scratch skills: AutoB P2 share when the 6 native skills may use scratch partners vs not
    sel = np.array([("fromscratch" not in x["run"]) and rb[x["run"]] == "automationbench" for x in S])
    print("AutoB native skills' P2 share with all 7 partners %.1f vs native partners only %.1f" % (100 * w[sel & flags["P2"]].sum() / w[sel].sum(), share_native("P2", "automationbench")))
    # pair overlap control, P2
    def words_with_counterpart(a, b, Mm):
        ia = [i for i in range(n) if S[i]["run"] == a]; ib = np.where(rid == ridx[b])[0]
        return sum(w[i] for i in ia if Mm[i, ib].any())
    def pair_overlap(a, b, k):
        Mm = meas[k]; wa = w[rid == ridx[a]].sum(); wb = w[rid == ridx[b]].sum()
        return 100 * (words_with_counterpart(a, b, Mm) + words_with_counterpart(b, a, Mm)) / (wa + wb)
    nativ = [r for r in runs if "fromscratch" not in r]
    cls = collections.defaultdict(list)
    for a, b in itertools.combinations(nativ, 2):
        pa = a.split("__"); pb = b.split("__")
        sameb = pa[2] == pb[2]; samem = pa[0] == pb[0]; samo = pa[1] == pb[1]
        if sameb:
            key = ("same opt, other model" if samo and not samem else "same model, other opt" if samem and not samo else "other model, other opt" if (not samem and not samo) else None)
            if key: cls[key].append((pair_overlap(a, b, "P2"), a, b))
    for key, v in cls.items():
        vals = [x[0] for x in v]
        print(f"pair overlap P2 same benchmark, {key}: pairs {len(v)} mean {np.mean(vals):.1f} [min {min(vals):.1f}, max {max(vals):.1f}] ; mean without the top pair {np.mean(sorted(vals)[:-1]):.1f}")
    print("record: same opt other model 6 pairs 2.5 [0.0, 9.1]; same model other opt 15 pairs 4.3 [0.0, 20.9]; other/other 12 pairs 0.4 [0.0, 2.0]")
