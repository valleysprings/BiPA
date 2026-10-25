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

    nat_runs=[r for r in runs if "fromscratch" not in r]
    def rel(a,b):
        pa=a.split("__"); pb=b.split("__")
        samem=pa[0]==pb[0]; samo=pa[1]==pb[1]
        return "same_opt_other_model" if samo and not samem else "same_model_other_opt" if samem and not samo else "other_other" if (not samem and not samo) else "same_same"
    out={}
    for meas_name in ("L","P2","P3"):
        Mm=meas[meas_name]
        num={"any":0.0,"same_opt_other_model":0.0,"same_model_other_opt":0.0,"other_other":0.0}; den=0.0
        for i in range(n):
            r=S[i]["run"]
            if "fromscratch" in r: continue
            den+=w[i]
            flags={"any":False,"same_opt_other_model":False,"same_model_other_opt":False,"other_other":False}
            for q in nat_runs:
                if q==r or rb[q]!=rb[r]: continue
                if Mm[i, rid==ridx[q]].any():
                    flags["any"]=True; flags[rel(r,q)]=True
            for k_,v in flags.items(): num[k_]+=w[i]*v
        print(meas_name, {k_: round(100*v/den,1) for k_,v in num.items()}, "(native skills, words", int(den),")")
    print("record all: L any 4.6 same-opt 0.9 same-model 3.7 other 0.2 ; P2 any 11.4 same-opt 4.4 same-model 7.7 other 0.7 ; P3 2.8 / 1.1 / 1.7 / 0.3")
