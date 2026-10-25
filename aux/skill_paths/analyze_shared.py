#!/usr/bin/env python
"""What is shared between the learned skills of independent optimizer runs, and what is idiosyncratic -> OUT/data/mech/mech_shared.md,
OUT/skill_paths/shared/*.json, OUT/skill_paths/fig/mech_shared.{pdf,png}.

Idempotent and read-only on the inputs. Inputs: OUT/skill_paths/paths/*.json (written by extract_{gepa,skillopt,skilladam}.py), the two seed skills,
the AutoBench task files under harness-bench/assets/benchmarks/automationbench/tasks (dev + test panels, read-only), the hand labels soft_labels.json next to
this script and, for the keyword categories and the audit functions, src/analysis/skilltext.py and aux/contamination_and_loading/cheatsheet.py.

    python -m aux.skill_paths.analyze_shared [--no-fig] [--skip-verify-dedupe]

What is measured (details and every threshold in the record):
  unit      = skilltext.units() logical unit (bullet, numbered item, table row, wrapped paragraph); headings, fences, separators, frontmatter are structure and are
              not text (as in skilltext.py). A unit is ADDED when one of its lines is inserted/replaced against the seed the run started from
              (difflib on non-empty stripped lines, skilltext.added_line_set logic); verbatim seed sentences inside a partly-new unit are removed.
  sentence  = a unit split at sentence ends (abbreviation-safe), markdown markers stripped, table cells joined. Sentences with < 4 normalized words are FRAGMENTS (no
              4-gram): kept out of every denominator and counted separately. Exact duplicates inside one skill count once.
  weight    = normalized words of the sentence (lowercase [a-z0-9]+ tokens, as cheatsheet.WORD).
  similarity (three ways of calling two sentences "the same"):
      L  lenient verbatim:   word 4-gram shingle Jaccard >= TAU_L
      S  strict verbatim:    exact word 3-gram containment of the shorter sentence in the other >= TAU_S, shorter sentence >= 6 words
      P2/P3 paraphrase-level: tf-idf cosine (stemmed content unigrams+bigrams) >= TAU_P2 / TAU_P3 (thresholds set against hand labels, see shared/soft_labels.json)
"""
import argparse
import collections
import difflib
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from src import layout  # noqa: E402

SCR = layout.OUT / "skill_paths"
PATHS = SCR / "paths"
OUTDIR = SCR / "shared"
FIGDIR = SCR / "fig"
RECORD = layout.OUT / "data/mech/mech_shared.md"
SEED_HAND = layout.SEED_FILE
SEED_MIN = layout.ASSETS / "skills/seed-minimal/SKILL.md"
EX = layout.EXECUTION
HB = layout.HARNESS_BENCH
LABELS = Path(__file__).with_name("soft_labels.json")
os.environ.setdefault("MPLCONFIGDIR", str(SCR / ".mplconfig"))

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------------------
# thresholds (justified in the record, section 2)
# ---------------------------------------------------------------------------------------------------------------------------------
TAU_L = 0.05          # L: 4-gram Jaccard, lenient verbatim (>= about one shared 4-word phrase in ordinary sentences)
TAU_S = 0.50          # S: exact 3-gram containment of the shorter sentence in the other, strict verbatim
S_MIN_WORDS = 6
TAU_P2 = 0.20         # P2: tf-idf cosine, paraphrase level, lenient (hand-labelled precision about 80% same rule, see soft_labels.json)
TAU_P3 = 0.30         # P3: tf-idf cosine, paraphrase level, strict
SWEEP_J = (0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
SWEEP_C = (0.30, 0.40, 0.50, 0.60, 0.80)
SWEEP_P = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
MIN_TOK = 4
BENCHES = ("tau3-bench", "automationbench", "terminal-bench-2-1")
BSHORT = {"tau3-bench": "tau3", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}
OPTS = ("gepa", "skillopt", "skilladam")
OLABEL = {"gepa": "GEPA", "skillopt": "SkillOpt", "skilladam": "SkillAdam"}
MLABEL = {"deepseek-flash": "DeepSeek", "gpt-6-luna": "Luna"}
OC = {"GEPA": "#009E73", "SkillOpt": "#0072B2", "SkillAdam": "#D55E00"}
RNG_SEED = 20261023


# ---------------------------------------------------------------------------------------------------------------------------------
# the project's own keyword coding and text audit
# ---------------------------------------------------------------------------------------------------------------------------------
from src.analysis.skilltext import CATS, CAT_IDS, classify, units  # noqa: E402
from aux.contamination_and_loading import cheatsheet  # noqa: E402

CHEAT = vars(cheatsheet)
CAT_LABEL = dict(CATS)
WORD = CHEAT["WORD"]

# ---------------------------------------------------------------------------------------------------------------------------------
# text -> units -> sentences
# ---------------------------------------------------------------------------------------------------------------------------------
ABBR = ("e.g.", "i.e.", "etc.", "vs.", "approx.", "incl.", "No.")
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'`(\[*✅❌])")


def toks(s):
    return WORD.findall(s.lower())


def nonempty(text):
    return [(i, l.strip()) for i, l in enumerate(text.splitlines()) if l.strip()]


def added_idx(base, text):
    """Raw line indices of `text` that are inserted/replaced against `base` (difflib on non-empty stripped lines, as the path files' added_lines)."""
    a = [l for _, l in nonempty(base)]
    bi = nonempty(text)
    b = [l for _, l in bi]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            out.update(bi[j][0] for j in range(j1, j2))
    return out


def added_lines(base, text):
    """The added lines (non-empty, stripped), same convention as the path files."""
    a = [l for _, l in nonempty(base)]
    b = [l for _, l in nonempty(text)]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            out += b[j1:j2]
    return out


def clean_unit(u):
    s = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|>\s*|[✅❌]\s*)", "", u)
    if s.count("|") >= 2:                       # markdown table row -> cells joined
        s = " ".join(c.strip() for c in s.strip().strip("|").split("|") if c.strip())
    return s


def split_sentences(s):
    for i, a in enumerate(ABBR):
        s = s.replace(a, f"\x00{i}\x00")
    out = []
    for p in SPLIT.split(s):
        for i, a in enumerate(ABBR):
            p = p.replace(f"\x00{i}\x00", a)
        p = p.strip()
        if p:
            out.append(p)
    return out


def text_sentences(text):
    """All rule-unit sentences of a skill text with the raw line indices of their unit: [(sentence, lines)]."""
    out = []
    for u in units(text):
        if u["kind"] != "rule":
            continue
        for s in split_sentences(clean_unit(u["text"])):
            out.append((s, u["lines"]))
    return out


def norm(s):
    return " ".join(toks(s))


def added_sentence_list(text, base):
    """Sentences of the ADDED units of `text` against `base`; verbatim sentences of `base` removed. Returns list of raw sentences (document order)."""
    idx = added_idx(base, text)
    base_norm = {norm(s) for s, _ in text_sentences(base)}
    out = []
    for u in units(text):
        if u["kind"] != "rule" or not any(i in idx for i in u["lines"]):
            continue
        for s in split_sentences(clean_unit(u["text"])):
            if norm(s) in base_norm:
                continue
            out.append(s)
    return out


def make_records(sent_list):
    """Dedupe (normalized) inside one skill; split fragments from real sentences."""
    seen, recs, dup = set(), [], 0
    for s in sent_list:
        n = norm(s)
        if not n:
            continue
        if n in seen:
            dup += 1
            continue
        seen.add(n)
        t = n.split()
        recs.append({"text": s, "norm": n, "ntok": len(t), "frag": len(t) < MIN_TOK, "cat": classify(s)})
    return recs, dup


# ---------------------------------------------------------------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------------------------------------------------------------
class Run:
    pass


def load_runs():
    seed_hand, seed_min = SEED_HAND.read_text(), SEED_MIN.read_text()
    runs = []
    for f in sorted(PATHS.glob("*.json")):
        d = json.loads(f.read_text())
        r = Run()
        r.key = f.stem
        r.d = d
        r.model, r.opt, r.bench, r.run_id = d["model"], d["optimizer"], d["benchmark"], d["run_id"]
        r.scratch = d["seed"] == "near-empty"
        r.partial = d["partial"]
        r.base = seed_min if r.scratch else seed_hand
        by = {s["candidate_id"]: s for s in d["steps"]}
        r.by = by
        r.final = by[d["returned_skill_id"]]["skill_text"]
        r.returned_seed = r.final == r.base
        r.label = f"{MLABEL[r.model]} {OLABEL[r.opt]}" + (" (scratch)" if r.scratch else "")
        runs.append(r)
    return runs, seed_hand, seed_min


def validate(runs, seed_hand, seed_min):
    """Sanity checks of the path files this analysis relies on. Returns (lines, ok)."""
    lines, ok = [], True

    def chk(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        lines.append(("OK   " if cond else "FAIL ") + msg)

    chk(len(runs) == 20, f"{len(runs)} path files (expected 20: 18 Table-1 runs + 2 fromscratch)")
    for r in runs:
        d = r.d
        st = d["steps"]
        chk(st[0]["event"] == "seed" and st[0]["skill_text"] == r.base, f"{r.key}: step 0 text equals the {'near-empty' if r.scratch else 'hand-written'} seed file")
        chk(all(s["skill_text"] for s in st), f"{r.key}: every step has skill_text ({len(st)} steps)")
        chk(all(s["parent_id"] in r.by for s in st[1:]), f"{r.key}: every candidate's parent_id is a step of the run")
        # recomputed added lines vs the path file's added_lines (extractor cross-check)
        bad = 0
        for s in st[1:]:
            if added_lines(r.by[s["parent_id"]]["skill_text"], s["skill_text"]) != s["added_lines"]:
                bad += 1
        chk(bad == 0, f"{r.key}: added_lines recomputed from skill_text equal the path file's for all candidates ({bad} differ)")
        sm = EX / r.model / r.opt / r.bench / r.run_id / "summary.json"
        try:
            fin = json.loads(sm.read_text())["final_skill"]
            chk(fin == r.final, f"{r.key}: returned skill text == summary.json final_skill (mtime {sm.stat().st_mtime:.0f})")
        except Exception as e:  # noqa: BLE001
            chk(False, f"{r.key}: summary.json not readable ({type(e).__name__})")
    return lines, ok


# ---------------------------------------------------------------------------------------------------------------------------------
# similarity machinery
# ---------------------------------------------------------------------------------------------------------------------------------
STOP = set("a an the and or of to in on for with by at as is are be been being this that these those it its from into than then so if when while before after not no nor "
           "do does did done should must can may will would could also any each every all only more most such other their there which who whom what how why where one two "
           "up out about over under again further once".split())


def stem(w):
    for suf in ("ingly", "edly", "ing", "ies", "ied", "es", "ed", "ly", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:-len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


def feats(t):
    c = [stem(w) for w in t if w not in STOP and not w.isdigit()]
    return c + [c[i] + "_" + c[i + 1] for i in range(len(c) - 1)]


def shingles(t, n):
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


class Sim:
    """Pairwise similarity of a list of sentence records (each with 'norm' and 'ntok'): J4, C3 (containment, with the 6-word rule) and tf-idf cosine."""

    def __init__(self, recs):
        self.n = len(recs)
        t = [r["norm"].split() for r in recs]
        s4 = [shingles(x, 4) for x in t]
        s3 = [shingles(x, 3) for x in t]
        n = self.n
        self.J = np.zeros((n, n))
        self.C = np.zeros((n, n))
        inv4, inv3 = collections.defaultdict(list), collections.defaultdict(list)
        for i in range(n):
            for g in s4[i]:
                inv4[g].append(i)
            for g in s3[i]:
                inv3[g].append(i)
        for i in range(n):
            cnt = collections.Counter(j for g in s4[i] for j in inv4[g])
            for j, c in cnt.items():
                self.J[i, j] = c / (len(s4[i]) + len(s4[j]) - c)
            cnt = collections.Counter(j for g in s3[i] for j in inv3[g])
            for j, c in cnt.items():
                if len(t[i]) >= S_MIN_WORDS and len(t[j]) >= S_MIN_WORDS:
                    self.C[i, j] = c / min(len(s3[i]), len(s3[j]))
        F = [feats(x) for x in t]
        df = collections.Counter()
        for f in F:
            df.update(set(f))
        vocab = {w: k for k, w in enumerate(sorted(df))}
        M = np.zeros((n, len(vocab)))
        for r, f in enumerate(F):
            for w, c in collections.Counter(f).items():
                M[r, vocab[w]] = (1 + math.log(c)) * math.log((1 + n) / (1 + df[w]))
        M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
        self.P = M @ M.T
        for a in (self.J, self.C, self.P):
            np.fill_diagonal(a, 0.0)

    def match(self, kind, tau=None):
        if kind == "L":
            return self.J >= (TAU_L if tau is None else tau)
        if kind == "S":
            return self.C >= (TAU_S if tau is None else tau)
        if kind == "P2":
            return self.P >= (TAU_P2 if tau is None else tau)
        if kind == "P3":
            return self.P >= (TAU_P3 if tau is None else tau)
        raise KeyError(kind)


KINDS = ("L", "S", "P2", "P3")
KIND_NAME = {"L": f"L: 4-gram Jaccard >= {TAU_L}", "S": f"S: 3-gram containment >= {TAU_S} (>= {S_MIN_WORDS} words)",
             "P2": f"P2: tf-idf cosine >= {TAU_P2}", "P3": f"P3: tf-idf cosine >= {TAU_P3}"}


# ---------------------------------------------------------------------------------------------------------------------------------
# the F-level universe: final-skill added sentences of every run + the hand-written seed's sentences
# ---------------------------------------------------------------------------------------------------------------------------------
def build_universe(runs, seed_hand):
    recs = []
    for r in runs:
        rr, dup = make_records(added_sentence_list(r.final, r.base))
        r.dup_removed = dup
        r.n_frag = sum(1 for x in rr if x["frag"])
        r.frag_words = sum(x["ntok"] for x in rr if x["frag"])
        for x in rr:
            if not x["frag"]:
                recs.append({**x, "owner": r.key, "bench": r.bench})
    seed_recs, _ = make_records([s for s, _ in text_sentences(seed_hand)])
    for x in seed_recs:
        if not x["frag"]:
            recs.append({**x, "owner": "SEED", "bench": "SEED"})
    return recs


# ---------------------------------------------------------------------------------------------------------------------------------
# context: arrays over the universe
# ---------------------------------------------------------------------------------------------------------------------------------
class Ctx:
    """Index structures over a universe of sentence records (owner key, bench, words, category)."""

    def __init__(self, U, runs, drop=None, keep=None):
        self.U = U
        self.n = len(U)
        cand = [r for r in runs if not (drop and drop(r))]
        idx = [np.array([i for i, u in enumerate(U) if u["owner"] == r.key and (keep is None or keep[i])], int) for r in cand]
        pairs = [(r, ix) for r, ix in zip(cand, idx) if len(ix)]                       # runs with at least one (kept) sentence
        self.runs = [r for r, _ in pairs]
        self.run_idx = [ix for _, ix in pairs]
        self.rk = {r.key: q for q, r in enumerate(self.runs)}
        self.words = np.array([u["ntok"] for u in U], float)
        self.cat = [u["cat"] for u in U]
        self.seed_idx = np.array([i for i, u in enumerate(U) if u["owner"] == "SEED"], int)
        self.bench = [r.bench for r in self.runs]
        self.W = [float(self.words[ix].sum()) for ix in self.run_idx]

    def others_same_bench(self, q):
        return [p for p in range(len(self.runs)) if p != q and self.bench[p] == self.bench[q]]

    def pool_other_bench(self, q):
        return [p for p in range(len(self.runs)) if self.bench[p] != self.bench[q]]


def profile(M, ctx, floor=True, max_subsets=3000):
    """Shared share of every run (same-benchmark pool = all other learned runs on its benchmark), the matched-pool chance floor (same number of other runs drawn from
    OTHER benchmarks, mean over all or up to max_subsets of the subsets), the pairwise pooled overlap, and per-sentence flags."""
    n, R = ctx.n, len(ctx.runs)
    A = np.zeros((n, R), bool)
    for q, ix in enumerate(ctx.run_idx):
        A[:, q] = M[:, ix].any(axis=1)
    res = {"runs": [], "pair": np.zeros((R, R)), "A": A}
    rng = random.Random(RNG_SEED)
    for q, ix in enumerate(ctx.run_idx):
        oth = ctx.others_same_bench(q)
        w = ctx.words[ix]
        flag = A[np.ix_(ix, oth)].any(axis=1) if oth else np.zeros(len(ix), bool)
        row = {"q": q, "n_others": len(oth), "n_sent": len(ix), "words": float(w.sum()), "flag": flag,
               "sent_shared": int(flag.sum()), "words_shared": float(w[flag].sum()),
               "share_words": float(w[flag].sum() / w.sum()), "share_sent": float(flag.mean())}
        if floor:
            pool = ctx.pool_other_bench(q)
            k = min(len(oth), len(pool))
            vals = []
            if k:
                Aii = A[np.ix_(ix, pool)]
                subs = list(itertools.combinations(range(len(pool)), k)) if math.comb(len(pool), k) <= max_subsets else \
                    [tuple(sorted(rng.sample(range(len(pool)), k))) for _ in range(max_subsets)]
                for sub in subs:
                    vals.append(w[Aii[:, list(sub)].any(axis=1)].sum() / w.sum())
            row["floor_words"] = float(np.mean(vals)) if vals else None
            row["floor_k"] = k
            row["floor_pool"] = len(pool)
        res["runs"].append(row)
    for a in range(R):
        for b in range(R):
            if a == b:
                continue
            ia, ib = ctx.run_idx[a], ctx.run_idx[b]
            sa = ctx.words[ia][A[ia, b]].sum()
            sb = ctx.words[ib][A[ib, a]].sum()
            res["pair"][a, b] = (sa + sb) / (ctx.W[a] + ctx.W[b])
    return res


def cat_composition(ctx, prof):
    """Words by category of shared and unique sentences, per run and pooled over runs."""
    per, pool = [], {"shared": collections.Counter(), "unique": collections.Counter()}
    for row in prof["runs"]:
        ix = ctx.run_idx[row["q"]]
        sh, un = collections.Counter(), collections.Counter()
        for i, f in zip(ix, row["flag"]):
            (sh if f else un)[ctx.cat[i]] += ctx.words[i]
        per.append({"shared": dict(sh), "unique": dict(un)})
        pool["shared"].update(sh)
        pool["unique"].update(un)
    return per, {k: dict(v) for k, v in pool.items()}


# ---------------------------------------------------------------------------------------------------------------------------------
# pair classes (control 4)
# ---------------------------------------------------------------------------------------------------------------------------------
PAIR_CLASSES = ("same optimizer, other model", "same model, other optimizer", "other model, other optimizer", "same optimizer, same model")


def pair_class(ra, rb):
    if ra.scratch or rb.scratch:
        return None
    so, sm = ra.opt == rb.opt, ra.model == rb.model
    if so and sm:
        return "same optimizer, same model"
    if so:
        return "same optimizer, other model"
    if sm:
        return "same model, other optimizer"
    return "other model, other optimizer"


def scratch_pair_class(ra, rb):
    """Pairs with a near-empty-seed run (always same benchmark here): scratch vs native by what they share."""
    s = ra if ra.scratch else rb
    o = rb if ra.scratch else ra
    if s.scratch and o.scratch:
        return "scratch vs scratch (other optimizer)"
    parts = ["scratch vs native"]
    parts.append("same optimizer" if s.opt == o.opt else "other optimizer")
    parts.append("same model" if s.model == o.model else "other model")
    return ", ".join(parts)


def pair_table(ctx, prof):
    """Group the pairwise pooled overlaps by pair class, within the benchmark and across benchmarks. Returns dict class -> {'within': [(a,b,v)], 'cross': [...]}."""
    R = len(ctx.runs)
    tab = {c: {"within": [], "cross": []} for c in PAIR_CLASSES}
    scr = collections.defaultdict(list)
    for a in range(R):
        for b in range(a + 1, R):
            ra, rb = ctx.runs[a], ctx.runs[b]
            v = float(prof["pair"][a, b])
            same_b = ra.bench == rb.bench
            if ra.scratch or rb.scratch:
                if same_b:
                    scr[scratch_pair_class(ra, rb)].append((ra.key, rb.key, v))
                continue
            tab[pair_class(ra, rb)]["within" if same_b else "cross"].append((ra.key, rb.key, v))
    return tab, dict(scr)


REL_CLASSES = ("same optimizer, other model", "same model, other optimizer", "other model, other optimizer")


def relation_decomp(ctx, prof):
    """Which partners carry the sharing (control 4, per sentence instead of per pair). For every native learned skill: the words that have a counterpart in a same-benchmark
    NATIVE partner of each relation class (same optimizer / other model, same model / other optimizer, other model AND other optimizer), the words with a counterpart in any
    native partner, and the size-matched chance floor of each class (the same number of partners of that class drawn from the other benchmarks, mean over all subsets)."""
    A = prof["A"]
    per = []
    for q, r in enumerate(ctx.runs):
        if r.scratch:
            continue
        ix = ctx.run_idx[q]
        w = ctx.words[ix]
        same = {c: [p for p, rp in enumerate(ctx.runs) if p != q and not rp.scratch and rp.bench == r.bench and pair_class(r, rp) == c] for c in REL_CLASSES}
        oth = {c: [p for p, rp in enumerate(ctx.runs) if p != q and not rp.scratch and rp.bench != r.bench and pair_class(r, rp) == c] for c in REL_CLASSES}
        row = {"q": q, "key": r.key, "bench": r.bench, "words": float(w.sum()), "k": {c: len(same[c]) for c in REL_CLASSES}, "shared": {}, "floor": {}}
        anyf = np.zeros(len(ix), bool)
        for c in REL_CLASSES:
            f = A[np.ix_(ix, same[c])].any(axis=1) if same[c] else np.zeros(len(ix), bool)
            anyf |= f
            row["shared"][c] = float(w[f].sum())
            if same[c] and oth[c]:
                k2 = min(len(same[c]), len(oth[c]))
                row["floor"][c] = float(np.mean([w[A[np.ix_(ix, list(sub))].any(axis=1)].sum() for sub in itertools.combinations(oth[c], k2)]))
            else:
                row["floor"][c] = None
        row["any"] = float(w[anyf].sum())
        per.append(row)
    return per


def decomp_pooled(per, bench=None):
    """Pooled over the skills that have at least one partner of the class: (observed share, chance floor share, number of skills)."""
    out = {}
    for c in REL_CLASSES + ("any",):
        rows = [x for x in per if (bench is None or x["bench"] == bench) and (c == "any" or x["k"][c] > 0)]
        den = sum(x["words"] for x in rows)
        if not rows or not den:
            out[c] = None
            continue
        num = sum(x["any"] if c == "any" else x["shared"][c] for x in rows)
        fl = None if c == "any" else sum(x["floor"][c] for x in rows if x["floor"][c] is not None) / den
        out[c] = (num / den, fl, len(rows))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# overlap with the hand-written seed (question 2)
# ---------------------------------------------------------------------------------------------------------------------------------
def seed_overlap(M, ctx):
    out = []
    for q, ix in enumerate(ctx.run_idx):
        flag = M[np.ix_(ix, ctx.seed_idx)].any(axis=1)
        w = ctx.words[ix]
        out.append({"q": q, "flag": flag, "words_echo": float(w[flag].sum()), "share_words": float(w[flag].sum() / w.sum()),
                    "sent_echo": int(flag.sum()), "share_sent": float(flag.mean())})
    return out


def closest_to_seed(sim, ctx, q, k=10, by="P"):
    """The k sentence pairs (run sentence, seed sentence) with the highest similarity `by` (P cosine or J Jaccard; ties on the other)."""
    ix = ctx.run_idx[q]
    best = []
    for i in ix:
        sj = ctx.seed_idx
        key = (sim.P[i, sj], sim.J[i, sj]) if by == "P" else (sim.J[i, sj], sim.P[i, sj])
        order = sorted(range(len(sj)), key=lambda t: (key[0][t], key[1][t]), reverse=True)[0]
        j = sj[order]
        best.append((float(sim.P[i, j]), float(sim.J[i, j]), float(sim.C[i, j]), int(i), int(j)))
    best.sort(key=(lambda x: (x[0], x[1])) if by == "P" else (lambda x: (x[1], x[0])), reverse=True)
    return best[:k]


# ---------------------------------------------------------------------------------------------------------------------------------
# task specificity of the AutoBench skills (question 3): corpora of the dev / test / train task files, method of cheatsheet.py
# ---------------------------------------------------------------------------------------------------------------------------------
AB = "automationbench"
AB_TASKS = HB / "assets/benchmarks" / AB / "tasks"
AMBIGUOUS_SERVICES = {"wave", "monday", "buffer", "meta", "notion"}


def file_texts(root):
    """Texts of the decodable non-binary files under a directory, same filter as cheatsheet.read_dir_text (which joins them with newlines)."""
    out = []
    if not root.is_dir():
        return out
    skip, mx = CHEAT["SKIP_SUFFIX"], CHEAT["MAX_FILE"]
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix.lower() in skip or f.stat().st_size > mx:
            continue
        try:
            out.append(f.read_text())
        except (UnicodeDecodeError, ValueError):
            continue
    return out


def panel_ids():
    ids = {}
    for model, tag in (("deepseek-flash", "native-20261008-https"), ("gpt-6-luna", "native-20261012")):
        proto = json.loads((EX / model / "skillopt" / AB / tag / "native_protocol.json").read_text())
        for k in ("sel", "test", "train"):
            lst = [r if isinstance(r, str) else r["task_id"] for r in proto["panels"][k]]
            if k in ids:
                assert ids[k] == lst, f"panel {k} differs between models"
            ids[k] = lst
    return {"dev": ids["sel"], "test": ids["test"], "train": ids["train"]}


def load_ab(ids_by_panel):
    """Per task: prompt (instruction.md + task.toml), env (environment/**), answer (solution/** + tests/**), lowercased; plus the deduplicated corpus text of a panel."""
    all_ids = sorted(p.name for p in AB_TASKS.iterdir() if p.is_dir())
    read_file = CHEAT["read_file"]
    tasks = {}
    configs = {}
    for panel, ids in ids_by_panel.items():
        for t in ids:
            if t in tasks:
                continue
            d = AB_TASKS / t
            prompt = "\n".join(x for x in (read_file(d / "instruction.md"), read_file(d / "task.toml")) if x)
            env_files = file_texts(d / "environment")
            ans_files = file_texts(d / "solution") + file_texts(d / "tests")
            tasks[t] = {"prompt": prompt, "env_files": env_files, "ans_files": ans_files}
            try:
                configs[t] = json.loads(read_file(d / "environment/agent_config.json"))
            except ValueError:
                configs[t] = {}
    return all_ids, tasks, configs


def corpus_for(ids, tasks):
    """bench_data in the format of cheatsheet.load_bench for a panel (identical file contents counted once in the corpus text)."""
    words, ngrams = CHEAT["words"], CHEAT["ngrams"]
    per_task, seen = {}, set()
    prompt_all, answer_all = [], []
    for t in ids:
        x = tasks[t]
        answer = "\n".join(x["ans_files"])
        per_task[t] = {"prompt": x["prompt"], "answer": answer}
        prompt_all.append(x["prompt"])
        for blob in x["ans_files"]:
            h = hashlib.sha1(blob.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                answer_all.append(blob)
    corp = {"prompt": "\n".join(prompt_all), "answer": "\n".join(answer_all)}
    sets = {kind: {n: ngrams(words(text), n) for n in (8, 13)} for kind, text in corp.items()}
    return {"per_task": per_task, "text": corp, "ngrams": sets}


CODE_SPAN = re.compile(r"`([^`]+)`")
SNAKE = re.compile(r"(?<![\w-])[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+(?![\w-])")
CAMEL = re.compile(r"(?<![\w-])(?:[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)(?![\w-])")
FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]+")


def service_lexicon(configs, ids):
    services, tools, fields = set(), set(), set()

    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)

    for t in ids:
        c = configs.get(t) or {}
        services.update(c.get("allowed_services") or [])
        services.update(k for k in (c.get("initial_state") or {}) if k != "meta")
        tools.update(c.get("zapier_tools") or [])
        fields.update(k for k in keys(c.get("initial_state") or {}) if "_" in k)
    services = {s for s in services if s not in AMBIGUOUS_SERVICES}
    return services, tools, fields


def sentence_tokens(sent, services, tools, fields):
    """Concrete-name candidates of one sentence: [(lowercased token, type)]. Types: service, tool, field, path, flag, identifier, code, number."""
    PATH_RE, VALUE_RE = CHEAT["PATH_RE"], CHEAT["VALUE_RE"]
    out = {}

    def add(tok, typ):
        tok = tok.strip().strip(".,:;)(\"'[]{}").lower()
        if len(tok) >= 4 and tok not in out:
            out[tok] = typ

    for m in PATH_RE.finditer(sent):
        w = m.group(0).strip().strip(".,:;)(\"'[]{}")
        if w.startswith(("/", "./")) or w.count("/") >= 2 or re.search(r"\.[A-Za-z]{1,5}$", w):      # a real path or file name, not a slash-joined pair of words
            add(w, "path")
    for m in FLAG.finditer(sent):
        add(m.group(0), "flag")
    for m in SNAKE.finditer(sent):
        w = m.group(0)
        add(w, "tool" if w in tools else "field" if w.lower() in {f.lower() for f in fields} else "identifier")
    for m in CAMEL.finditer(sent):
        add(m.group(0), "tool" if m.group(0) in tools else "identifier")
    for m in VALUE_RE.finditer(sent):
        add(m.group(0), "number")
    for m in CODE_SPAN.finditer(sent):
        for piece in m.group(1).split():
            add(piece, "code")
    low = sent.lower()
    for s in sorted(services):
        pat = r"(?<![a-z0-9_])" + s.replace("_", r"[ _]") + r"(?![a-z0-9_])"
        if re.search(pat, low):
            add(s, "service")
    return out


def ground(tok, tasks, ids):
    """Number of tasks of a panel whose files contain the token (case-insensitive substring), split by corpus."""
    n_vis = n_ans = n_any = 0
    for t in ids:
        x = tasks[t]
        if "low" not in x:
            x["low"] = {"vis": (x["prompt"] + "\n" + "\n".join(x["env_files"])).lower(), "ans": "\n".join(x["ans_files"]).lower()}
        v = tok in x["low"]["vis"]
        a = tok in x["low"]["ans"]
        n_vis += v
        n_ans += a
        n_any += v or a
    return {"df": n_any, "df_vis": n_vis, "df_ans": n_ans}


def classify_specificity(sentences, tasks, panels, lex):
    """Tag every sentence: concrete tokens, grounded in dev/test files, task-specific (<= 3 tasks of dev+test), answer-only. Returns list of dicts."""
    services, tools, fields = lex
    union = panels["dev"] + panels["test"]
    cache = {}
    out = []
    for s in sentences:
        toks_ = sentence_tokens(s, services, tools, fields)
        info = []
        for tok, typ in toks_.items():
            if tok not in cache:
                cache[tok] = {"union": ground(tok, tasks, union), "dev": ground(tok, tasks, panels["dev"]), "test": ground(tok, tasks, panels["test"]),
                              "train": ground(tok, tasks, panels["train"])}
            g = cache[tok]
            info.append({"tok": tok, "type": typ, **g["union"], "df_dev": g["dev"]["df"], "df_test": g["test"]["df"], "df_train": g["train"]["df"]})
        concrete = bool(info)
        grounded = [x for x in info if x["df"] > 0]
        specific = [x for x in grounded if x["df"] <= 3]
        answer_only = [x for x in grounded if x["df_vis"] == 0 and x["df_ans"] > 0]
        if specific:
            cls = "task-specific"
        elif grounded:
            cls = "environment-wide"
        elif concrete:
            cls = "named, not in dev/test files"
        else:
            cls = "no concrete name"
        out.append({"text": s, "tokens": info, "class": cls, "n_answer_only": len(answer_only),
                    "train_specific": [x["tok"] for x in info if 0 < x["df_train"] <= 3 and x["df"] == 0]})
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# proposal-level universe: every candidate's added sentences (kept or not), de-duplicated inside a run
# ---------------------------------------------------------------------------------------------------------------------------------
def build_universe_P(runs):
    recs, info = [], {}
    for r in runs:
        seen, n_cand = set(), 0
        for s in r.d["steps"]:
            if s["event"] != "candidate":
                continue
            n_cand += 1
            parent = r.by[s["parent_id"]]["skill_text"]
            for sent in added_sentence_list(s["skill_text"], parent):
                n = norm(sent)
                if not n or len(n.split()) < MIN_TOK or n in seen:
                    continue
                seen.add(n)
                recs.append({"text": sent, "norm": n, "ntok": len(n.split()), "frag": False, "cat": classify(sent), "owner": r.key, "bench": r.bench})
        info[r.key] = {"n_candidates": n_cand, "n_sent": len(seen)}
    return recs, info


# ---------------------------------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------------------------------
def pct(x, d=1):
    return "-" if x is None else f"{100 * x:.{d}f}"


def md_table(head, rows):
    lines = ["| " + " | ".join(str(h) for h in head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def top_cats(counter, k=2):
    tot = sum(counter.values())
    if not tot:
        return "-"
    items = sorted(counter.items(), key=lambda kv: -kv[1])[:k]
    return ", ".join(f"{c} {100 * v / tot:.0f}%" for c, v in items)


def short_label(r):
    return f"{BSHORT[r.bench]} {r.label}"


def clip(s, n=150):
    s = s.replace("|", "/").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "..."


def cal_table(sim, U, labels):
    """Hand labels -> per cosine band counts."""
    idx = {}
    for i, u in enumerate(U):
        idx.setdefault(u["norm"], i)
    bands = [(0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.40), (0.40, 9.0)]
    rows, missing = [], 0
    cnt = {b: collections.Counter() for b in bands}
    for lab in labels:
        if lab["a"] not in idx or lab["b"] not in idx:
            missing += 1
            continue
        c = sim.P[idx[lab["a"]], idx[lab["b"]]]
        for b in bands:
            if b[0] <= c < b[1]:
                cnt[b][lab["label"]] += 1
    for b in bands:
        n = sum(cnt[b].values())
        if not n:
            continue
        s, r_, d = cnt[b]["S"], cnt[b]["R"], cnt[b]["D"]
        hi = "+" if b[1] > 1 else f"-{b[1]:.2f}"
        rows.append([f"{b[0]:.2f}{hi}", n, s, r_, d, pct(s / n, 0), pct((s + r_) / n, 0)])
    return rows, missing


def mtime_note(d):
    keep = {k: v for k, v in d["mtimes_seen"].items() if k in ("summary.json", "skillopt/best_skill.md", "skillopt/history.json", "gepa/candidates.json", "gepa/run_log.json",
                                                              "gepa/summary.json", "final/SKILL.md", "<run root>")}
    return "; ".join(f"{k} {v[5:16]}" for k, v in keys_sorted(keep))


def keys_sorted(d):
    return sorted(d.items())


def pooled_stats(prof, ctx, benches=None):
    """(shared share of words, chance-floor share of words) pooled over the runs of the given benchmarks (None = all)."""
    rows = [r for r in prof["runs"] if benches is None or ctx.runs[r["q"]].bench in benches]
    W = sum(r["words"] for r in rows)
    return (sum(r["words_shared"] for r in rows) / W, sum((r["floor_words"] or 0) * r["words"] for r in rows) / W)


# ---------------------------------------------------------------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------------------------------------------------------------
def compute(verify_dedupe=False):
    runs, seed_hand, seed_min = load_runs()
    checks, ok = validate(runs, seed_hand, seed_min)
    assert ok, "\n".join(c for c in checks if c.startswith("FAIL"))
    U = build_universe(runs, seed_hand)
    sim = Sim(U)
    ctx = Ctx(U, runs)
    R = {"checks": checks, "ctx": ctx, "U": U, "sim": sim, "runs": runs}

    # --- question 1: shared vs unique, per kind
    R["prof"], R["cats"], R["seed"], R["pairs"], R["scratch_pairs"] = {}, {}, {}, {}, {}
    for kind in KINDS:
        M = sim.match(kind)
        prof = profile(M, ctx)
        R["prof"][kind] = prof
        R["cats"][kind] = cat_composition(ctx, prof)
        R["seed"][kind] = seed_overlap(M, ctx)
        R["pairs"][kind], R["scratch_pairs"][kind] = pair_table(ctx, prof)
        R.setdefault("decomp", {})[kind] = relation_decomp(ctx, prof)

    ctx_ns = Ctx(U, runs, drop=lambda r: r.scratch)          # same analysis without the two near-empty-seed runs in the pools
    R["ctx_ns"] = ctx_ns
    R["prof_ns"] = {kind: profile(sim.match(kind), ctx_ns) for kind in KINDS}
    # seed-echo sentences (counterpart in the hand-written seed under L or P2) removed from every pool and every numerator
    echo = np.zeros(len(U), bool)
    for i, u in enumerate(U):
        if u["owner"] != "SEED":
            echo[i] = bool(sim.P[i, ctx.seed_idx].max() >= TAU_P2 or sim.J[i, ctx.seed_idx].max() >= TAU_L)
    R["echo"] = echo
    ctx_ne = Ctx(U, runs, keep=~echo)
    R["ctx_ne"] = ctx_ne
    R["prof_ne"] = {kind: profile(sim.match(kind), ctx_ne) for kind in KINDS}

    def pooled(prof, runs_idx=None):
        return pooled_stats(prof, ctx, runs_idx)

    sweeps = {}
    for name, mat, taus in (("J4 Jaccard", lambda t: sim.J >= t, SWEEP_J), ("3-gram containment", lambda t: sim.C >= t, SWEEP_C), ("tf-idf cosine", lambda t: sim.P >= t, SWEEP_P)):
        sweeps[name] = []
        for t in taus:
            pr = profile(mat(t), ctx)
            sweeps[name].append({"tau": t, "pooled": pooled(pr), "by_bench": {b: pooled(pr, {b}) for b in BENCHES},
                                 "n_pairs_same_bench": int(sum(mat(t)[i, j] for i in range(ctx.n) for j in range(i + 1, ctx.n)
                                                              if U[i]["bench"] == U[j]["bench"] and U[i]["owner"] != U[j]["owner"] and U[i]["owner"] != "SEED"))})
    R["sweeps"] = sweeps
    # agreement of the keyword first pass with the hand-coded category on the nine DeepSeek skills (read-only cache of skilltext.py)
    try:
        coded = json.loads((layout.CACHE / "analysis/main/skilltext.json").read_text())["coded_units"]
        units_all = [u for v in coded.values() for u in v]
        agree = sum(1 for u in units_all if u["first_pass"] == u["category"])
        R["cat_agree"] = f"{agree} of {len(units_all)} ({100 * agree / len(units_all):.0f}%)"
    except Exception as e:  # noqa: BLE001
        R["cat_agree"] = f"n/a ({type(e).__name__})"

    # --- calibration of the cosine threshold with hand labels
    labels = json.loads(LABELS.read_text())["labels"]
    R["cal"], R["cal_missing"] = cal_table(sim, U, labels)
    R["n_labels"] = len(labels)

    # --- question 2: closest pairs to the hand-written seed for the near-empty runs (and echo shares for every run)
    R["closest"] = {}
    for q, r in enumerate(ctx.runs):
        if r.scratch:
            R["closest"][r.key] = {"by_cos": closest_to_seed(sim, ctx, q, 10, "P"), "by_J": closest_to_seed(sim, ctx, q, 10, "J"),
                                  "n_J_pos": int(sum(1 for i in ctx.run_idx[q] if sim.J[i, ctx.seed_idx].max() > 0))}

    # --- best same-benchmark match of every sentence (audit table)
    rows = []
    for i, u in enumerate(U):
        if u["owner"] == "SEED":
            continue
        q = ctx.rk[u["owner"]]
        oth = [j for p in ctx.others_same_bench(q) for j in ctx.run_idx[p]]
        jbest = max(oth, key=lambda j: (sim.P[i, j], sim.J[i, j])) if oth else None
        sbest = max(ctx.seed_idx, key=lambda j: (sim.P[i, j], sim.J[i, j]))
        rows.append({"run": u["owner"], "text": u["text"], "ntok": u["ntok"], "cat": u["cat"],
                     "shared": {k: bool(R["prof"][k]["runs"][q]["flag"][list(ctx.run_idx[q]).index(i)]) for k in KINDS},
                     "best_other": None if jbest is None else {"run": U[jbest]["owner"], "text": U[jbest]["text"], "cos": round(float(sim.P[i, jbest]), 3),
                                                              "J4": round(float(sim.J[i, jbest]), 3), "C3": round(float(sim.C[i, jbest]), 3)},
                     "best_seed": {"text": U[sbest]["text"], "cos": round(float(sim.P[i, sbest]), 3), "J4": round(float(sim.J[i, sbest]), 3)}})
    R["sentence_rows"] = rows

    # --- question 3: task specificity of the AutoBench skills
    panels = panel_ids()
    all_ids, tasks, configs = load_ab(panels)
    union = panels["dev"] + panels["test"]
    lex = service_lexicon(configs, union)
    R["lex"] = {"services": sorted(lex[0]), "n_tools": len(lex[1]), "n_fields": len(lex[2])}
    R["panels"] = {k: len(v) for k, v in panels.items()}
    bench_data = {}
    for pn in ("dev", "test"):
        bd = corpus_for(panels[pn], tasks)
        bd["ids"] = all_ids
        bench_data[pn] = bd
    audit = CHEAT["audit_text"]
    spec = {}
    for q, r in enumerate(ctx.runs):
        if r.bench != AB:
            continue
        sents = [U[i]["text"] for i in ctx.run_idx[q]]
        tags = classify_specificity(sents, tasks, panels, lex)
        text_added = "\n".join(added_lines(r.base, r.final))
        spec[r.key] = {"tags": tags, "audit": {pn: audit(text_added, bench_data[pn]) for pn in ("dev", "test")}}
    seed_sents = [U[i]["text"] for i in ctx.seed_idx]
    R["seed_tags"] = classify_specificity(seed_sents, tasks, panels, lex)
    R["seed_audit"] = {pn: audit("\n".join(l for l in seed_hand.splitlines() if l.strip()), bench_data[pn]) for pn in ("dev", "test")}
    R["spec"] = spec
    if verify_dedupe:
        load_bench = {}
        orig = CHEAT["load_bench"](AB, panels["dev"])
        r0 = next(r for r in ctx.runs if r.bench == AB)
        text_added = "\n".join(added_lines(r0.base, r0.final))
        a1, a2 = audit(text_added, orig), audit(text_added, bench_data["dev"])
        diffs = {k: (a1[k]["share"], a2[k]["share"]) for k in ("ov5_prompt", "ov5_answer", "ov8_answer", "ov13_answer") if a1[k]["share"] != a2[k]["share"]}
        R["verify_dedupe"] = {"run": r0.key, "n_ngram_keys_differing": len(diffs), "diffs": diffs}

    # --- proposal level
    UP, pinfo = build_universe_P(runs)
    simP = Sim(UP)
    ctxP = Ctx(UP, runs)
    R["P"] = {"U": UP, "info": pinfo, "ctx": ctxP, "prof": {}, "pairs": {}}
    for kind in KINDS:
        MP = simP.match(kind)
        R["P"]["prof"][kind] = profile(MP, ctxP)
    return R


# ---------------------------------------------------------------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------------------------------------------------------------
def run_order(r):
    return (BENCHES.index(r.bench), r.scratch, ["deepseek-flash", "gpt-6-luna"].index(r.model), OPTS.index(r.opt))


def rowmap(prof, ctx):
    return {ctx.runs[row["q"]].key: row for row in prof["runs"]}


def spec_summary(R):
    ctx = R["ctx"]
    out = {}
    for key, sp in R["spec"].items():
        q = ctx.rk[key]
        tags = sp["tags"]
        flag = R["prof"]["P2"]["runs"][q]["flag"]
        ws = ctx.words[ctx.run_idx[q]]
        n = len(tags)
        cnt = collections.Counter(t["class"] for t in tags)
        wsh = collections.Counter()
        for t, w in zip(tags, ws):
            wsh[t["class"]] += w
        out[key] = {"n": n, "cnt": cnt, "wshare": {c: wsh[c] / ws.sum() for c in wsh}, "any_concrete": sum(1 for t in tags if t["tokens"]),
                    "grounded": sum(1 for t in tags if any(x["df"] > 0 for x in t["tokens"])), "specific": cnt["task-specific"],
                    "answer_only": sum(1 for t in tags if t["n_answer_only"]), "flag": flag}
    return out


SPEC_CLASSES = ("no concrete name", "named, not in dev/test files", "environment-wide", "task-specific")


def cats_by_bench(R, kind):
    ctx = R["ctx"]
    per = R["cats"][kind][0]
    acc = {b: {"shared": collections.Counter(), "unique": collections.Counter()} for b in list(BENCHES) + ["all"]}
    for row, c in zip(R["prof"][kind]["runs"], per):
        b = ctx.runs[row["q"]].bench
        for part in ("shared", "unique"):
            acc[b][part].update(c[part])
            acc["all"][part].update(c[part])
    return acc


def cat_sentence(counter, k=3):
    t = sum(counter.values())
    top = sorted(counter.items(), key=lambda kv: -kv[1])[:k]
    return ", ".join(f"{c} {100 * v / t:.0f}%" for c, v in top) if t else "empty"


def build_record(R):
    ctx, U, sim, runs = R["ctx"], R["U"], R["sim"], R["runs"]
    lab = {r.key: r for r in runs}
    out = []
    P = R["prof"]
    rm = {k: rowmap(P[k], ctx) for k in KINDS}
    rm_ns = {k: rowmap(R["prof_ns"][k], R["ctx_ns"]) for k in KINDS}
    echo = {k: {ctx.runs[s["q"]].key: s for s in R["seed"][k]} for k in KINDS}
    cats = {k: R["cats"][k][0] for k in KINDS}
    cats_run = {k: {ctx.runs[row["q"]].key: c for row, c in zip(P[k]["runs"], cats[k])} for k in KINDS}
    ordered = sorted(runs, key=run_order)
    learned = [r for r in ordered if r.key in ctx.rk]
    empty = [r for r in ordered if r.key not in ctx.rk]
    pooled_all = {k: pooled_stats(P[k], ctx) for k in KINDS}
    pb = {b: {k: pooled_stats(P[k], ctx, {b}) for k in KINDS} for b in BENCHES}
    pb_ns = {k: pooled_stats(R["prof_ns"][k], R["ctx_ns"], {AB}) for k in KINDS}
    nsent = {b: sum(rm["P2"][r.key]["n_sent"] for r in learned if r.bench == b) for b in BENCHES}
    words = {b: sum(rm["P2"][r.key]["words"] for r in learned if r.bench == b) for b in BENCHES}
    spec = spec_summary(R)

    # ------------------------------------------------------------------ header sections
    what = (
        "How much of what GEPA, SkillOpt and SkillAdam write into a skill is a strategy that independent runs share, and how much is idiosyncratic. "
        "Unit of analysis: the sentences of the text a run ADDED to the seed it started from, in the skill the run returned (final skill; the text added step by step "
        "is analysed in 1d). Four questions: (1) per learned skill, which share of the added text has a near-duplicate in the added text of at least one other run on "
        "the same benchmark (shared) and which has none (unique), and in which behaviour categories of `skilltext.py` the two parts fall; (2) how much of the added "
        "text of the two near-empty-seed runs is already stated by the hand-written seed; (3) how much of the AutoBench added text names tools, paths, fields or "
        "values that occur in the dev or test task files; (4) a control: overlap of two skills from the same optimizer but different models against overlap of skills "
        "from different optimizers, and against the same pairs taken across benchmarks, and, per sentence, which relation of partner (same optimizer, same model, "
        "neither) carries the sharing, so that overlap caused by a common model or a common optimizer prompt is not "
        "read as convergence on a common strategy. Descriptive only; no new rollouts, no model calls.")
    n_lr = len(learned)
    runs_txt = (
        f"20 path files in `OUT/skill_paths/paths/` (extractors `extract_{{gepa,skillopt,skilladam}}.py`): the 18 Table-1 runs (DeepSeek native-20261008-https, Luna native-20261012; "
        "GEPA, SkillOpt, SkillAdam x tau3, AutoBench, TB2) and the two Luna AutoBench runs from the near-empty seed (`fromscratch`: GEPA, SkillOpt; there is no SkillAdam "
        f"fromscratch run). {n_lr} runs have added text; the three Luna TB2 runs ({', '.join(r.label for r in empty)}) returned the seed text, their added text is empty, so TB2 has "
        "three learned skills (DeepSeek) and every TB2 skill has only two potential partners (their proposals are in 1d). Learned skills per benchmark: tau3 "
        f"{sum(1 for r in learned if r.bench == 'tau3-bench')}, AutoBench {sum(1 for r in learned if r.bench == AB)} (6 native + 2 fromscratch), TB2 "
        f"{sum(1 for r in learned if r.bench == 'terminal-bench-2-1')}. Returned skill = the skill whose test evaluation exists (`returned_skill_id`); its text equals "
        "`summary.json` final_skill in all 20 runs (check below). PARTIAL runs, as marked by the extractors (read-only, mtimes seen): "
        + "; ".join(f"`{r.key}` ({r.d['partial_reasons'][0] if r.d.get('partial_reasons') else 'see anomalies'}; {mtime_note(r.d)})" for r in ordered if r.partial)
        + ". For both the optimizer loop is complete (all planned steps are in the logs; the process still alive replicates test evaluations); their returned skill is final, "
        "only test numbers may still grow. Seeds: `skills/seed/SKILL.md` (580 words, hand-written, verification-before-completion) and `$BIPA_ASSETS/skills/seed-minimal/SKILL.md` "
        "(`Solve the task.`). Code imported from `src/analysis/skilltext.py` (`CATS`, `RX`, `classify`, `units`) and "
        "`aux/contamination_and_loading/cheatsheet.py` (`WORD`, `PATH_RE`, `VALUE_RE`, `audit_text`, file filters).")
    agg = (
        "**Text.** `skilltext.units()` splits a skill into logical units (bullet, numbered item, table row, wrapped paragraph); headings, code-fence marks, table separators and "
        "frontmatter (name/description) are structure and are not text, as in skilltext.py. A unit is ADDED when one of its lines is inserted or replaced against the seed the "
        "run started from (difflib on non-empty stripped lines; recomputed added lines equal the path files' `added_lines` for every candidate, check below); sentences of "
        "an added unit that occur verbatim in the seed are dropped. A unit is split into sentences (markdown markers stripped, table cells joined, e.g./i.e. protected). "
        f"Sentences with fewer than {MIN_TOK} normalized words (lowercase [a-z0-9]+, `cheatsheet.WORD`) are FRAGMENTS: they cannot hold a 4-gram, are left out of all "
        "denominators and are counted separately. Exact duplicates inside one skill count once. Weight of a sentence = its normalized words; shares are shares of words "
        "unless marked as sentence shares. "
        "**Same sentence, four ways.** L (lenient verbatim): word 4-gram shingle Jaccard >= " f"{TAU_L}. S (strict verbatim, the exact-3-gram rule): at least {int(TAU_S * 100)}% of the exact word "
        f"3-grams of the shorter sentence occur in the other, both sentences >= {S_MIN_WORDS} words. P2 / P3 (paraphrase level, an addition of this record): tf-idf cosine "
        f"(stemmed content-word unigrams + bigrams, stop words out, idf over all {len(U)} sentences of the 20 skills plus the seed) >= {TAU_P2} / {TAU_P3}. "
        f"Why these values: see 5 (sweeps against a chance floor, and {R['n_labels']} hand-labelled pairs); L and S are almost never reached by two independent runs (the sentences are "
        "rephrased, not copied), so a verbatim measure alone would read as 'nothing is shared'; P2/P3 find the same rule in different words (precision against the hand labels "
        f"about {R['cal'][2][5]}% at the P2 band, about {R['cal'][4][5]}% at P3 and above). **Shared** = the sentence has a counterpart (by the measure) in at least one OTHER "
        "learned run on the same benchmark; unique = none. The share depends on the number of partners (tau3 5, AutoBench 7, TB2 2), so every share is set next to a "
        "**chance floor**: the same count of partner runs drawn from the OTHER benchmarks (all subsets, or 3000 random ones), averaged. Observed minus floor is what "
        "belongs to the benchmark rather than to generic skill language. **Pair overlap** (control 4) = (shared words of A against B + shared words of B against A) / "
        "(words of A + words of B). **Categories**: `skilltext.classify` first pass (keyword regex, 8 mutually exclusive categories, no hand overrides) applied to each sentence. "
        "**Task specificity**: sentence-level tags on the AutoBench skills (3).")

    # ------------------------------------------------------------------ 0 checks
    n_ok = sum(1 for c in R["checks"] if c.startswith("OK"))
    chk = [f"- {n_ok} of {len(R['checks'])} input checks passed ({len(R['checks']) - n_ok} failed). Per run: step 0 equals the seed file, every step has text, every parent exists, recomputed "
           "added lines equal the path file's `added_lines` for all candidates, the returned skill text equals `summary.json` final_skill."]
    if "verify_dedupe" in R:
        v = R["verify_dedupe"]
        chk.append(f"- Fidelity of the corpus builder to `cheatsheet.load_bench` (identical file contents counted once): on the dev panel, 5/8/13-gram overlap shares of the added text of "
                   f"`{v['run']}` against the prompt and answer corpora differ in {v['n_ngram_keys_differing']} of 4 numbers.")
    chk.append(f"- Universe: {len(U)} non-fragment sentences ({len(U) - len(ctx.seed_idx)} added, {len(ctx.seed_idx)} of the hand-written seed). Fragments left out: "
               + ", ".join(f"{short_label(r)} {r.n_frag} ({int(r.frag_words)} words)" for r in learned if r.n_frag) + ". Exact duplicates removed inside a skill: "
               + (", ".join(f"{short_label(r)} {r.dup_removed}" for r in learned if r.dup_removed) or "none") + ".")
    chk.append(f"- Hand labels for the cosine thresholds: {R['n_labels']} pairs, {R['cal_missing']} not found in the current universe.")
    sec0 = "\n".join(chk)

    # ------------------------------------------------------------------ 1 per skill
    rows = []
    for r in ordered:
        if r.key not in ctx.rk:
            rows.append([BSHORT[r.bench], r.label, "returned the seed", "0", "-", "-", "-", "-", "-", "-", "-", "-"])
            continue
        p2 = rm["P2"][r.key]
        rows.append([BSHORT[r.bench], r.label + (" †" if r.partial else ""), f"{p2['n_sent']} ({r.n_frag})", int(p2["words"]), pct(echo["P2"][r.key]["share_words"], 0)]
                    + [pct(rm[k][r.key]["share_words"]) for k in KINDS] + [pct(p2["floor_words"]),
                       top_cats(cats_run["P2"][r.key]["shared"]), top_cats(cats_run["P2"][r.key]["unique"])])
    t1 = md_table(["bench", "skill", "added sentences (fragments)", "added words", "seed-echo % (P2)", "shared % (L)", "shared % (S)", "shared % (P2)", "shared % (P3)", "P2 chance floor %",
                   "shared part (P2): top categories", "unique part (P2): top categories"], rows)
    t1_note = ("Shares of added WORDS. Partners = all other learned runs on the benchmark (tau3 5, AutoB 7, TB2 2); chance floor = same number of partners drawn from other "
               "benchmarks. † = partial run (see Runs). Seed-echo = share of the added words whose sentence has a counterpart (P2) in the hand-written seed (GEPA and "
               "SkillAdam rewrite seed lines in place; those rewrites are 'added' by the line diff).")

    # pooled
    prow = []
    for b in BENCHES:
        prow.append([BSHORT[b], sum(1 for r in learned if r.bench == b), nsent[b], int(words[b])] + [f"{pct(pb[b][k][0])} ({pct(pb[b][k][1])})" for k in KINDS])
    prow.append(["AutoB, 6 native only", 6, sum(rm_ns["P2"][r.key]["n_sent"] for r in learned if r.bench == AB and not r.scratch),
                 int(sum(rm_ns["P2"][r.key]["words"] for r in learned if r.bench == AB and not r.scratch))] + [f"{pct(pb_ns[k][0])} ({pct(pb_ns[k][1])})" for k in KINDS])
    prow.append(["all", n_lr, sum(nsent.values()), int(sum(words.values()))] + [f"{pct(pooled_all[k][0])} ({pct(pooled_all[k][1])})" for k in KINDS])
    t1b = md_table(["benchmark", "skills", "sentences", "words", f"L ({KIND_NAME['L'].split(': ')[1]})", f"S ({KIND_NAME['S'].split(': ')[1]})",
                    f"P2 (cos >= {TAU_P2})", f"P3 (cos >= {TAU_P3})"], prow)
    t1b_note = "Pooled shared share of words, observed (chance floor from other benchmarks in brackets), %. The AutoB native-only row drops the two fromscratch skills from the partner pools."
    ne = {b: {k: pooled_stats(R["prof_ne"][k], R["ctx_ne"], {b}) for k in KINDS} for b in BENCHES}
    ne_all = {k: pooled_stats(R["prof_ne"][k], R["ctx_ne"]) for k in KINDS}
    n_echo = int(R["echo"].sum())
    w_echo = float(ctx.words[R["echo"]].sum())
    nerow = [[BSHORT[b]] + [f"{pct(ne[b][k][0])} ({pct(ne[b][k][1])})" for k in KINDS] for b in BENCHES] + [["all"] + [f"{pct(ne_all[k][0])} ({pct(ne_all[k][1])})" for k in KINDS]]
    t1b2 = md_table(["benchmark (seed-echo sentences removed)", "L", "S", "P2", "P3"], nerow)
    t1b2_note = (f"Same as above with the {n_echo} seed-echo sentences ({int(w_echo)} words; counterpart in the hand-written seed under L or P2) removed from every pool and every numerator, so that "
                 "rewritten seed lines cannot count as shared strategy.")

    # categories
    cb = {k: cats_by_bench(R, k) for k in ("P2",)}
    crow = []
    for c in CAT_IDS:
        row = [c]
        for b in list(BENCHES) + ["all"]:
            sh, un = cb["P2"][b]["shared"], cb["P2"][b]["unique"]
            row += [pct(sh[c] / sum(sh.values()), 0) if sum(sh.values()) else "-", pct(un[c] / sum(un.values()), 0) if sum(un.values()) else "-"]
        crow.append(row)
    crow.append(["words in the part"] + [x for b in list(BENCHES) + ["all"] for x in (int(sum(cb["P2"][b]["shared"].values())), int(sum(cb["P2"][b]["unique"].values())))])
    t1c = md_table(["category"] + [f"{BSHORT.get(b, b)} {part}" for b in list(BENCHES) + ["all"] for part in ("shared", "unique")], crow)
    t1c_note = ("Share of the words of the shared part (P2) and of the unique part that fall into each category, %, pooled over the learned skills of the benchmark; categories: "
                + "; ".join(f"`{c}` = {d}" for c, d in CATS) + ". Under L and S fewer than 10 sentences are shared in total, too few to code.")

    # examples of shared pairs
    pairs_all = []
    for i in range(len(U)):
        for j in range(i + 1, len(U)):
            if U[i]["owner"] != "SEED" and U[j]["owner"] != "SEED" and U[i]["owner"] != U[j]["owner"] and U[i]["bench"] == U[j]["bench"]:
                pairs_all.append((float(sim.P[i, j]), i, j))
    pairs_all.sort(reverse=True)
    erow = [[f"{c:.2f}", f"{sim.J[i, j]:.2f}", f"{sim.C[i, j]:.2f}", BSHORT[U[i]["bench"]], f"{short_label(lab[U[i]['owner']])[len(BSHORT[U[i]['bench']]) + 1:]} / {short_label(lab[U[j]['owner']])[len(BSHORT[U[j]['bench']]) + 1:]}",
             clip(U[i]["text"], 120), clip(U[j]["text"], 120)] for c, i, j in pairs_all[:16]]
    t1e = md_table(["cos", "J4", "C3", "bench", "runs", "sentence A", "sentence B"], erow)
    rng = random.Random(RNG_SEED)
    uniq = [(i, u) for i, u in enumerate(U) if u["owner"] != "SEED" and not rm["P2"][u["owner"]]["flag"][list(ctx.run_idx[ctx.rk[u["owner"]]]).index(i)]]
    samp = rng.sample(uniq, 12)
    urow = [[short_label(lab[u["owner"]]), u["cat"], clip(u["text"], 190)] for i, u in samp]
    t1u = md_table(["skill", "category", "sentence (unique under P2)"], urow)

    # 1d proposals
    UP, ctxP, pinfo = R["P"]["U"], R["P"]["ctx"], R["P"]["info"]
    rmP = {k: rowmap(R["P"]["prof"][k], ctxP) for k in KINDS}
    prow2 = []
    for r in ordered:
        if r.key not in ctxP.rk:
            prow2.append([BSHORT[r.bench], r.label, pinfo[r.key]["n_candidates"], 0, "-", "-", "-", "-", "-", "-"])
            continue
        x = rmP["P2"][r.key]
        prow2.append([BSHORT[r.bench], r.label + (" †" if r.partial else ""), pinfo[r.key]["n_candidates"], f"{x['n_sent']} / {int(x['words'])}"]
                     + [pct(rmP[k][r.key]["share_words"]) for k in KINDS] + [pct(x["floor_words"])])
    pb2 = {b: {k: pooled_stats(R["P"]["prof"][k], ctxP, {b}) for k in KINDS} for b in BENCHES}
    for b in BENCHES:
        prow2.append([BSHORT[b], "pooled", "", "", *[f"{pct(pb2[b][k][0])} ({pct(pb2[b][k][1])})" for k in KINDS], ""])
    t1d = md_table(["bench", "run", "candidates", "unique proposed sentences / words", "shared % L", "S", "P2", "P3", "P2 chance floor %"], prow2)
    t1d_note = ("Proposal level: every candidate of the run (kept or not, dev-scored or not; the added sentences of each candidate against ITS parent, de-duplicated inside the run) "
                "against the proposals of the other runs on the benchmark. GEPA proposes many near-copies of its parent, so its denominators are inflated; Luna TB2 appears here "
                "because its runs proposed text although they returned the seed. Pooled rows: observed (chance floor).")

    # ------------------------------------------------------------------ 2 seed overlap
    srow = []
    for r in ordered:
        if r.key not in ctx.rk:
            continue
        srow.append([short_label(r) + (" †" if r.partial else "")] + [pct(echo[k][r.key]["share_words"]) for k in KINDS] + [f"{echo['P2'][r.key]['sent_echo']} / {rm['P2'][r.key]['n_sent']}"])
    t2 = md_table(["skill", "L %", "S %", "P2 %", "P3 %", "P2 sentences / added sentences"], srow)
    t2c = []
    for r in ordered:
        if not r.scratch or r.key not in R["closest"]:
            continue
        cl = R["closest"][r.key]
        rows_c = [[n + 1, f"{c:.2f}", f"{j:.2f}", f"{cc:.2f}", clip(U[i]["text"], 150), clip(U[jj]["text"], 150)] for n, (c, j, cc, i, jj) in enumerate(cl["by_cos"])]
        t2c.append(f"**{short_label(r)}** †: the 10 added sentences closest to any sentence of the hand-written seed (ranked by cosine; J4 > 0 for only {cl['n_J_pos']} of its "
                   f"{rm['P2'][r.key]['n_sent']} sentences, so ranking by Jaccard would be arbitrary):\n\n" + md_table(["#", "cos", "J4", "C3", "added sentence", "closest seed sentence"], rows_c))
    t2c = "\n\n".join(t2c)

    def cat_mix(ix):
        c = collections.Counter()
        for i in ix:
            c[ctx.cat[i]] += float(ctx.words[i])
        t = sum(c.values())
        return [pct(c[k] / t, 0) if t else "-" for k in CAT_IDS], t

    mrows = []
    mx, mt = cat_mix(ctx.seed_idx)
    mrows.append(["hand-written seed (its own text)", len(ctx.seed_idx), int(mt)] + mx)
    for r in ordered:
        if r.key in ctx.rk and r.scratch:
            mx, mt = cat_mix(ctx.run_idx[ctx.rk[r.key]])
            mrows.append([short_label(r) + " †", len(ctx.run_idx[ctx.rk[r.key]]), int(mt)] + mx)
    for lbl, sel in (("AutoB native skills, pooled (6)", lambda r: r.bench == AB and not r.scratch), ("all native learned skills, pooled (15)", lambda r: not r.scratch)):
        ixs = [i for r in ordered if r.key in ctx.rk and sel(r) for i in ctx.run_idx[ctx.rk[r.key]]]
        mx, mt = cat_mix(ixs)
        mrows.append([lbl, len(ixs), int(mt)] + mx)
    t2m = md_table(["text", "sentences", "words"] + list(CAT_IDS), mrows)

    # ------------------------------------------------------------------ 3 specificity
    keys_ab = [r.key for r in ordered if r.bench == AB and r.key in ctx.rk]
    sprow = []
    tot = collections.Counter()
    tot_n = 0
    for k in keys_ab:
        s = spec[k]
        a = R["spec"][k]["audit"]
        sprow.append([short_label(lab[k])[len(BSHORT[lab[k].bench]) + 1:] + (" †" if lab[k].partial else ""), s["n"]] + [f"{s['cnt'][c]} ({pct(s['cnt'][c] / s['n'], 0)}%)" for c in SPEC_CLASSES]
                     + [s["answer_only"], f"{pct(a['dev']['ov8_answer']['share'], 2)} / {pct(a['test']['ov8_answer']['share'], 2)}", f"{len(a['dev']['task_ids'])} / {len(a['test']['task_ids'])}",
                        f"{len(a['dev']['paths_specific']) + len(a['dev']['values_specific']) + len(a['dev']['commands'])} / {len(a['test']['paths_specific']) + len(a['test']['values_specific']) + len(a['test']['commands'])}"])
        tot.update(s["cnt"])
        tot_n += s["n"]
    sd = R["seed_audit"]
    seedc = collections.Counter(t["class"] for t in R["seed_tags"])
    sprow.append(["hand-written seed (control)", len(R["seed_tags"])] + [f"{seedc[c]} ({pct(seedc[c] / len(R['seed_tags']), 0)}%)" for c in SPEC_CLASSES]
                 + [sum(1 for t in R["seed_tags"] if t["n_answer_only"]), f"{pct(sd['dev']['ov8_answer']['share'], 2)} / {pct(sd['test']['ov8_answer']['share'], 2)}",
                    f"{len(sd['dev']['task_ids'])} / {len(sd['test']['task_ids'])}",
                    f"{len(sd['dev']['paths_specific']) + len(sd['dev']['values_specific']) + len(sd['dev']['commands'])} / {len(sd['test']['paths_specific']) + len(sd['test']['values_specific']) + len(sd['test']['commands'])}"])
    sprow.append(["all 8 AutoB skills", tot_n] + [f"{tot[c]} ({pct(tot[c] / tot_n, 0)}%)" for c in SPEC_CLASSES]
                 + [sum(spec[k]["answer_only"] for k in keys_ab), "", "", ""])
    t3 = md_table(["skill", "added sentences"] + list(SPEC_CLASSES) + ["sentences with answer-only token", "8-gram overlap with answer corpus % (dev / test)",
                                                                       "dev/test task ids named", "specific paths+values+commands in answer corpus (dev / test)"], sprow)
    # cross tab shared vs unique
    xt = {"shared": collections.Counter(), "unique": collections.Counter()}
    xw = {"shared": 0, "unique": 0}
    for k in keys_ab:
        for t, f in zip(R["spec"][k]["tags"], spec[k]["flag"]):
            xt["shared" if f else "unique"][t["class"]] += 1
    t3x = md_table(["part (P2)", "sentences"] + list(SPEC_CLASSES),
                   [[p, sum(xt[p].values())] + [f"{xt[p][c]} ({pct(xt[p][c] / max(1, sum(xt[p].values())), 0)}%)" for c in SPEC_CLASSES] for p in ("shared", "unique")])
    # token inventory
    inv = collections.defaultdict(lambda: {"type": None, "skills": set(), "df": None, "df_dev": None, "df_test": None, "df_train": None, "df_vis": None, "df_ans": None})
    for k in keys_ab:
        for t in R["spec"][k]["tags"]:
            for x in t["tokens"]:
                e = inv[x["tok"]]
                e.update({"type": x["type"], "df": x["df"], "df_dev": x["df_dev"], "df_test": x["df_test"], "df_train": x["df_train"], "df_vis": x["df_vis"], "df_ans": x["df_ans"]})
                e["skills"].add(k)
    typ = collections.defaultdict(lambda: [0, 0, 0])
    for tok, e in inv.items():
        typ[e["type"]][0] += 1
        typ[e["type"]][1] += e["df"] > 0
        typ[e["type"]][2] += 0 < e["df"] <= 3
    t3t = md_table(["token type", "distinct tokens in the 8 added texts", "found in dev/test task files", "of which in <= 3 of the 150 tasks"],
                   [[t, v[0], v[1], v[2]] for t, v in sorted(typ.items())])
    top = sorted(inv.items(), key=lambda kv: (-len(kv[1]["skills"]), kv[0]))[:14]
    t3k = md_table(["token", "type", "AutoB skills naming it (of 8)", "tasks containing it: dev30 / test120 / train160", "visible to the agent (prompt+environment) / answer corpus (solution+tests), of 150"],
                   [[f"`{t}`", e["type"], len(e["skills"]), f"{e['df_dev']} / {e['df_test']} / {e['df_train']}", f"{e['df_vis']} / {e['df_ans']}"] for t, e in top])
    spec_list = []
    for k in keys_ab:
        for t in R["spec"][k]["tags"]:
            if t["class"] == "task-specific":
                sp_t = [f"`{x['tok']}` ({x['type']}, {x['df']} tasks)" for x in t["tokens"] if 0 < x["df"] <= 3]
                spec_list.append(f"- {lab[k].label}: {clip(t['text'], 200)}  [{'; '.join(sp_t)}]")
    ao_list = []
    for k in keys_ab:
        for t in R["spec"][k]["tags"]:
            if t["n_answer_only"]:
                ao = [f"`{x['tok']}` ({x['type']}, {x['df_ans']} tasks)" for x in t["tokens"] if x["df_vis"] == 0 and x["df_ans"] > 0]
                ao_list.append(f"- {lab[k].label}: {clip(t['text'], 200)}  [{'; '.join(ao)}]")
    sec3 = (t3 + "\n\n" + "Sentence classes are exclusive and ordered by evidence: task-specific (a concrete name found in at most 3 of the 150 dev+test task files) > environment-wide "
            "(found, but in more than 3) > named but not found in any dev/test file > no concrete name. Concrete names = paths and file names (`cheatsheet.PATH_RE`), CLI flags, "
            "snake_case / camelCase identifiers, words inside backticks, numbers of 3 or more digits (`cheatsheet.VALUE_RE`), and the service names of the `allowed_services` / "
            f"`initial_state` of the dev+test tasks ({', '.join(R['lex']['services'])}); {R['lex']['n_tools']} Zapier tool names and {R['lex']['n_fields']} snake_case field names occur in the "
            "task configs. 'Answer-only' = found in solution/tests but not in prompt, task.toml or environment files. The last three columns reuse `cheatsheet.audit_text` on the ADDED "
            "text of each skill against the dev panel (30 tasks) and the test panel (120 tasks), answer corpus = solution/** + tests/**.\n\n"
            "Shared against unique sentences (P2), pooled over the 8 skills:\n\n" + t3x + "\n\nToken inventory:\n\n" + t3t + "\n\nThe most widely named tokens:\n\n" + t3k
            + f"\n\nAll {len(spec_list)} sentences tagged task-specific, verbatim:\n\n" + ("\n".join(spec_list) if spec_list else "(none)")
            + f"\n\nAll {len(ao_list)} sentences with an answer-only token:\n\n" + ("\n".join(ao_list) if ao_list else "(none)"))

    # ------------------------------------------------------------------ 4 control
    ctab = []
    for k in KINDS:
        for cl in PAIR_CLASSES:
            for setting in ("within", "cross"):
                v = [x[2] for x in R["pairs"][k][cl][setting]]
                if setting == "within" and cl == PAIR_CLASSES[3]:
                    continue
                ctab.append((k, cl, setting, v))
    crow4 = []
    for cl in PAIR_CLASSES:
        for setting in ("within", "cross"):
            if setting == "within" and cl == PAIR_CLASSES[3]:
                continue
            cells = []
            n = len(R["pairs"]["L"][cl][setting])
            for k in KINDS:
                v = [x[2] for x in R["pairs"][k][cl][setting]]
                cells.append(f"{pct(float(np.mean(v)))} [{pct(min(v))}, {pct(max(v))}]" if v else "-")
            crow4.append([cl, "same benchmark" if setting == "within" else "different benchmarks", n] + cells)
    t4 = md_table(["pair class", "pairs taken", "pairs"] + [f"{k}" for k in KINDS], crow4)
    wrow = []
    for cl in PAIR_CLASSES[:3]:
        for a, b, v in sorted(R["pairs"]["P2"][cl]["within"], key=lambda x: -x[2]):
            va = {k: [y[2] for y in R["pairs"][k][cl]["within"] if y[0] == a and y[1] == b][0] for k in KINDS}
            wrow.append([cl, f"{BSHORT[lab[a].bench]}: {lab[a].label} vs {lab[b].label}"] + [pct(va[k]) for k in KINDS])
    t4p = md_table(["pair class", "pair (same benchmark)", "L %", "S %", "P2 %", "P3 %"], wrow)
    srow4 = []
    for cl, l in sorted(R["scratch_pairs"]["P2"].items()):
        cells = []
        for k in KINDS:
            v = [x[2] for x in R["scratch_pairs"][k][cl]]
            cells.append(f"{pct(float(np.mean(v)))} [{pct(min(v))}, {pct(max(v))}]")
        srow4.append([cl, len(l)] + cells)
    t4s = md_table(["pairs with a fromscratch run (AutoBench)", "pairs"] + list(KINDS), srow4)
    sec4 = (t4 + "\n\nPair overlap, % of the words of the two skills that have a counterpart in the other (mean [min, max] over pairs). Only the native runs enter the classes; "
            "'same optimizer, same model' exists only across benchmarks. TB2 has no same-optimizer-other-model pair (the Luna TB2 skills are empty).\n\n"
            "Every same-benchmark pair of the classes (so that the means can be judged by eye):\n\n" + t4p + "\n\nPairs with a fromscratch run, kept apart because the seed differs:\n\n" + t4s)
    dr = []
    for b in list(BENCHES) + [None]:
        for kind in ("L", "P2", "P3"):
            d = decomp_pooled(R["decomp"][kind], b)
            cells = [f"{pct(d[c][0])} ({pct(d[c][1])}) [{d[c][2]}]" if d[c] else "-" for c in REL_CLASSES]
            dr.append([BSHORT[b] if b else "all", kind, f"{pct(d['any'][0])} [{d['any'][2]}]" if d["any"] else "-"] + cells)
    t4d = md_table(["benchmark", "measure", "any native partner [skills]", "same optimizer, other model: observed (floor) [skills]", "same model, other optimizer",
                    "other model, other optimizer"], dr)
    sec4 += ("\n\n**Which partners carry the sharing (per sentence).** Share of the added words of the six-per-benchmark native skills that have a counterpart in a same-benchmark NATIVE partner "
             "of the given relation to the skill; in brackets the chance floor (the same number of partners of that relation drawn from the other benchmarks, mean over all subsets); "
             "[n] = skills that have at least one partner of that relation (tau3 and AutoB: 1 same-optimizer, 2 same-model, 2 other-model-other-optimizer partners; TB2: the three DeepSeek "
             "skills only have same-model partners). The scratch runs are not used as partners here.\n\n" + t4d)

    # ------------------------------------------------------------------ 5 calibration
    swr = []
    for name, rowsw in R["sweeps"].items():
        for x in rowsw:
            swr.append([name, x["tau"], f"{pct(x['pooled'][0])} ({pct(x['pooled'][1])})"] + [f"{pct(x['by_bench'][b][0])} ({pct(x['by_bench'][b][1])})" for b in BENCHES] + [x["n_pairs_same_bench"]])
    t5s = md_table(["measure", "threshold", "all (floor)", "tau3", "AutoB", "TB2", "same-benchmark sentence pairs above"], swr)
    t5l = md_table(["cosine band", "labelled pairs", "same rule (S)", "related (R)", "different (D)", "S %", "S or R %"], R["cal"])

    # ------------------------------------------------------------------ headline numbers (computed)
    h = {k: pooled_all[k] for k in KINDS}
    e2 = sum(rm["P2"][r.key]["words_shared"] for r in learned) / sum(rm["P2"][r.key]["words"] for r in learned)
    ps = {cl: {k: (float(np.mean([x[2] for x in R["pairs"][k][cl]["within"]])) if R["pairs"][k][cl]["within"] else None,
                   float(np.mean([x[2] for x in R["pairs"][k][cl]["cross"]])) if R["pairs"][k][cl]["cross"] else None) for k in KINDS} for cl in PAIR_CLASSES}
    cs = {b: len(R["spec"][b]["tags"]) for b in keys_ab}
    reading = (
        f"- Verbatim sharing is rare. Pooled over the {n_lr} learned skills, {pct(h['L'][0])}% of the added words have a lenient verbatim counterpart (L) in another run on the same "
        f"benchmark (chance floor {pct(h['L'][1])}%) and {pct(h['S'][0])}% a strict one (S; floor {pct(h['S'][1])}%). At paraphrase level (P2) it is {pct(h['P2'][0])}% (floor {pct(h['P2'][1])}%), "
        f"at P3 {pct(h['P3'][0])}% (floor {pct(h['P3'][1])}%). So about {pct(1 - h['P2'][0], 0)}% of what the optimizers write has no counterpart even in other words in any other run on the "
        "same benchmark under the most lenient measure used here; because P2 misses paraphrases that share few content words, this is an upper bound on the idiosyncratic share.\n"
        f"- What is shared is benchmark-level behaviour, not task-level fact (on AutoBench no added sentence, shared or not, names a task-specific token, 3; tau3 and TB2 were not audited): rules such as 'a success response is not proof the change persisted', 'writes that share state run one at a time', 'preserve source values "
        "verbatim', 'explicit yes before a write', 'stop when the conversation has ended' (examples in 1e). Pooled over all 17 skills the shared part (P2) is " + cat_sentence(cats_by_bench(R, "P2")["all"]["shared"])
        + "; the unique part is " + cat_sentence(cats_by_bench(R, "P2")["all"]["unique"]) + " (words, 1c). Verification rules, the theme of the hand-written seed, are "
        "about half of the added words of the native skills (mix in 2) but mostly fall in the unique part: each run words them differently.\n"
        f"- Control: pair overlap (P2, same benchmark) is {pct(ps['same optimizer, other model']['P2'][0])}% for same optimizer / other model, {pct(ps['same model, other optimizer']['P2'][0])}% "
        f"for same model / other optimizer and {pct(ps['other model, other optimizer']['P2'][0])}% for other model / other optimizer; across benchmarks the same classes give "
        f"{pct(ps['same optimizer, other model']['P2'][1])}%, {pct(ps['same model, other optimizer']['P2'][1])}% and {pct(ps['other model, other optimizer']['P2'][1])}% "
        f"(and {pct(ps['same optimizer, same model']['P2'][1])}% for same optimizer and model, different benchmark). Read against the other-model / other-optimizer cell, where neither a "
        "common model nor a common optimizer prompt is available, the excess is the part that a common model or optimizer explains.\n"
        f"- AutoBench task specificity: {tot['task-specific']} of {tot_n} added sentences ({pct(tot['task-specific'] / tot_n, 0)}%) carry a concrete name found in at most 3 of the 150 dev/test "
        f"task files; {tot['environment-wide']} ({pct(tot['environment-wide'] / tot_n, 0)}%) carry one found in more than 3 (tool, CLI, service or field names of the environment); "
        f"{tot['no concrete name']} ({pct(tot['no concrete name'] / tot_n, 0)}%) carry no concrete name at all.")
    dd = {k: decomp_pooled(R["decomp"][k], None) for k in KINDS}
    scr_runs = [r for r in ordered if r.scratch and r.key in ctx.rk]
    vi = list(CAT_IDS).index("verify")
    seed_verify = mrows[0][3 + vi]
    scr_txt = "; ".join(f"{short_label(r)}: {pct(echo['L'][r.key]['share_words'])} / {pct(echo['S'][r.key]['share_words'])} / {pct(echo['P2'][r.key]['share_words'])} / "
                        f"{pct(echo['P3'][r.key]['share_words'])}% (closest pair cos {R['closest'][r.key]['by_cos'][0][0]:.2f})" for r in scr_runs)
    scr_ver = ", ".join(f"{short_label(r)} {m[3 + vi]}%" for r, m in zip(scr_runs, mrows[1:1 + len(scr_runs)]))
    reading += (
        f"\n- Which partners carry the sharing (4, per sentence, native skills only): P2 counterparts in a partner that differs in model AND optimizer cover {pct(dd['P2']['other model, other optimizer'][0])}% "
        f"of the added words (chance floor {pct(dd['P2']['other model, other optimizer'][1])}%), against {pct(dd['P2']['same model, other optimizer'][0])}% for same-model partners "
        f"(floor {pct(dd['P2']['same model, other optimizer'][1])}%) and {pct(dd['P2']['same optimizer, other model'][0])}% for same-optimizer partners (floor {pct(dd['P2']['same optimizer, other model'][1])}%); "
        f"under L: {pct(dd['L']['other model, other optimizer'][0])}% ({pct(dd['L']['other model, other optimizer'][1])}), {pct(dd['L']['same model, other optimizer'][0])}% ({pct(dd['L']['same model, other optimizer'][1])}), "
        f"{pct(dd['L']['same optimizer, other model'][0])}% ({pct(dd['L']['same optimizer, other model'][1])}). The sharing of 1 therefore sits with a common model (in these runs also the model that writes "
        "the proposals: GEPA's reflection_lm and SkillOpt's optimizer_model are the run's own model; SkillAdam's context also carries the run's model, its proposer call was not read) or a common "
        "optimizer, and is at the chance floor between runs that differ in both. Descriptive: 1 to 2 partners per class and skill.\n"
        f"- Overlap with the hand-written seed (2): the added words of the two near-empty-seed skills with a counterpart in the seed, under L / S / P2 / P3: {scr_txt}. At topic level the verify category "
        f"holds {scr_ver} of their words against {seed_verify}% of the seed's; the ten closest pairs of 2 are listed so that a reader can judge by eye whether any of them is the same rule in other words.")
    caveats = (
        "- The thresholds are arbitrary. L, S and the sweep in 5 show the dependence: the verbatim shares are near zero over the whole range 0.10-0.50 (Jaccard) and 0.5-0.8 (containment); the P "
        "shares fall from the P2 to the P3 value by a factor of about " f"{h['P2'][0] / max(h['P3'][0], 1e-9):.1f}. The cosine thresholds were set against {R['n_labels']} pairs labelled by one rater (the analyst), "
        f"not blind to the cosine, with at most 12 pairs per band; precision at the P2 band is an estimate with a wide interval ({R['cal'][2][2]} of {R['cal'][2][1]} pairs). Recall is not measured: tf-idf cosine misses "
        "paraphrases with few shared content words, so shared shares are lower bounds of shared substance.\n"
        "- Very few runs: 17 skills, 3 learned TB2 skills, 3 optimizers x 2 models, one run per cell. Pair classes hold 6, 15 and 12 same-benchmark pairs; pairs share runs and are not independent, "
        "no interval or test is given. The share of a skill depends on the size of its partner pool (5, 7, 2) and on its own length (SkillAdam skills hold 4 to 12 sentences: one sentence moves "
        "its share by 10 points); read the pooled rows and the floor before the per-skill rows.\n"
        "- Categories are the first-pass keyword categories of skilltext.py (no hand overrides) applied to sentences, not to the hand-coded units; the verify and domain regular expressions are "
        f"broad. Against the hand codes of the nine DeepSeek skills the first pass agrees with the final category for {R['cat_agree']} hand-coded units (assets/cache/analysis/main/skilltext.json).\n"
        "- The near-empty-seed runs are two runs of one model on one benchmark; their overlap with the seed (2) is a statement about those two skills; the lexical measures do not see a rule restated without shared content words, "
        "and the topic-level mix is coarse (8 keyword categories). Both are partial runs (loop finished, test replicates still being written).\n"
        "- The task-specificity tags are heuristic (surface forms of names, substring lookup in the task files, document frequency <= 3 as in cheatsheet.py); single-word field names (status, subject, amount) "
        "are not tagged, so field-name mentions are under-counted; no answer-key information beyond what the n-gram audit shows was searched for.\n"
        "- Added text is defined by a line diff: a seed line rewritten in place counts as added (see seed-echo); GEPA's final skills are largely rewritten seeds, SkillOpt and SkillAdam append.")

    # ------------------------------------------------------------------ assemble
    numbers = (
        "### 0. Inputs and checks\n\n" + sec0
        + "\n\n### 1a. Shared against unique added text, per learned skill\n\n" + t1 + "\n\n" + t1_note
        + "\n\n### 1b. Pooled per benchmark\n\n" + t1b + "\n\n" + t1b_note + "\n\n" + t1b2 + "\n\n" + t1b2_note
        + "\n\n### 1c. Categories of the shared and the unique part\n\n" + t1c + "\n\n" + t1c_note
        + "\n\n### 1d. Proposal level (all candidates, not only the returned skill)\n\n" + t1d + "\n\n" + t1d_note
        + "\n\n### 1e. What the shared pairs look like (the 16 closest same-benchmark pairs by cosine) and a sample of unique sentences\n\n" + t1e + "\n\n" + t1u
        + "\n\n### 2. Overlap with the hand-written seed\n\n" + t2 + "\n\nShare of the added words with a counterpart in the hand-written seed (seed-echo), % by measure.\n\n" + t2c
        + "\n\nTopic level (not sentence level): behaviour-category mix of ALL words of a text (`skilltext.classify` first pass, % of words, rows sum to 100 up to rounding). The hand-written seed is "
        "about verifying before claiming completion; a sentence-level counterpart can be absent while the topic is the same.\n\n" + t2m
        + "\n\n### 3. Task specificity of the AutoBench skills\n\n" + sec3
        + "\n\n### 4. Control: same optimizer other model, other optimizer, other benchmark\n\n" + sec4
        + "\n\n### 5. Thresholds: sweeps and hand labels\n\n" + t5s + "\n\nPooled shared share of words, observed (chance floor from other benchmarks), final skills.\n\n" + t5l
        + "\n\nHand labels per cosine band (S = states substantially the same rule, R = related but not the same rule, D = different; `shared/soft_labels.json`)."
        + "\n\n### 6. Reading\n\n" + reading + "\n\n### 7. Caveats\n\n" + caveats)
    text = "\n".join([
        "# mech_shared", "",
        "Audit record written by the script named below.", "",
        "## What it shows", "", what, "",
        "## Runs", "", runs_txt, "",
        "## Aggregation", "", agg, "",
        "## Numbers", "", numbers, "",
        "## Regenerate (from the repository root)", "",
        "```", "python -m aux.skill_paths.analyze_shared", "```", "",
        "Inputs: `OUT/skill_paths/paths/*.json`; outputs: this file, `OUT/skill_paths/shared/{shared_results.json,shared_sentences.json}`, `OUT/skill_paths/fig/mech_shared.{pdf,png}`.", ""])
    return text


# ---------------------------------------------------------------------------------------------------------------------------------
# figure (serif 7 pt, Okabe-Ito optimizer colours, no top/right spines)
# ---------------------------------------------------------------------------------------------------------------------------------
CAT_COLORS = {"serialize": "#332288", "parallelize": "#88CCEE", "limit_read": "#44AA99", "ask_stop": "#117733", "verify": "#DDCC77", "domain": "#CC6677",
              "format": "#AA4499", "other": "#BBBBBB"}
SPEC_COLORS = {"no concrete name": "#E0E0E0", "named, not in dev/test files": "#9ECAE1", "environment-wide": "#3182BD", "task-specific": "#D55E00"}


def draw_figure(R, pdf, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    ctx, U = R["ctx"], R["U"]
    P2 = rowmap(R["prof"]["P2"], ctx)
    Lr = rowmap(R["prof"]["L"], ctx)
    ordered = [r for r in sorted(R["runs"], key=run_order) if r.key in ctx.rk]

    def style(ax):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=6.5, length=2, pad=1)

    fig = plt.figure(figsize=(6.8, 5.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.1], width_ratios=[1.4, 1.1, 1.1], left=0.075, right=0.975, bottom=0.165, top=0.915, hspace=0.40, wspace=0.52)
    # ---- (a) shared share per skill
    axa = fig.add_subplot(gs[0, :])
    xs, x, gaps = [], 0.0, {}
    prev = None
    for r in ordered:
        if prev is not None and r.bench != prev:
            x += 0.9
        gaps.setdefault(r.bench, []).append(x)
        xs.append(x)
        x += 1.0
        prev = r.bench
    for r, xx in zip(ordered, xs):
        col = OC[OLABEL[r.opt]]
        hatch = "xxxx" if r.scratch else ("////" if r.model == "gpt-6-luna" else None)
        v = 100 * P2[r.key]["share_words"]
        axa.bar(xx, v, width=0.72, color=col, edgecolor="white" if hatch else col, hatch=hatch, linewidth=0.0 if not hatch else 0.0, zorder=2)
        axa.plot([xx - 0.36, xx + 0.36], [100 * P2[r.key]["floor_words"]] * 2, color="#222222", lw=1.1, zorder=4)
        axa.plot(xx, 100 * Lr[r.key]["share_words"], marker="o", ms=3.2, mfc="white", mec="#222222", mew=0.7, lw=0, zorder=5)
        axa.text(xx, -3.2, f"{P2[r.key]['n_sent']}", ha="center", va="top", fontsize=5.4, color="#555555")
    axa.set_xticks(xs)
    axa.set_xticklabels([("DS" if r.model == "deepseek-flash" else "Lu") + (" (s)" if r.scratch else "") for r in ordered], fontsize=6)
    axa.tick_params(axis="x", length=0, pad=1.5)
    for b, xl in gaps.items():
        axa.text(np.mean(xl), 1.015, {"tau3-bench": "tau3", AB: "AutoB", "terminal-bench-2-1": "TB2"}[b], transform=axa.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
    axa.set_ylabel("added words with a\ncounterpart elsewhere (%)", fontsize=6.8, labelpad=1)
    axa.set_ylim(0, 48)
    axa.set_xlim(-0.7, max(xs) + 0.7)
    axa.set_title("(a) share of each learned skill's added text that another run on the same benchmark also writes (bars: paraphrase level P2)", fontsize=7, loc="left", pad=17)
    style(axa)
    axa.yaxis.grid(True, color="#EEEEEE", lw=0.5, zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, label=o) for o, c in OC.items()]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="#999999", label="DeepSeek"), plt.Rectangle((0, 0), 1, 1, facecolor="#999999", edgecolor="white", hatch="////", label="Luna"),
                plt.Rectangle((0, 0), 1, 1, facecolor="#999999", edgecolor="white", hatch="xxxx", label="Luna, near-empty seed (s)"),
                plt.Line2D([], [], color="#222222", lw=1.1, label="chance floor (P2)"),
                plt.Line2D([], [], marker="o", ms=3.2, mfc="white", mec="#222222", lw=0, label="verbatim (L)")]
    axa.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=5.8, handlelength=1.4, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.0)
    axa.text(-0.55, -3.2, "n =", ha="right", va="top", fontsize=5.4, color="#555555")
    # ---- (b) control
    axb = fig.add_subplot(gs[1, 0])
    PT = R["pairs"]["P2"]
    slots = [("same optimizer, other model", "within", -0.17), ("same optimizer, other model", "cross", 0.17),
             ("same model, other optimizer", "within", 0.83), ("same model, other optimizer", "cross", 1.17),
             ("other model, other optimizer", "within", 1.83), ("other model, other optimizer", "cross", 2.17),
             ("same optimizer, same model", "cross", 3.0)]
    jit = random.Random(3)
    for cl, setting, xx in slots:
        vals = [100 * v for _, _, v in PT[cl][setting]]
        hollow = setting == "cross"
        for v in vals:
            axb.plot(xx + jit.uniform(-0.06, 0.06), v, marker="o", ms=2.6, mfc="white" if hollow else "#882255", mec="#882255", mew=0.6, lw=0, alpha=0.9, zorder=3)
        if vals:
            axb.plot([xx - 0.12, xx + 0.12], [np.mean(vals)] * 2, color="#222222", lw=1.2, zorder=4)
    axb.set_xticks([0, 1, 2, 3])
    axb.set_xticklabels(["same opt.\nother\nmodel", "same\nmodel\nother\nopt.", "other\nmodel\nother\nopt.", "same opt.\nsame\nmodel"], fontsize=5.4, linespacing=1.0)
    axb.tick_params(axis="x", length=0, pad=2)
    axb.set_ylabel("pair overlap, P2 (%)", fontsize=6.8, labelpad=1)
    axb.set_ylim(-0.5, 23)
    axb.set_xlim(-0.45, 3.4)
    axb.set_title("(b) control: pairs of runs", fontsize=7, loc="left", pad=3)
    style(axb)
    axb.legend(handles=[plt.Line2D([], [], marker="o", ms=2.6, mfc="#882255", mec="#882255", lw=0, label="same benchmark"),
                        plt.Line2D([], [], marker="o", ms=2.6, mfc="white", mec="#882255", lw=0, label="other benchmark"),
                        plt.Line2D([], [], color="#222222", lw=1.2, label="mean")], loc="upper right", frameon=False, fontsize=5.8, handletextpad=0.3, borderaxespad=0.0)
    # ---- (c) categories
    axc = fig.add_subplot(gs[1, 1])
    cb = cats_by_bench(R, "P2")
    rowsc = [(b, part) for b in list(BENCHES) + ["all"] for part in ("shared", "unique")]
    ypos, yy = [], 0.0
    for i, (b, part) in enumerate(rowsc):
        ypos.append(yy)
        yy += 1.0 if part == "shared" else 1.55
    for (b, part), y in zip(rowsc, ypos):
        c = cb[b][part]
        tot = sum(c.values())
        left = 0.0
        for cat in CAT_IDS:
            v = 100 * c.get(cat, 0) / tot if tot else 0
            axc.barh(y, v, left=left, height=0.8, color=CAT_COLORS[cat], edgecolor="white", linewidth=0.4)
            left += v
    axc.set_yticks(ypos)
    axc.set_yticklabels([f"{ {'tau3-bench': 'tau3', AB: 'AutoB', 'terminal-bench-2-1': 'TB2', 'all': 'all'}[b]} {part}" for b, part in rowsc], fontsize=5.8)
    axc.invert_yaxis()
    axc.set_xlim(0, 100)
    axc.set_xlabel("share of the words of the part (%)", fontsize=6.5, labelpad=1)
    axc.set_title("(c) categories, shared vs unique (P2)", fontsize=7, loc="left", pad=3)
    style(axc)
    axc.tick_params(axis="y", length=0)
    axc.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=CAT_COLORS[c], label=c.replace("_", " ")) for c in CAT_IDS], loc="upper center", bbox_to_anchor=(0.42, -0.21), ncol=3, frameon=False,
               fontsize=5.4, handlelength=0.9, columnspacing=0.7, handletextpad=0.3)
    # ---- (d) AutoBench specificity
    axd = fig.add_subplot(gs[1, 2])
    spec = spec_summary(R)
    keys_ab = [r.key for r in ordered if r.bench == AB]
    labs = [next(r for r in ordered if r.key == k) for k in keys_ab]
    ylabels, rows_d = [], []
    for r in labs:
        s = spec[r.key]
        rows_d.append([s["cnt"][c] / s["n"] for c in SPEC_CLASSES])
        ylabels.append(("DS " if r.model == "deepseek-flash" else "Lu ") + OLABEL[r.opt] + (" (s)" if r.scratch else ""))
    seedc = collections.Counter(t["class"] for t in R["seed_tags"])
    rows_d.append([seedc[c] / len(R["seed_tags"]) for c in SPEC_CLASSES])
    ylabels.append("seed (control)")
    for i, row in enumerate(rows_d):
        left = 0.0
        for c, v in zip(SPEC_CLASSES, row):
            axd.barh(i, 100 * v, left=left, height=0.72, color=SPEC_COLORS[c], edgecolor="white", linewidth=0.4)
            left += 100 * v
    axd.set_yticks(range(len(rows_d)))
    axd.set_yticklabels(ylabels, fontsize=5.8)
    axd.invert_yaxis()
    axd.set_xlim(0, 100)
    axd.set_xlabel("added sentences (%)", fontsize=6.5, labelpad=1)
    axd.set_title("(d) AutoBench specificity", fontsize=7, loc="left", pad=3)
    style(axd)
    axd.tick_params(axis="y", length=0)
    axd.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=SPEC_COLORS[c], label=c.replace("named, not in dev/test files", "named, not in task files")) for c in SPEC_CLASSES], loc="upper center", bbox_to_anchor=(0.30, -0.21), ncol=2, frameon=False, fontsize=5.4,
               handlelength=0.9, columnspacing=0.7, handletextpad=0.3)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(pdf, metadata={"CreationDate": None, "Creator": None, "Producer": None, "Title": "mech_shared"})
    fig.savefig(png, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------------------------------------------------------------
def write_json(R):
    ctx = R["ctx"]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    res = {"thresholds": {"L_J4": TAU_L, "S_C3": TAU_S, "S_min_words": S_MIN_WORDS, "P2_cos": TAU_P2, "P3_cos": TAU_P3, "min_tokens": MIN_TOK},
           "n_sentences": len(R["U"]), "checks": R["checks"],
           "runs": {}, "pooled": {}, "sweeps": R["sweeps"], "calibration": R["cal"], "pairs": {}, "scratch_pairs": R["scratch_pairs"], "proposal_level": {}, "lex": R["lex"], "panels": R["panels"]}
    for kind in KINDS:
        prof = R["prof"][kind]
        for row, c, e in zip(prof["runs"], R["cats"][kind][0], R["seed"][kind]):
            r = ctx.runs[row["q"]]
            d = res["runs"].setdefault(r.key, {"label": r.label, "bench": r.bench, "partial": r.partial, "scratch": r.scratch})
            d[kind] = {"n_sent": row["n_sent"], "words": row["words"], "sent_shared": row["sent_shared"], "words_shared": row["words_shared"], "share_words": row["share_words"],
                       "share_sent": row["share_sent"], "floor_words": row["floor_words"], "floor_k": row["floor_k"], "floor_pool": row["floor_pool"], "categories": c,
                       "seed_echo_share_words": e["share_words"]}
        res["pooled"][kind] = {"all": pooled_stats(prof, ctx), **{b: pooled_stats(prof, ctx, {b}) for b in BENCHES}}
        res["pairs"][kind] = {cl: {s: [list(x) for x in v] for s, v in d.items()} for cl, d in R["pairs"][kind].items()}
        profP = R["P"]["prof"][kind]
        res["proposal_level"][kind] = {R["P"]["ctx"].runs[row["q"]].key: {"n_sent": row["n_sent"], "words": row["words"], "share_words": row["share_words"], "floor_words": row["floor_words"]}
                                       for row in profP["runs"]}
    res["task_specificity"] = {k: {"classes": dict(collections.Counter(t["class"] for t in v["tags"])), "audit": {pn: {kk: (vv if not isinstance(vv, set) else sorted(vv)) for kk, vv in a.items()}
                                                                                                                   for pn, a in v["audit"].items()}} for k, v in R["spec"].items()}
    res["closest_to_seed"] = {k: {"by_cos": [[round(c, 3), round(j, 3), round(cc, 3), R["U"][i]["text"], R["U"][jj]["text"]] for c, j, cc, i, jj in v["by_cos"]],
                                  "by_J": [[round(c, 3), round(j, 3), round(cc, 3), R["U"][i]["text"], R["U"][jj]["text"]] for c, j, cc, i, jj in v["by_J"]]} for k, v in R["closest"].items()}
    (OUTDIR / "shared_results.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    sent = []
    for row in R["sentence_rows"]:
        q = ctx.rk[row["run"]]
        sent.append(row)
    spec_rows = {}
    for k, v in R["spec"].items():
        spec_rows[k] = [{"text": t["text"], "class": t["class"], "tokens": t["tokens"]} for t in v["tags"]]
    (OUTDIR / "shared_sentences.json").write_text(json.dumps({"sentences": sent, "autobench_specificity": spec_rows}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--skip-verify-dedupe", action="store_true", help="skip the fidelity check of the corpus builder against cheatsheet.load_bench (slow); the check line then disappears from the record")
    a = ap.parse_args()
    R = compute(verify_dedupe=not a.skip_verify_dedupe)
    write_json(R)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(build_record(R))
    if not a.no_fig:
        draw_figure(R, FIGDIR / "mech_shared.pdf", FIGDIR / "mech_shared.png")
    print(f"wrote {RECORD}, {OUTDIR}/shared_results.json, shared_sentences.json" + ("" if a.no_fig else f", {FIGDIR}/mech_shared.pdf/.png"))


if __name__ == "__main__":
    main()
