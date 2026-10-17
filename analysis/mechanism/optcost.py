"""Optimization cost and skill statistics of the optimizer runs (DeepSeek-V4.1-Flash).

Paper elements: tab:optcost (appendix C: episodes consumed and executed on the training sample and on V, agent tokens, optimizer calls and tokens,
largest optimizer prompt, prompts cut to the limit), the training-cost cells quoted in facts (Sec. 2), and the returned-skill statistics (words, blocks,
changed lines). cost.json also holds the test-episode cost ratios (requests, calls, input and output tokens, USD at assumed list prices).

Sources: evaluation files of the train / dev / test splits of each run (per-episode requests), summary.json (episodes, arm_episodes, llm),
llm/call-*.json (provider usage of every optimizer call). No model calls.
Outputs (assets/cache/analysis/optcost/): training-cost-cells.csv, training-cost.json, training-cost.tex, skills-raw.csv, skills.json, skills.tex,
skills-diff-excerpts.txt, llm.json, cost.json, cost_table.tex, optcost.tex.
"""
import argparse, csv, difflib, json, math, re, statistics as st
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from analysis.common import *
from graph.episode import classify

MODEL = "deepseek-flash"
OPT_METHODS = ("gepa", "skillopt", "skilladam")
NAME = {"native-noseed": "No skill", "native-seed": "Seed skill", "gepa": "GEPA", "skillopt": "SkillOpt", "skilladam": "SkillAdam"}
# assumed list prices, USD per 1M tokens
PRICES = {"main": {MODEL: dict(cached=0.028, uncached=0.28, out=0.42)}}
PRICES["nocache"] = {m: dict(cached=p["uncached"], uncached=p["uncached"], out=p["out"]) for m, p in PRICES["main"].items()}

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# per-episode extraction

def grp(comp):
    if comp.startswith("read.") or comp == "lookup": return "read"
    if comp in ("grep", "glob", "KB_search"): return "search"
    if comp.startswith("bash"): return "bash"
    if comp in ("act", "edit", "write"): return "act"
    return "other"   # msg, conv, discover, todo, other

def fin(v): return isinstance(v, (int, float)) and math.isfinite(v) and v >= 0

def ep_rec(ep):
    cap, seen, reqs = ep.get("capture") or {}, set(), []
    for r in ep.get("requests", []):
        if r["request_id"] not in seen: seen.add(r["request_id"]); reqs.append(r)
    calls = [c for r in reqs for c in r.get("tool_calls", [])]
    ex = [c for c in calls if c.get("executed") is True]
    n = len(ex)
    if cap.get("invocations_complete") is not True or any(c.get("executed") not in (True, False) for c in calls): n = None
    rc = cap.get("requests_complete") is True and all(fin(r["input_tokens"]) for r in reqs)
    X = sum(r["input_tokens"] for r in reqs) if rc else None
    out = cache = None
    if rc:
        us = [r.get("captured_usage") or {} for r in reqs]
        if all(fin(r.get("output_tokens")) for r in reqs): out = sum(r["output_tokens"] for r in reqs)
        if all(fin(u.get("cache_tokens")) for u in us): cache = sum(u["cache_tokens"] for u in us)
    g = {"read": 0, "search": 0, "bash": 0, "act": 0, "other": 0}; comps = {}
    for c in ex:
        comp, _ = classify(c["tool"], c.get("arguments") or ""); g[grp(comp)] += 1; comps[comp] = comps.get(comp, 0)+1
    return dict(success=ep.get("success") if ep.get("outcome_valid", True) else None, N=n, X=X, out=out, cache=cache, nreq=len(reqs), g=g, comps=comps)

def file_recs(path):
    d = json.loads(Path(path).read_text()); sa = d.get("slot_accounting") or {}
    eps = {}
    for e in d["episodes"]: eps.setdefault((e["task_id"], e.get("repetition", 0)), e)
    panel = [(t["task_id"], t.get("repetition", 0)) for t in d["panel"]]
    return dict(path=str(path), panel={k: ep_rec(eps[k]) if k in eps else None for k in panel}, all=[ep_rec(e) for e in d["episodes"]], sa=sa)

def jobs(model):
    J = {}
    for bench in BENCHES:
        for m in METHODS:
            run = current_run(m, model, bench)
            if not run or not (run/"summary.json").is_file(): continue
            t = final_skill(m, run); seedret = t == SEED_TEXT
            tr = current_run("native-seed", model, bench) if (seedret or m == "native-seed") else run
            v = vid(SEED_TEXT) if (seedret or m == "native-seed") else vid(t)
            J[(model, bench, m)] = dict(run=run, test=eval_file(tr, "test", v), seedret=seedret and m != "native-seed", skill_words=len(t.split()) if t else 0)
            if m in OPT_METHODS:
                J[(model, bench, m)]["train"] = sorted((run/"results/evaluations/train").glob("*/evaluation.json"))
                J[(model, bench, m)]["dev"] = sorted((run/"results/evaluations/dev").glob("*/evaluation.json"))
    return J

def llm_usage(run):
    """Provider usage of the optimizer calls of a run: sums over llm/call-*.json (calls without a usage record are counted in `missing`)."""
    calls = sorted((run/"llm").glob("call-*.json")); pt = ct = cached = n = miss = mx = over = 0
    for f in calls:
        r = json.loads(f.read_text()).get("response") or {}
        u = r.get("usage") if "usage" in r else r
        if isinstance(u, dict) and "prompt_tokens" in u:
            n += 1; pt += u["prompt_tokens"]; ct += u.get("completion_tokens", 0); mx = max(mx, u["prompt_tokens"]); over += u["prompt_tokens"] > 400_000
            cached += u.get("prompt_cache_hit_tokens") or ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        else: miss += 1
    return dict(calls=len(calls), with_usage=n, prompt=pt, completion=ct, cached=cached, missing=miss, max_prompt=mx, n_over_400k=over)

def extract(model):
    J = jobs(model); paths = []
    for j in J.values():
        if j["test"]: paths.append(j["test"])
        paths += j.get("train", []) + j.get("dev", [])
    paths = sorted(set(paths), key=str)
    with ProcessPoolExecutor(16) as pool: res = dict(zip(map(str, paths), pool.map(file_recs, paths, chunksize=1)))
    out = {}
    for k, j in J.items():
        o = dict(run=j["run"].name, seedret=j["seedret"], skill_words=j["skill_words"], test=res[str(j["test"])] if j["test"] else None)
        s = json.loads((j["run"]/"summary.json").read_text()); o["summary_episodes"] = s.get("episodes"); o["arm_episodes"] = s.get("arm_episodes")
        if "train" in j:
            o["train"] = [res[str(p)] for p in j["train"]]; o["dev"] = [res[str(p)] for p in j["dev"]]; o["llm"] = llm_usage(j["run"])
            o["llm_model"] = json.loads(sorted((j["run"]/"llm").glob("call-*.json"))[0].read_text())["request"]["model"]
        out[k] = o
    return out

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# cost.json: test-episode metrics, ratios to the Seed row, training cost per cell

def usd(model, X, cache, out, sc="main"):
    if X is None or cache is None or out is None: return None
    p = PRICES[sc][model]; return ((X-cache)*p["uncached"] + cache*p["cached"] + out*p["out"]) / 1e6
def gm(v):
    v = [x for x in v if x]
    return gmean(v)
def mean(v): v = [x for x in v if x is not None]; return sum(v)/len(v) if v else None
def med(v): v = [x for x in v if x is not None]; return st.median(v) if v else None
def met(r, model):
    return dict(req=r["nreq"], calls=r["N"], inp=r["X"], out=r["out"], cache=r["cache"], usd=usd(model, r["X"], r["cache"], r["out"]), usd_nc=usd(model, r["X"], r["cache"], r["out"], "nocache"))
KEYS = ("req", "calls", "inp", "out", "usd", "usd_nc")

def cost_cells(D, model):
    res = {}
    for bench in BENCHES:
        seed = D[(model, bench, "native-seed")]["test"]["panel"]
        for m in METHODS:
            o = D.get((model, bench, m))
            if not o: continue
            rows = o["test"]["panel"]
            ks = [k for k, r in rows.items() if r]
            M = {k: met(rows[k], model) for k in ks}
            ab = {}
            for key in KEYS + ("cache",):
                v = [M[k][key] for k in ks]
                ab[key] = dict(mean=mean(v), median=med(v), n=sum(x is not None for x in v), gmean=gm(v))
            ab["cache_frac_median"] = med([M[k]["cache"]/M[k]["inp"] for k in ks if M[k]["cache"] is not None and M[k]["inp"]])
            # paired ratios vs Seed on the Table-1 common set (verified outcome both sides, N,X>0 both)
            rat = {key: [] for key in KEYS}; common_ = []
            for k in ks:
                r, b = rows[k], seed.get(k)
                if not b or r["success"] is None or b["success"] is None: continue
                if not all(x and x > 0 for x in (r["N"], r["X"], b["N"], b["X"])): continue
                common_.append(k); mb = met(b, model)
                for key in KEYS:
                    if M[k][key] and mb[key]: rat[key].append(math.log(M[k][key]/mb[key]))
            rho = {key: gmean(np.exp(v)) for key, v in rat.items()}
            gsum = {g: sum(rows[k]["g"][g] for k in ks if rows[k]["N"] is not None) for g in ("read", "search", "bash", "act", "other")}
            tot = sum(gsum.values()) or 1
            Ns = [rows[k]["N"] for k in ks if rows[k]["N"] is not None]
            succ = [rows[k]["success"] for k in ks if rows[k]["success"] is not None]
            beh = dict(share={g: v/tot for g, v in gsum.items()}, N_p10=float(np.percentile(Ns, 10)), N_p90=float(np.percentile(Ns, 90)),
                       frac_N_le3=sum(n <= 3 for n in Ns)/len(Ns), frac_N_ge40=sum(n >= 40 for n in Ns)/len(Ns), req_per_call=gm([rows[k]["nreq"]/rows[k]["N"] for k in ks if rows[k]["N"]]),
                       calls_per_req=mean([rows[k]["N"]/rows[k]["nreq"] for k in ks if rows[k]["N"] is not None and rows[k]["nreq"]]),
                       out_per_req=mean([rows[k]["out"]/rows[k]["nreq"] for k in ks if rows[k]["out"] is not None and rows[k]["nreq"]]),
                       comps={c: sum(rows[k]["comps"].get(c, 0) for k in ks if rows[k]["N"] is not None)/tot for c in sorted({c for k in ks for c in rows[k]["comps"]})},
                       succ=sum(s is True for s in succ), known=len(succ))
            tr = None
            if "train" in o:
                tr = {}
                for sp in ("train", "dev"):
                    fs = o[sp]; eps = [e for f in fs for e in f["all"]]
                    slots = sum(f["sa"].get("slots", len(f["panel"])) for f in fs); exe = sum(f["sa"].get("executed", f["sa"].get("slots", len(f["panel"]))) for f in fs)
                    def tot_(key, fresh=False):   # fresh share per file = executed / slots
                        s = 0
                        for f in fs:
                            sl = f["sa"].get("slots", len(f["panel"])) or 1; fr = f["sa"].get("executed", sl)/sl if fresh else 1
                            s += fr*sum(e[key] or 0 for e in f["all"])
                        return s
                    def usd_tot(fresh=False):
                        s = 0
                        for f in fs:
                            sl = f["sa"].get("slots", len(f["panel"])) or 1; fr = f["sa"].get("executed", sl)/sl if fresh else 1
                            s += fr*sum(usd(model, e["X"], e["cache"], e["out"]) or 0 for e in f["all"])
                        return s
                    se = (o["summary_episodes"] or {}).get(sp) or {}
                    tr[sp] = dict(consumed=se.get("episodes", 0), fresh=se.get("executed", 0), file_slots=slots, file_fresh=exe,
                                  N=tot_("N"), req=tot_("nreq"), X=tot_("X"), out=tot_("out"), cache=tot_("cache"),
                                  N_fresh=tot_("N", True), X_fresh=tot_("X", True), out_fresh=tot_("out", True), usd=usd_tot(), usd_fresh=usd_tot(True),
                                  unknown=sum(1 for e in eps if e["X"] is None))
                l = o["llm"]; p = PRICES["main"][model]
                tr["llm"] = dict(l, usd=((l["prompt"]-l["cached"])*p["uncached"] + l["cached"]*p["cached"] + l["completion"]*p["out"])/1e6, model=o["llm_model"])
            res[f"{model}|{bench}|{m}"] = dict(run=o["run"], returned_seed=o["seedret"], skill_words=o["skill_words"], n_test=len(ks), n_common=len(common_), abs=ab, rho=rho,
                                               n_pair={key: len(v) for key, v in rat.items()}, beh=beh, train=tr)
    return res

def aggregates(C, D, model):
    """Geometric mean over benchmarks of the per-cell ratios, training cost pooled over tau3 and AutoBench, USD composition of the Seed row's test episodes."""
    agg = {}
    for m in METHODS:
        cells = [C[f"{model}|{b}|{m}"] for b in BENCHES if f"{model}|{b}|{m}" in C]
        a = {k: gm([c["rho"][k] for c in cells]) for k in ("req", "calls", "inp", "out", "usd", "usd_nc")}
        a["n_cells"] = len(cells); a["cells_equal_seed"] = sum(c["returned_seed"] for c in cells)
        agg[f"{model}|{m}"] = a
    def pool(m, benches=("tau3-bench", "automationbench")):
        t = dict(tr=0, trf=0, dv=0, dvf=0, N=0, req=0, X=0, out=0, calls=0, ltok=0, lusd=0, usd=0, usdf=0, cache=0)
        for b in benches:
            c = C[f"{model}|{b}|{m}"]["train"]
            for sp, (a, f) in (("train", ("tr", "trf")), ("dev", ("dv", "dvf"))):
                t[a] += c[sp]["consumed"]; t[f] += c[sp]["fresh"]
                for k in ("N", "req", "X", "out", "cache"): t[k] += c[sp][k]
                t["usd"] += c[sp]["usd"]; t["usdf"] += c[sp]["usd_fresh"]
            l = c["llm"]; t["calls"] += l["calls"]; t["ltok"] += l["prompt"] + l["completion"]; t["lusd"] += l["usd"]
        return t
    train = {f"{model}|{m}": pool(m) for m in OPT_METHODS if all(f"{model}|{b}|{m}" in C for b in ("tau3-bench", "automationbench"))}
    comp = {}; p = PRICES["main"][model]
    for b in BENCHES:
        rows = [r for r in D[(model, b, "native-seed")]["test"]["panel"].values() if r and r["X"] is not None and r["cache"] is not None and r["out"] is not None]
        X = sum(r["X"] for r in rows); ca = sum(r["cache"] for r in rows); o = sum(r["out"] for r in rows)
        parts = dict(cached=ca*p["cached"], uncached=(X-ca)*p["uncached"], out=o*p["out"]); s = sum(parts.values())
        comp[f"{model}|{b}"] = {k: v/s for k, v in parts.items()}
    return agg, train, comp

def cost_table_tex(agg, train, model):
    f2 = lambda v: "\\tbd{}" if v is None else f"{v:.2f}"
    L = ["% Cost of the returned skills. Per-task ratio to the Seed row on the Table-1 common task set, geometric mean over tasks, then over the benchmarks a method finished.",
         "\\begin{tabular}{@{}lrrrrrc@{}}\\toprule", "Method & Requests & Tool calls & Input tok. & Output tok. & USD$^\\ast$ & $=$Seed \\\\ \\midrule"]
    for m in METHODS:
        a = agg[f"{model}|{m}"]
        if m == "native-seed": row = ["1.00"]*5 + ["--"]
        else: row = [f2(a[k]) for k in ("req", "calls", "inp", "out", "usd")] + [f"{a['cells_equal_seed']}/{a['n_cells']}" if m != "native-noseed" else "--"]
        if m == "gepa": L.append("\\addlinespace[2pt]")
        L.append(NAME[m] + " & " + " & ".join(row) + " \\\\")
    L += ["\\bottomrule\\end{tabular}", "", "\\begin{tabular}{@{}lrrrrrr@{}}\\toprule",
          "Method & \\multicolumn{2}{c}{Rollout episodes} & Agent tok. & Opt.\\ LLM & \\multicolumn{2}{c}{USD$^\\ast$} \\\\",
          " & train (fresh) & dev (fresh) & in/out (M) & calls / M tok. & rollouts & opt.\\ LLM \\\\ \\midrule"]
    for m in OPT_METHODS:
        if f"{model}|{m}" not in train: continue
        t = train[f"{model}|{m}"]
        dev = "n/a" if m == "skilladam" else f"{t['dv']} ({t['dvf']})"
        L.append(NAME[m] + f" & {t['tr']} ({t['trf']}) & {dev} & {t['X']/1e6:.0f}/{t['out']/1e6:.1f} & {t['calls']} / {t['ltok']/1e6:.1f} & {t['usd']:.0f} & {t['lusd']:.1f} \\\\")
    L += ["\\bottomrule\\end{tabular}"]
    return "\n".join(L) + "\n"

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# training-cost cells (episode and token counts per run), skills, optimizer LLM calls

def split_cost(run, split):
    tot = dict(files=0, slots=0, executed=0, hits=0, N=0, X=0, Nex=0.0, Xex=0.0, unk=0, mt=[])
    for f in (run/"results/evaluations"/split).glob("*/evaluation.json"):
        d = json.loads(f.read_text()); sa = d.get("slot_accounting") or {}
        n = x = 0
        for e in d["episodes"]:
            r = task_row(e); n += r["N"] or 0; x += r["X"] or 0; tot["unk"] += r["N"] is None
        sl = sa.get("slots") or len(d["panel"]); fr = (sa.get("executed", sl))/sl if sl else 1
        tot["files"] += 1; tot["slots"] += sl; tot["executed"] += sa.get("executed", sl); tot["hits"] += sa.get("cache_hits", 0)
        tot["N"] += n; tot["X"] += x; tot["Nex"] += n*fr; tot["Xex"] += x*fr; tot["mt"].append(f.stat().st_mtime)
    return tot

def training_cells(model):
    rows = []
    for bench in BENCHES:
        for m in OPT_METHODS:
            run = current_run(m, model, bench)
            rec = dict(model=model, bench=bench, method=m, run=run.name if run else None)
            if not run or not (run/"summary.json").is_file(): rec["status"] = "none"; rows.append(rec); continue
            s = json.loads((run/"summary.json").read_text()); mt = []
            for sp in ("train", "dev"):
                c = split_cost(run, sp); mt += c["mt"]
                ep = (s.get("episodes") or {}).get(sp) or {}
                rec.update({f"{sp}_episodes": ep.get("episodes", 0), f"{sp}_executed": ep.get("executed", 0), f"{sp}_evalfile_slots": c["slots"], f"{sp}_evalfile_executed": c["executed"],
                            f"{sp}_N_all": c["N"], f"{sp}_X_all": c["X"], f"{sp}_N_exec_approx": round(c["Nex"]), f"{sp}_X_exec_approx": round(c["Xex"])})
            rec["agent_X_all_train+dev"] = rec["train_X_all"]+rec["dev_X_all"]; rec["agent_N_all_train+dev"] = rec["train_N_all"]+rec["dev_N_all"]
            rec["agent_X_exec_approx"] = rec["train_X_exec_approx"]+rec["dev_X_exec_approx"]; rec["agent_N_exec_approx"] = rec["train_N_exec_approx"]+rec["dev_N_exec_approx"]
            u = llm_usage(run)
            rec.update(llm_calls_files=u["calls"], llm_prompt_tokens=u["prompt"], llm_completion_tokens=u["completion"], llm_calls_without_usage=u["missing"])
            sl = s.get("llm") or {}
            rec.update(summary_llm_calls=sl.get("calls"), summary_llm_prompt=sl.get("prompt_tokens"), summary_llm_completion=sl.get("completion_tokens"), summary_llm_replayed=sl.get("replayed"))
            rec["wall_hours_approx"] = round((max(mt)-min(mt))/3600, 1) if mt else None
            rec["final_words"] = len(final_skill(m, run).split())
            rows.append(rec)
    return rows

def training_cost(R, model):
    """training-cost.json / .tex: pooled over the benchmarks a method finished (SkillAdam has no TB2 run in the earlier cell files)."""
    cells = {}; pool = {}
    for r in R:
        if not r.get("train_episodes"): continue
        d = dict(train_consumed=int(r["train_episodes"]), train_fresh=int(r["train_executed"]), dev_consumed=int(r["dev_episodes"]), dev_fresh=int(r["dev_executed"]),
                 agent_X_M_all=float(r["agent_X_all_train+dev"])/1e6, agent_X_M_fresh_approx=float(r["agent_X_exec_approx"])/1e6, agent_N_all=int(float(r["agent_N_all_train+dev"])),
                 llm_calls=int(r["llm_calls_files"]), llm_prompt_tokens=int(r["llm_prompt_tokens"]), llm_completion_tokens=int(r["llm_completion_tokens"]), run=r["run"])
        cells[f"{r['model']}|{r['bench']}|{r['method']}"] = d
        p = pool.setdefault((r["model"], r["method"]), dict(cells=0)); p["cells"] += 1
        for k, v in d.items():
            if k != "run": p[k] = p.get(k, 0)+v
    js = {"cells": cells, "pooled_over_benchmarks": {f"{m}|{me}": v for (m, me), v in pool.items()},
          "notes": "train/dev consumed = episodes the optimizer consumed from its budget (summary.json episodes.*.episodes, cache hits included); fresh = executed (not cache hits). agent_X_M_all: input tokens of all consumed train+dev episodes at their original cost; fresh_approx scales each evaluation file by executed/slots. LLM = llm/call-*.json (summary.json under-reports on resumed runs)."}
    L = ["% Training cost pooled over the benchmarks a method finished. Episodes: consumed (fresh = executed, rest are cache hits); X: agent input tokens of train+dev rollouts, cache hits at original cost; LLM: optimizer calls and tokens (prompt+completion).",
         "Method & Train eps (fresh) & Dev eps (fresh) & Agent input (M tok) & Opt. LLM calls & Opt. LLM (M tok) \\\\"]
    for me in OPT_METHODS:
        p = pool.get((model, me))
        if not p: continue
        nm = NAME[me]
        L.append(f"{nm}{'$^\\ddagger$' if me == 'skilladam' else ''} & {p['train_consumed']} ({p['train_fresh']}) & {p['dev_consumed']} ({p['dev_fresh']}) & {p['agent_X_M_all']:.0f} & {p['llm_calls']} & {(p['llm_prompt_tokens']+p['llm_completion_tokens'])/1e6:.1f} \\\\")
    return js, "\n".join(L) + "\n"

SUMMARY = {("tau3-bench", "gepa"): "Full rewrite: evidence before claims and before dependent actions (gate function, do not act on unproven authorization); drops code-centric overview.",
           ("tau3-bench", "skillopt"): "Append-only: rows/red flags for unexecuted actions and projections stated as fact; Tool-Use and Task-Resolution extension.",
           ("tau3-bench", "skilladam"): "Adds tool-task verification: grounded claims, grounded preconditions (re-read before mutation), grounded completion (re-read after).",
           ("automationbench", "skillopt"): "Appends blocker handling (use allowed alternative, disclose unmet step) and verify-the-outcome rules.",
           ("automationbench", "skilladam"): "Generalizes IDENTIFY/RUN/READ to actions and re-read records; adds Scope Before Executing (enumerate inputs, read all governing sources).",
           ("terminal-bench-2-1", "skillopt"): "Large append (+154 lines): independent checks of constraints, hash inputs, repeated trials, verify the verifier, TB2-specific rules."}

def skills(model):
    """Returned skill vs the seed: words, blocks, headings, changed lines; the line diffs go to skills-diff-excerpts.txt."""
    blocks = lambda t: [b for b in re.split(r"\n\s*\n", t) if b.strip()]
    heads = lambda t: sum(1 for l in t.splitlines() if l.startswith("#"))
    rows, txt = [], []
    for bench in BENCHES:
        for m in OPT_METHODS:
            run = current_run(m, model, bench)
            if not run or not (run/"summary.json").is_file(): continue
            t = final_skill(m, run); same = t == SEED_TEXT
            diff = list(difflib.unified_diff(SEED_TEXT.splitlines(), t.splitlines(), lineterm="", n=0))
            add = [l for l in diff if l.startswith("+") and not l.startswith("+++")]
            rem = [l for l in diff if l.startswith("-") and not l.startswith("---")]
            rows.append(dict(model=model, bench=bench, method=m, is_seed=same, words=len(t.split()), seed_words=len(SEED_TEXT.split()), blocks=len(blocks(t)), seed_blocks=len(blocks(SEED_TEXT)),
                             headings=heads(t), seed_headings=heads(SEED_TEXT), lines_added=len(add), lines_removed=len(rem)))
            if not same:
                txt.append(f"=== {model} {bench} {m}  words {len(t.split())} (seed {len(SEED_TEXT.split())}) +{len(add)}/-{len(rem)} lines")
                txt += ["  ADD "+l[1:][:260] for l in add[:14]] + ["  REM "+l[1:][:200] for l in rem[:8]]
    js, Lt = {}, ["% Returned skill vs seed (580 words, 27 blocks). Words / blocks; 'seed' = byte-identical to the seed.", "Cell & Method & Words & Blocks & Essential change \\\\"]
    for r in rows:
        k = (r["bench"], r["method"])
        js[f"{r['model']}|{r['bench']}|{r['method']}"] = dict(is_seed=r["is_seed"], words=r["words"], seed_words=580, blocks=r["blocks"], seed_blocks=27,
                                                           summary="returned the seed text unchanged" if r["is_seed"] else SUMMARY.get(k, ""))
        Lt.append(f"DeepSeek {BN[r['bench']].replace('AutoB', 'AutoBench')} & {NAME[r['method']]} & {r['words']} & {r['blocks']} & {'seed (unchanged)' if r['is_seed'] else SUMMARY.get(k, '')} \\\\")
    return rows, "\n".join(txt), js, "\n".join(Lt) + "\n"

def optimizer_calls(model):
    """llm.json: per run, the optimizer's calls and tokens from llm/call-*.json next to the totals in summary.json (the two differ on resumed runs)."""
    out = {}
    for m in OPT_METHODS:
        for bench in BENCHES:
            run = current_run(m, model, bench)
            if not run or not (run/"summary.json").is_file(): continue
            u = llm_usage(run)
            out[f"{m}|{bench}"] = dict(files=u["calls"], sum_prompt=u["prompt"], sum_compl=u["completion"], max_prompt=u["max_prompt"], n_over_400k=u["n_over_400k"],
                                       summary_llm=json.loads((run/"summary.json").read_text()).get("llm"))
    return out

def optcost_tex(C, L, model):
    """tab:optcost rows: training and V episodes consumed (executed), agent tokens, optimizer calls, tokens, largest prompt, prompts cut."""
    names = {"gepa": "GEPA", "skillopt": "SkillOpt", "skilladam": "SkillAdam"}; bn = {"tau3-bench": "\\taubench{}", "automationbench": "AutoBench", "terminal-bench-2-1": "TB2"}
    rows = []
    for m in names:
        for i, bench in enumerate(BENCHES):
            c = C.get(f"{model}|{bench}|{m}"); l = L.get(f"{m}|{bench}")
            if not c or not l: continue
            t, d = c["train"]["train"], c["train"]["dev"]
            dev = "--" if m == "skilladam" else f"{d['consumed']} ({d['fresh']})"
            cut = (l["summary_llm"] or {}).get("truncated_calls", 0)
            rows.append(f"{names[m] if i == 0 else ''} & {bn[bench]} & {t['consumed']} ({t['fresh']}) & {dev} & {(t['X']+d['X'])/1e6:.0f} / {(t['out']+d['out'])/1e6:.1f} & {l['files']} & "
                        f"{l['sum_prompt']/1e6:.2f} / {l['sum_compl']/1e6:.2f} & {l['max_prompt']/1e3:.0f} & {cut} \\\\")
        rows.append("\\addlinespace[2pt]")
    return "\n".join(rows[:-1]) + "\n"

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(out_dir("optcost")))
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)
    D = extract(MODEL)
    C = cost_cells(D, MODEL)
    agg, train, comp = aggregates(C, D, MODEL)
    (out/"cost.json").write_text(json.dumps(dict(prices=PRICES, cells=C, aggregate_test_ratio=agg, training_pooled_tau3_autob=train, seed_usd_composition=comp), indent=1, default=float))
    (out/"cost_table.tex").write_text(cost_table_tex(agg, train, MODEL))
    R = training_cells(MODEL)
    keys = []; [keys.append(k) for r in R for k in r if k not in keys]
    with open(out/"training-cost-cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(R)
    js, tx = training_cost([{k: ("" if v is None else v) for k, v in r.items()} for r in R], MODEL)
    (out/"training-cost.json").write_text(json.dumps(js, indent=1)); (out/"training-cost.tex").write_text(tx)
    rows, excerpts, sj, st_tex = skills(MODEL)
    with open(out/"skills-raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    (out/"skills-diff-excerpts.txt").write_text(excerpts); (out/"skills.json").write_text(json.dumps(sj, indent=1)); (out/"skills.tex").write_text(st_tex)
    L = optimizer_calls(MODEL)
    (out/"llm.json").write_text(json.dumps(L, indent=1))
    (out/"optcost.tex").write_text(optcost_tex(C, L, MODEL))
    print((out/"optcost.tex").read_text())

if __name__ == "__main__": main()
