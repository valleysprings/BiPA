"""Optimization cost of the optimizer runs (DeepSeek-V4.1-Flash): the optimizer-cost table (episodes consumed and executed on the
training sample and on the dev set, agent tokens, optimizer calls and tokens, largest optimizer prompt, prompts cut to the prompt limit).

Sources: evaluation files of the train / dev / test splits of each run (per-episode requests), summary.json (episodes, arm_episodes, llm),
llm/call-*.json (provider usage of every optimizer call). No model calls.
Run:  python -m aux.cost_appendix.training_cost   -> assets/cache/analysis/optcost/llm.json and optcost.tex
"""
import argparse, json, math, statistics as st
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from src.utils.dataloader import *
from src.graph.episode import classify

MODEL = "deepseek-flash"
OPT_METHODS = ("gepa", "skillopt", "skilladam")
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
# per-cell episode metrics and the training cost of each run

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

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# optimizer LLM calls and the optimizer-cost table

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
    """Optimizer-cost table rows: training and V episodes consumed (executed), agent tokens, optimizer calls, tokens, largest prompt, prompts cut."""
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
    C = cost_cells(extract(MODEL), MODEL)
    L = optimizer_calls(MODEL)
    (out/"llm.json").write_text(json.dumps(L, indent=1))
    (out/"optcost.tex").write_text(optcost_tex(C, L, MODEL))
    print((out/"optcost.tex").read_text())

if __name__ == "__main__": main()
