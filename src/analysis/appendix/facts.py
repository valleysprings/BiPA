"""Facts for the paper's prose and appendix (DeepSeek-V4.1-Flash, no BACH rows): output tokens and price-weighted cost, within-task price slopes,
compaction scan, filter counts and sample sizes. Writes facts-v9.md (main) from the numbers below.

f1  Sec. 3.4 / 5 numbers: output share of tokens and of price (cache reads 0.1x, output 4x the uncached input price), within-task slopes of log y on log requests
    for input, output and price (task fixed effects); main runs and all runs (reruns included).
f4  appendix, compaction: scan of all stored DeepSeek episodes for a request whose input shrinks versus the previous request (history compaction would show so).
f6  filter counts on V (the selection split): R >= 2, >= 1 executed call, positive history part; f6b the same on the test files with the e_R range.
render  facts-v9.md: the sections above plus the optimizer-cost, benchmark and budget sections (their numbers come from optcost.py, tab:optcost and its llm.json).
Outputs (assets/cache/analysis/facts/): f1.json, f4.json, f6.json, f6b.json, facts-v9.md.
"""
import argparse, collections, json
import numpy as np
from src.utils.dataloader import *
from src.graph import episode as W
from src.analysis.appendix.slopes import rerun_files, fixed_part, MODEL, DS_ROOT

# ---- f1: output tokens and price ------------------------------------------------------------------------------------------------------------------------

def eprice(reqs, Xrg_rows=None, mode="meas", P=(1.0, 0.1, 4.0)):
    """Price of an episode in units of the uncached input price: uncached input u, re-sent input c, output o per token. The re-sent part is the measured cache read
    (meas), all of the previous request's input (prev) or the exact component carry-over (xrg); request 1 re-sends nothing."""
    u, c, o = P; tot = 0.0; prev = 0
    for t, r in enumerate(reqs):
        X = r["input_tokens"]; out = r["output_tokens"]
        if mode == "meas": res = (r.get("captured_usage") or {}).get("cache_tokens") or 0
        elif mode == "prev": res = min(prev, X) if t > 0 else 0
        elif mode == "xrg": res = (Xrg_rows[t] if t > 0 else 0)
        res = min(res, X); tot += (X-res)*u + res*c + out*o; prev = X
    return tot

def episodes_f1(bench, which="main"):
    recs = []
    for d, lst in rerun_files(bench).items():
        for rn, f in lst:
            if which == "main" and rn != "main": continue
            eps = {}
            for e in json.load(open(f))["episodes"]: eps.setdefault(e["task_id"], e)
            for t, we in {e.task: e for e in W.load(f)}.items():
                if we.R < 2 or we.N < 1: continue
                F = fixed_part(we)
                if we.X - we.R*F <= 0 or F <= 0: continue
                e = eps[t]; seen = set(); reqs = []
                for r in e["requests"]:
                    if r["request_id"] not in seen: seen.add(r["request_id"]); reqs.append(r)
                if len(reqs) != we.R: continue
                hf = np.concatenate([[0.0], np.minimum(we.Xrg[1:], we.Xrg[:-1]).sum(1)])   # history + fixed carried over per request
                recs.append(dict(skill=d, run=rn, task=t, R=len(reqs), inp=sum(r["input_tokens"] for r in reqs), out=sum(r["output_tokens"] for r in reqs),
                                 cache=sum((r.get("captured_usage") or {}).get("cache_tokens") or 0 for r in reqs),
                                 cm=eprice(reqs, mode="meas"), cp=eprice(reqs, mode="prev"), cx=eprice(reqs, list(hf), mode="xrg"), c0=eprice(reqs, mode="meas", P=(1, 1, 4)), succ=e.get("success")))
    return recs

def slope(recs, key):
    rs = [r for r in recs if r[key] > 0]
    x = np.log([r["R"] for r in rs]); y = np.log([r[key] for r in rs]); g = collections.defaultdict(list)
    for i, r in enumerate(rs): g[r["task"]].append(i)
    g = {k: v for k, v in g.items() if len(v) > 1}; idx = [i for v in g.values() for i in v]
    xs, ys = x.copy(), y.copy()
    for v in g.values(): xs[v] -= x[v].mean(); ys[v] -= y[v].mean()
    xs, ys = xs[idx], ys[idx]; return round(float((xs*ys).sum()/(xs*xs).sum()), 3), len(xs), len(g)

def f1():
    out = {}
    for which in ("main", "all"):
        for b in BENCHES:
            rs = episodes_f1(b, which); res = {}
            for lab, sel in (("seed", lambda r: r["skill"] == "seed"), ("optimized", lambda r: r["skill"] != "seed"), ("all", lambda r: True)):
                sub = [r for r in rs if sel(r)]
                if not sub: continue
                i = sum(r["inp"] for r in sub); o = sum(r["out"] for r in sub); c = sum(r["cache"] for r in sub)
                cm = sum(r["cm"] for r in sub); cp = sum(r["cp"] for r in sub); cx = sum(r["cx"] for r in sub)
                res[lab] = dict(n=len(sub), out_share_tokens=round(o/(i+o), 4), out_tokens_mean=round(o/len(sub)), in_tokens_mean=round(i/len(sub)), cache_frac_of_input=round(c/i, 4),
                                out_share_of_price_meas=round(4*o/cm, 4), out_share_of_price_prev=round(4*o/cp, 4), out_share_of_price_xrg=round(4*o/cx, 4),
                                price_mean_uncached_input_units=round(cm/len(sub)))
            res["slope_log_out"] = slope(rs, "out"); res["slope_log_in"] = slope(rs, "inp")
            res["slope_log_price_meas_cache"] = slope(rs, "cm"); res["slope_log_price_prev"] = slope(rs, "cp"); res["slope_log_price_xrg"] = slope(rs, "cx"); res["slope_log_price_nocache_out4x"] = slope(rs, "c0")
            out[f"noBACH|{which}|{b}"] = res
            print(f"noBACH|{which}|{b}", json.dumps(res), flush=True)
    return out

# ---- f4: compaction scan --------------------------------------------------------------------------------------------------------------------------------

def f4():
    tot = dict(files=0, eps=0, maxin=0, maxfile=None); by = {}
    for f in DS_ROOT.glob("*/*/*/results/evaluations/*/*/evaluation.json"):
        _, bench, _, _, _, split = f.relative_to(DS_ROOT).parts[:6]
        d = json.load(open(f)); tot["files"] += 1
        b = by.setdefault((bench, split), dict(eps=0, eps_none=0, reqs=0, drops=0, eps_drop=0, maxin=0, eps_ge190k=0))
        for e in d["episodes"]:
            seen = set(); rs = []
            for r in e.get("requests", []):
                if r["request_id"] in seen: continue
                seen.add(r["request_id"]); rs.append(r)
            if not rs: b["eps_none"] += 1; continue
            b["eps"] += 1; b["reqs"] += len(rs); dropped = False; mx = 0
            for i, r in enumerate(rs):
                x = r.get("input_tokens") or 0; mx = max(mx, x)
                if i > 0 and x < (rs[i-1].get("input_tokens") or 0) - 200:   # input shrank vs the previous request
                    b["drops"] += 1; dropped = True
            b["eps_drop"] += dropped; b["eps_ge190k"] += mx >= 190_000; b["maxin"] = max(b["maxin"], mx)
            if mx > tot["maxin"]: tot["maxin"] = mx; tot["maxfile"] = str(f)
    tot["eps"] = sum(v["eps"] for v in by.values())
    print(json.dumps(tot, indent=1))
    for k, v in sorted(by.items()): print(k, v)
    return {f"{k[0]}|{k[1]}": v for k, v in by.items()} | {"_total": tot}

# ---- f6, f6b: filter counts -----------------------------------------------------------------------------------------------------------------------------

def stats(f):
    d = json.load(open(f)); panel = [t["task_id"] for t in d["panel"]]
    eps = {}
    for e in d["episodes"]: eps.setdefault(e["task_id"], e)
    Ws = {e.task: e for e in W.load(f)}
    r = dict(file=str(f.relative_to(H)), panel=len(panel), episodes=len(eps), loaded=len(Ws), R_ge2=0, N_ge1=0, both=0, pos=0)
    ok = set()
    for t, we in Ws.items():
        a = we.R >= 2; b = we.N >= 1; r["R_ge2"] += a; r["N_ge1"] += b
        if a and b:
            r["both"] += 1; F = fixed_part(we)
            if we.X - we.R*F > 0 and F > 0: r["pos"] += 1; ok.add(t)
    return r, ok

def f6():
    out = {}
    for b in BENCHES:
        sf = eval_file(current_run("native-seed", MODEL, b), "dev", vid(SEED_TEXT)); rs, oks = stats(sf); out[f"{b}|seed"] = rs
        print(b, "seed", rs)
        for disp, meth in (("GEPA", "gepa"), ("SkillOpt", "skillopt"), ("SkillAdam", "skilladam")):
            run = current_run(meth, MODEL, b); t = final_skill(meth, run)
            if t == SEED_TEXT: print("  ", disp, "returned seed"); continue
            f = eval_file(run, "dev", vid(t))
            if not f: print("  ", disp, "no dev eval of final skill"); continue
            r, ok = stats(f); r["paired_with_seed"] = len(ok & oks); out[f"{b}|{disp}"] = r; print("  ", disp, r)
    return out

def f6b():
    out = {}
    for b in BENCHES:
        rows = []
        for d, l in rerun_files(b).items():
            for rn, f in l:
                dd = json.load(open(f)); eps = {e["task_id"] for e in dd["episodes"]}
                Ws = {e.task: e for e in W.load(f)}; keep = 0; eR = []
                for t, we in Ws.items():
                    if we.R < 2 or we.N < 1: continue
                    F = fixed_part(we)
                    if we.X - we.R*F <= 0 or F <= 0: continue
                    keep += 1; eR.append(1+(1-we.R*F/we.X)*we.R/(we.R-1))
                rows.append((d, rn, len(dd["panel"]), len(eps), len(Ws), keep, (round(min(eR), 3), round(max(eR), 3)) if eR else None))
        out[b] = rows
        print(b)
        for r in rows: print("  ", r)
    return out

# ---- facts-v9.md ----------------------------------------------------------------------------------------------------------------------------------------

def render(f1):
    sec=lambda k:f1[k]
    L=[];A=L.append
    A("# facts-v9 (DeepSeek-V4.1-Flash only; no SE/CI)\n")
    A("Producer: `bipa/src/analysis/appendix/facts.py` (output tokens and price, compaction scan, filter counts; this file is its `render`), `optcost.py` (optimizer calls, `llm.json`), `reruns.py` (reruns). Outputs in `assets/cache/analysis/facts/`. Run from `bipa/` with `bach-harness/.venv/bin/python`.\n")
    A("**Note for writers:** all numbers below are WITHOUT BACH (`noBACH`): BiPA reports the seed and the three baseline optimizers only.\n")
    A("## 1. Output tokens and price-weighted cost (#11)\n")
    A("**Assumptions (one line):** price of a request = 1.0 x uncached input + 0.1 x re-sent (cached) input + 4.0 x output tokens, in units of the uncached input price; re-sent = measured cache-read tokens (`captured_usage.cache_tokens`, primary), cross-checked by the approximation X_{t-1} (all of request t-1's input) and by the exact component carry-over sum_g min(Xrg[t,g], Xrg[t-1,g]) (= X_t minus content new at t); request 1 re-sends nothing. Sample = the paper's test episodes of the seed and of the optimized final skills (main runs; filters R>=2, >=1 call, positive history; no episode dropped), pooled sums. Output tokens as the provider reports them.\n")
    A("Optimized skills in the main sample (noBACH): GEPA, SkillOpt and SkillAdam on all three benchmarks since the https reruns (3 x 75 tau3; 3 x 120 AutoB; 3 x 18 TB2). Source `f1.json`, keys `noBACH|main|<bench>`.\n")
    A("| Benchmark | sample | n | output share of input+output tokens | output share of price (measured cache) | same, approx X_{t-1} | cache-read share of input tokens |\n|---|---|---|---|---|---|---|")
    for b,n in (("tau3-bench","tau3"),("automationbench","AutoB"),("terminal-bench-2-1","TB2")):
        d=sec(f"noBACH|main|{b}")
        for g in ("seed","optimized","all"):
            x=d[g]; A(f"| {n} | {g} | {x['n']} | {100*x['out_share_tokens']:.2f}% | {100*x['out_share_of_price_meas']:.1f}% | {100*x['out_share_of_price_prev']:.1f}% | {100*x['cache_frac_of_input']:.1f}% |")
    A("\nMean per seed episode: tau3 input 721,321 / output 10,195 tokens; AutoB 1,021,629 / 14,682; TB2 1,675,770 / 30,393.\n")
    A("**Within-task slopes of log y on log requests** (task fixed effects; main sample without BACH; (stale: pre-rerun) the episode counts n = 300 / 360 / 36 and the with-BACH note (375 episodes, slope 1.26) refer to the first runs — the https rerun sample is larger, see `f1.json`):\n")
    A("| y | tau3 | AutoB | TB2 |\n|---|---|---|---|")
    for lab,k in [("input tokens","slope_log_in"),("output tokens","slope_log_out"),("price, measured cache 0.1x, output 4x","slope_log_price_meas_cache"),("price, approx X_{t-1} re-sent","slope_log_price_prev"),("price, exact carry-over","slope_log_price_xrg"),("price, NO cache discount, output 4x","slope_log_price_nocache_out4x")]:
        A(f"| {lab} | "+" | ".join(f"{sec(f'noBACH|main|{b}')[k][0]:.2f}" for b in ("tau3-bench","automationbench","terminal-bench-2-1"))+" |")
    A("\n(stale: pre-rerun) All runs (seed and rerun episodes included; n = 600 / 840 / 107): input 1.28 / 1.51 / 1.64; output 0.99 / 0.84 / 0.62; price (measured cache) 1.12 / 1.14 / 0.91 (`noBACH|all|...`).\n")
    A("(stale: pre-rerun, the two slope tables above are recomputed on the https reruns but this reading was written on the first runs) Reading: output is 1.3-2.2% of tokens but 25-45% of the price-weighted cost; output grows slower than turns (slope 0.59-0.88 main), so it dilutes the elasticity. With the 0.1x cache price the price slope is 1.05 / 1.12 / 0.92 (about 1), against 1.25 / 1.49 / 1.79 for input tokens: the history term is weighed down by the discount. The approximations give a lower tau3 price slope (0.84) than measured cache (1.05); quote the measured column.\n")
    A("## 2. Optimization cost (#10, #8, #49)\n")
    A("Run per cell = `common.current_run(method,'deepseek-flash',bench)`: `native-20261008-https` (GEPA, SkillOpt, SkillAdam), under `assets/artifacts/execution/deepseek-flash/<method>/<bench>/<run>/`. Sources: `summary.json` (`episodes`, `arm_episodes`, `llm`), `llm/call-*.json` (provider usage; `optcost/llm.json`), `.scratch/v6/cost/cost.json` (train/dev consumed and fresh episodes, agent input/output sums). (stale: pre-rerun) The table below and the rest of this section were computed on the first runs (`native-20261004-upstream-batch` / `native-20261005`), not the https reruns; SkillAdam TB2 was computed from its evaluation files: 249 episodes in files, 450.0M input, 7.8M output tokens (summary: 255 consumed, 221 executed; its test: seed 72.2 vs final 66.7).\n")
    A("Columns: episodes consumed (executed fresh; the rest served from the shared episode cache, counted at original cost) for training / selection (dev split V); test = seed and final skill each once (n tasks each); `arm_episodes` = distinct episodes of the run; agent input / output tokens summed over training+selection episodes, millions; optimizer calls = files in `llm/` / `summary.json` count (differ where a run was resumed; the files are what cost.json used); optimizer prompt / completion tokens, millions (files); largest single analyst prompt (provider-counted tokens); calls whose prompt was cut to the 512k limit.\n")
    A("| Method | Bench | train (fresh) | dev=V (fresh) | test | arm_episodes | agent in / out (M) | opt. calls files/summary | opt. prompt / compl. (M) | largest prompt | truncated calls |\n|---|---|---|---|---|---|---|---|---|---|---|")
    A("""| GEPA | tau3 | 61 (59) | 153 (124) | 75 x2 | train 60, dev 150 | 163 / 1.8 | 7 / 7 | 0.67 / 0.08 | 207,473 | 0 |
| GEPA | AutoB | 65 (51) | 30 (0) | 120 x2 | train 63, dev 30 | 96 / 1.2 | 13 / 7 | 1.99 / 0.15 | 212,764 | 0 |
| GEPA | TB2 | 64 (61) | 48 (39) | 18 x2 | train 54, dev 45 | 157 / 3.2 | 11 / 2 | 1.21 / 0.08 | 265,414 | 0 |
| SkillOpt | tau3 | 167 (167) | 188 (187) | 75 x2 | train 160, dev 150, test 225 | 253 / 2.8 | 34 / 34 | 4.66 / 0.23 | 299,739 | 5 |
| SkillOpt | AutoB | 163 (143) | 184 (154) | 120 x2 | train 160, dev 150, test 360 | 391 / 4.7 | 19 / 15 | 3.38 / 0.12 | 359,799 | 26 |
| SkillOpt | TB2 | 161 (100) | 63 (54) | 18 x2 | train 62, dev 27, test 54 | 354 / 6.6 | 36 / 10 | 4.19 / 0.18 | 392,688 | 0 |
| SkillAdam | tau3 | 273 (253) | none | 75 x2 | train 270 | 219 / 2.2 | 41 / 41 | 4.14 / 0.57 | 165,500 | 0 |
| SkillAdam | AutoB | 224 (174) | none | 120 x2 | train 200 | 223 / 2.8 | 21 / 19 | 2.39 / 0.31 | 216,722 | 0 |
| SkillAdam | TB2 | 255 (221) | none | 18 x2 | train 223 | 450 / 7.8 | 28 / 4 | 2.96 / 0.36 | 205,257 | 0 |
""")
    A("Pooled over tau3+AutoB (old Table 3, `.scratch/v6/cost/cost_table.tex`; drop its BACH and Luna rows): GEPA 126 train (110 fresh) / 183 dev (124), agent 259M in / 3.0M out, optimizer 20 calls / 2.9M tokens; SkillOpt 330 (310) / 372 (341), 644M / 7.6M, 53 calls / 8.4M; SkillAdam 497 (427) / none, 442M / 5.0M, 62 calls / 7.4M.\n")
    A("Largest analyst prompt: 392,688 provider-counted tokens (SkillOpt TB2); per-run medians 60k-148k. The optimizer client fits every prompt to 512,000 estimated tokens by dropping its earliest content and keeping the first 16,000 (`bipa/src/utils/llm.py`: `MAX_PROMPT_TOKENS, KEEP_HEAD = 512_000, 16_000`). It triggered only in SkillOpt tau3 (5 calls, 2.33M dropped tokens by the client's estimate) and SkillOpt AutoB (26 calls, 3.57M dropped). Optimizer-LLM USD (assumed list prices, `.scratch/v6/cost/prices.md`, not measured): GEPA 0.22 / 0.48 / 0.25 (tau3 / AutoB / TB2), SkillOpt 1.39 / 0.93 / 1.23, SkillAdam 1.11 / 0.62 (tau3 / AutoB); quote only as assumed.\n")
    A("**SkillAdam validation step:** none. It has no selection split and never reads V; its acceptance gate (upstream default: primary gain > 0, strict) scores the candidate on the SAME training batch it was proposed from (`summary.json` deviations: 'same-batch validation (validation ids = training ids)'). True training episodes: 160 batch episodes (tau3 T=16 x b=10; AutoB 15 batches of 10; TB2 T=18 from three reshuffled passes over its 62 tasks) plus the candidate rollouts on those batches: 270 / 200 / 223 distinct episodes (`arm_episodes`), 273 / 224 / 255 consumed incl. repeats (`episodes`). Accepted iterations: tau3 5 of 16, AutoB 1 of 16 (12 patch failures), TB2 5 of 18 (7 patch failures). So SkillAdam consumed 1.3-1.7x the nominal 160.\n")
    A("SkillOpt note: upstream's loop also evaluates its baseline, best and final skills on the test split (`arm_episodes.test` = 225 / 360 / 54; `extra.baseline_test_hard`, `test_hard`, `final_test_hard`); these are logged, not used for selection, and the paper's test numbers use the same cached episodes.\n")
    A("## 3. Benchmarks (#6)\n")
    A("""- **AutomationBench** (Shepard and Salimans, 2026), white paper arXiv:2604.18934 (title 'AutomationBench', authors Daniel Shepard and Robin Salimans, submitted 2026-04-21; abstract line: 'a benchmark for evaluating AI agents on cross-application workflow orchestration via REST APIs'; fetched from arxiv.org/abs/2604.18934). Code https://github.com/zapier/AutomationBench (Zapier, Inc.), pinned commit 66cc02f0cadb4749e865392c46b184ccf81b86ea; README at `harness-bench/.runtime/cache-data/benchmarks/automationbench/source-66cc02f0.../README.md`. Not in-house: 600 scored public tasks in six domains of 100 each (sales, marketing, operations, support, finance, HR); 200 further 'simple' tasks are unscored and excluded; the official leaderboard uses a held-out private set, we use the public one. Each task starts a simulated business environment over 47 simulated SaaS tools (CRM, email, sheets, calendar, ...) from trigger data, with no user interaction; the agent finds the simulated APIs through a shell CLI (toolset `api`). Scoring: programmatic assertions on the final world state (3 to 64 per task, median 11.5, mean 16.0, over all 600 `tests/grading_config.json`), e.g. a sheet row has a given status, an email to a named person contains an amount. `partial_credit` = fraction of assertions met; `task_completed_correctly` = 1 iff every assertion passes (the official pass rate); our success = `task_completed_correctly == 1` (`harness-bench/src/harness/harbor.py` near line 1406). Harbor adapter: harbor-framework/harbor PR #2236 (commit c1d9675a). Per task 2 CPU, 4 GiB, 30-minute agent budget. Not in references.bib; suggested entry: `@misc{shepard2026automationbench, title={AutomationBench}, author={Shepard, Daniel and Salimans, Robin}, year={2026}, eprint={2604.18934}, archivePrefix={arXiv}}`.
- **Terminal-Bench 2.1**: Terminal-Bench (Merrill et al., 2026, 'Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces', arXiv:2601.11868; bibtex key `merrill2026terminalbenchbenchmarkingagentshard` in `harness-bench/assets/benchmarks/terminal-bench-2-1/README.md`). 2.1 is 'a more verified iteration of Terminal-Bench 2.0': 26 tasks modified (bugs, timeouts or resources, robustness to reward hacking; many from Z.ai's 2.0-verified changes). Repo https://github.com/harbor-framework/terminal-bench-2-1, commit 7131e4375048a0e408a8fb404b5f499d726b695b. 89 tasks (split config; the folder lists 91 entries). Each task runs in its own container; a test script gives reward 1 = success.
- **tau3-bench** (text): `harness-bench/assets/benchmarks/tau3-bench/README.md`: Harbor adapter over upstream sierra-research/tau2-bench v1.0.1 (commit fc0055dc4e0a316c3f83133267fbd6faaa770992, includes banking grading fixes), text-only (no voice), official tau2 evaluator behind an MCP sidecar; 375 tasks = airline 50, retail 114, telecom 114, banking-knowledge 97 (counted from task folder names); banking retrieval by BM25; the simulated user and natural-language assertion judge are model calls. Success = reward 1. references.bib has NO tau entry; the README gives only the repo URL. From memory, unverified: tau-bench, Yao et al. 2024, arXiv:2406.12045; tau2-bench, Barres et al. 2025, arXiv:2506.07982. I found no tau3 paper in the files: cite the repo (v1.0.1) plus those two after checking them.
""")
    A("## 4. Compaction and context (#116)\n")
    A("""- Configuration: the OpenCode (v1.18.23) config assembled in `harness-bench/src/harness/harbor.py` (about lines 1651-1717) sets only `small_model`, the provider, title generation off, the build agent's reasoning-effort hint and permission denials for task, webfetch and websearch. It sets no compaction or pruning option and, for this model, no context or output limit (`configs/models.yaml` gives `opencode_context_tokens` only to glm-5.3-flash). So OpenCode's own defaults apply. I could not inspect OpenCode's source, so do NOT claim its default; the safe sentence is 'we do not enable or tune compaction'.
- Empirical scan of all 23,400 stored DeepSeek episodes (971 evaluation files: all methods, splits, reruns, OOD; `facts.py` f4): in every TEST episode (AutoB 4,799, tau3 3,301, TB2 542 incl. reruns and other-method skills) the input tokens of consecutive requests never fall by more than 200 tokens, 0 events (history compaction or pruning would show as a drop). Drops occur only in TRAINING episodes (AutoB 84 episodes / 349 events, TB2 7 / 24), where tiny requests of tens to hundreds of tokens sit between full ones (an auxiliary request, not a history rewrite); none in dev or test. The string 'compaction' appears in no DeepSeek test evaluation file (grep; raw attempt logs are no longer on disk, so there is no event-level record). Conclusion: no DeepSeek test episode shows a compaction.
- Max input tokens per request: test tau3 386,592; AutoB 199,943; TB2 192,456; dev 270,245 / 187,434 / 213,676; train 417,053 / 190,884 / 353,836 (global max 417,053, tau3 train). Test episodes with a request of at least 190k tokens: tau3 118, AutoB 3, TB2 1.
""")
    A("## 5. GEPA tau3 skill sentence (#30)\n")
    A("(stale: pre-rerun) Verbatim, `assets/artifacts/execution/deepseek-flash/gepa/tau3-bench/native-20261004-upstream-batch/final/SKILL.md` line 49, section '## One action at a time' (the FIRST GEPA tau3 run; the https rerun returns a different skill): **\"At most one tool call per step; never combine a user message with a tool call.\"** In the paper's appendix source it is the added line at `appendix/e-returned-skills.tex:86`. The seed skill has no such rule, so it is new in GEPA's skill. The next bullets: 'Tools that can only be called once (e.g., modify/exchange an order) must include ALL items/details -- collect every item before calling.' and 'Guide the user through one step at a time and wait for the result before the next step.' Supporting ratios (cost.json, GEPA tau3 vs seed on the same tasks): requests x1.079, tool calls x0.979, input tokens x1.086, output tokens x1.072; more model requests for fewer tool calls, as a one-call-per-step rule produces.\n")
    A("## 6. Sample sizes and composition of the test episodes (#43, #67)\n")
    A("""**On V (selection split; `f6.json`)**, filters R>=2, >=1 executed call, positive history part (X - R F > 0, F > 0): seed on V passes on 30 of 30 tasks (tau3), 30 of 30 (AutoB), 9 of 9 (TB2). Final skills on V (https reruns): GEPA 30/30, 30/30, 9/9; SkillOpt 30/30, 30/30, 9/9; SkillAdam's final skill is evaluated on 10 of the 30 V tasks (tau3, AutoB) and 9/9 (TB2) — all paired with the seed. No optimizer returned the seed in the https reruns. The filters remove no task on V.
**On test**: the panels are complete in every run (75 per skill tau3, 120 AutoB, 18 TB2). Episodes lost to the episode-graph filters: TB2 seed rep-1 (17 of 18 loaded), TB2 GEPA rep-2 (17 loaded, 16 kept), TB2 SkillOpt main (17 of 18 loaded); none in tau3 or AutoB.
**Composition** (`check.json`, `facts.py` f6b; https reruns). (stale: pre-rerun) v8's 1,622 = tau3 675 + AutoB 840 + TB2 107 with the old BACH loop; without BACH it was 1,547. **Post-rerun, without BACH (use this): 2,556 episodes = tau3 900 (4 skills x 3 runs x 75) + AutoB 1,440 (4 x 3 x 120) + TB2 216 (4 x 3 x 18); 852 distinct skill-task pairs (tau3 4 x 75 = 300, AutoB 480, TB2 72); 1,704 episodes are reruns.** Per-episode e_R = 1 + s_H R/(R-1), without BACH: tau3 1.005-1.941, AutoB 1.304-1.924, TB2 1.055-1.919 (overall 1.005-1.941); none above 2. Slopes of log input tokens on log requests (task FE): main runs tau3 1.35 on 300 episodes / 75 tasks, AutoB 1.56 (480 / 120), TB2 1.58 (71 / 18); seed-only 3 runs: tau3 1.25 (225 / 75), AutoB 1.44 (360 / 120), TB2 1.66 (53 / 18); task x skill FE over all reruns: 1.29 / 1.48 / 1.57. Predicted 1 + s_H (no BACH): 1.27 / 1.69 / 1.55.
Rerun spreads (success %, per run; https reruns): seed tau3 78.7 / 76.0 / 74.7 (pooled 76.4), AutoB 40.8 / 49.2 / 45.0 (45.0), TB2 72.2 / 77.8 / 77.8 (75.9) — unchanged, the seed runs are the same; GEPA tau3 77.3 / 76.0 / 73.3 (75.6), AutoB 41.7 / 43.3 / 47.5 (44.2), TB2 77.8 / 77.8 / 66.7 (74.1); SkillOpt tau3 74.7 / 80.0 / 82.7 (79.1), AutoB 43.3 / 40.8 / 40.8 (41.7), TB2 66.7 / 66.7 / 61.1 (64.8); SkillAdam tau3 74.7 / 77.3 / 73.3 (75.1), AutoB 44.2 / 44.2 / 45.8 (44.7), TB2 66.7 / 66.7 / 72.2 (68.5) (`check.json` success_runs).
""")
    A("## 7. A.1 'two differences' and budgets by method (#8)\n")
    A("""Current A.1 (`appendix/c-setup.tex`, 'Budgets and episodes') names one difference (GEPA early stop) and says 'within the same budget'. Actual budgets (episodes consumed, training / selection on V; tau3, AutoB, TB2):
1. **GEPA**: minibatch 3 per iteration (iteration t reflects on shared batch B_t), at most T=20 iterations instead of upstream's 600 metric calls, our adapter stops after 3 iterations without improvement past the tenth: training 61 / 65 / 64, selection 153 / 30 / 48 (distinct: train 60 / 63 / 54, dev 150 / 30 / 45). Candidates scored on V (30 tasks, TB2 9), not upstream's 50.
2. **SkillOpt**: upstream batch b=40, minibatch 8, T=4 steps over the 160-episode sample (TB2: reshuffled passes over 62), selection on V: training 167 / 163 / 161, selection 188 / 184 / 63.
3. **SkillAdam**: upstream b=10, T=16 (AutoB 15 effective, TB2 18), no early stop, acceptance gate on the same training batch, no selection split: 270 / 200 / 223 distinct training episodes (273 / 224 / 255 consumed), 0 selection episodes.
Differences common to all three (`summary.json` deviations): (a) rollouts run on the shared executor over the same stratified batches B_1..B_T (one pass over the 160-episode sample; TB2 three reshuffled passes over its 62 tasks) instead of each upstream's random sampler; (b) binary success only (no partial credit; unknown outcomes abort), with an episode cache so a repeated skill-task pair is not rerun. Suggested wording: replace 'within the same budget' by 'with a nominal budget of 160 training episodes (consumed: GEPA 61-65, SkillOpt 161-167, SkillAdam 224-273; SkillAdam has no selection split)', and list the two differences as the GEPA early stop and the shared batches and executor.
""")
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(out_dir("facts")))
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)
    r1 = f1()
    for name, res in (("f1", r1), ("f4", f4()), ("f6", f6()), ("f6b", f6b())): (out/f"{name}.json").write_text(json.dumps(res, indent=1))
    (out/"facts-v9.md").write_text(render(r1))

if __name__ == "__main__": main()
