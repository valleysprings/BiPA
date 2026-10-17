"""Freeze the submission protocol: assets/cache/analysis/submission-20261008/{protocol.json, PROTOCOL.md}, read by every analysis through env BACH_SUBMISSION_MANIFEST (common.manifest / manifest_reps).

Per (bench, row) the manifest pins one source run (No skill: the run common.latest_run returns today; Seed: native-20261003 + rep-1 + rep-2; GEPA / SkillOpt / SkillAdam: BASELINE_TAG; BACH: BACH_TAG), the sha256 of its summary / history / final skill,
its three test executions {rep: evaluation.json, panel hash, request-id signature, missing/invalid counts}, the dev/train panels, budget, model and cache identity. A cell is `complete` only with summary.json and three distinct
test executions (repetitions 0-2), else `pending` with the reason; nothing falls back to another run. A skill equal to the seed text reuses the seed row's executions (`returned_seed`, `executions_from`).
Execution identity = sha256 of the sorted request_ids of an evaluation's episodes: cached copies of one execution share it (logged in `dedup_log`), independent reruns with identical success vectors differ and stay distinct.
The old BACH run (bach-v8) is kept only under `bach_v8_reference` (not for the table). Read-only on artifacts; run from bipa: /mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation/bach-harness/.venv/bin/python -m src.analysis.submission_protocol
"""
import argparse, datetime, json
from src.analysis import common
from src.analysis.common import *

MODEL = "deepseek-flash"
BASELINE_TAG, BACH_TAG, REF_TAG = "native-20261008-https", "bach-v9", "bach-v8"
SUB = H/"assets/cache/analysis/submission-20261008"
HISTORY = {"skillopt": "skillopt/history.json", "gepa": "gepa/run_log.json", "skilladam": "skilladam/checkpoint.json", "ours": "result.json"}
def fsha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None
def hlist(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def execution(f):
    """One evaluation.json as an execution record: panel hash, size, request-id signature (None = no identity), missing / unknown / invalid episode counts, outcome reasons."""
    d = json.loads(Path(f).read_text()); panel = [(t["task_id"], t.get("repetition", 0), t.get("weight")) for t in d["panel"]]
    eps = {}
    for e in d["episodes"]: eps.setdefault((e["task_id"], e.get("repetition", 0)), e)
    ids = {r["request_id"] for e in eps.values() for r in e.get("requests", [])}; why = {}
    for e in eps.values(): why[str(e.get("outcome_reason"))] = why.get(str(e.get("outcome_reason")), 0) + 1
    return {"file": str(f), "sha256": fsha(f), "reps": sorted({p[1] for p in panel}), "panel_hash": hlist(sorted(panel, key=str)), "panel_dir": Path(f).parent.name.split("-")[-1], "n_tasks": len(panel),
            "signature": sha("\n".join(sorted(ids))) if ids else None, "n_request_ids": len(ids), "no_request_episodes": sum(not e.get("requests") for e in eps.values()),
            "missing": sum((t[0], t[1]) not in eps for t in panel), "unknown": sum(e.get("success") is None for e in eps.values()), "invalid": sum(e.get("outcome_valid") is not True for e in eps.values()),
            "successes": sum(e.get("success") is True for e in eps.values()), "outcome_reasons": why}

def scan(dirs, version):
    """[(dir, evaluation.json)] of the test split for a skill version over run dirs, in dir order, newest first within a dir."""
    out = []
    for run in dirs:
        fs = [f for f in (Path(run)/"results/evaluations/test").glob("*/evaluation.json") if f.parent.name.startswith(version[:12])]
        out += [(run, f) for f in sorted(fs, key=lambda f: -f.stat().st_mtime)]
    return out

def pin(found, bench, row, log):
    """Pin one execution per repetition (first dir, newest file wins, as table1.rep_files); other files go to the log as cached copies (same signature) or independent reruns (different signature)."""
    ex, got = {}, {}
    for run, f in found:
        e = execution(f); e["source_run"] = Path(run).name
        for rep in e["reps"]:
            if rep not in got: got[rep] = e; ex[str(rep)] = e
            else:
                k = "no_identity" if not e["signature"] else "cached_copy" if e["signature"] == got[rep]["signature"] else "other_panel" if e["panel_hash"] != got[rep]["panel_hash"] else "independent_rerun"
                log.append({"bench": bench, "row": row, "rep": rep, "kind": k, "pinned": got[rep]["file"], "other": e["file"], "signature": e["signature"]})
    return ex

def build(bench, method, run, log, seed_text=SEED_TEXT):
    """The manifest cell of one (bench, row) from its pinned run dir (None = run absent)."""
    c = {"status": "pending", "run": str(run) if run else None}
    if not run or not Path(run).is_dir(): c["reason"] = "run dir absent: " + (f"{BACH_TAG} not started" if method == "ours" else "not started"); return c
    c["run_name"] = Path(run).name; summ = Path(run)/"summary.json"; proto = Path(run)/"native_protocol.json"; split = Path(run)/"split.json"
    s = json.loads(summ.read_text()) if summ.is_file() else {}
    p = json.loads(proto.read_text()) if proto.is_file() else {}; sp = json.loads(split.read_text()) if split.is_file() else {}
    c["summary_sha256"] = fsha(summ); c["protocol_sha256"] = fsha(proto) or fsha(split); c["completion_audit_sha256"] = fsha(Path(run)/"completion_audit.json")
    c["history"] = {HISTORY[method]: fsha(Path(run)/HISTORY[method])} if method in HISTORY else {}
    c["config_hash"] = p.get("config_hash") or s.get("config_hash")
    panels = p.get("panels") or {k: sp[k] for k in ("stream", "sel", "test") if k in sp}
    c["panels"] = {k: {"n": len(v), "sha256": hlist(v)} for k, v in panels.items()}
    c["budget"] = {k: s.get(k) for k in ("T", "panels", "episodes", "arm_episodes", "llm")}
    text = s.get("final_skill") if summ.is_file() else None
    if method == "native-noseed": text = None
    elif method == "native-seed": text = seed_text
    ver = vid(text); c["final"] = {"version_id": ver, "skill_sha256": sha(text) if text is not None else None, "file_sha256": fsha(Path(run)/"final/SKILL.md"), "returned_seed": text == seed_text and method != "native-seed"}
    reps_dir = EX/MODEL/"test-reps"/f"{bench}__{method}__{Path(run).name}"
    if method == "native-seed": dirs = [run, Path(run).parent/"rep-1", Path(run).parent/"rep-2"] + [d for d in sorted(Path(run).parent.iterdir()) if d.is_dir() and d != Path(run) and d.name not in ("rep-1", "rep-2")]
    else: dirs = [run, reps_dir]
    cache = None
    c["executions"] = pin(scan(dirs, ver), bench, method, log)
    first = next(iter(c["executions"].values()), None)
    if first:
        ev = json.loads(Path(first["file"]).read_text()).get("provenance", {})
        cache = (ev.get("evaluation_cache") or {}).get("root")
    c["model_config"] = {"model": p.get("model") or s.get("model"), "harness": p.get("harness") or s.get("harness"), "cache_root": p.get("cache_root") or sp.get("cache_root"), "evaluation_cache_root": cache,
                         "reasoning": "not recorded in run artifacts", "protocol": p.get("protocol") or s.get("protocol"), "feedback_mode": p.get("feedback_mode")}
    why = []
    if not summ.is_file(): why.append("no summary.json")
    if (Path(run)/"ABORTED.md").exists(): why.append("ABORTED.md")
    if c["final"]["returned_seed"]:
        c["executions_from"] = "native-seed"; c.pop("executions")
        c["status"] = "pending" if why else "complete"; c["reason"] = "; ".join(why) or None; return c
    sigs = {e["signature"] for e in c["executions"].values()}
    if sorted(c["executions"]) != ["0", "1", "2"]: why.append("test executions found for repetitions %s, need 0,1,2" % sorted(c["executions"]))
    elif None in sigs or len(sigs) < 3: why.append("test executions not three distinct request-id signatures")
    c["status"] = "pending" if why else "complete"; c["reason"] = "; ".join(why) or None
    return c

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--out", default=str(SUB))
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)
    log, cells, ref = [], {}, {}
    for bench in BENCHES:
        row = {}
        for m in METHODS:
            cell = EX/MODEL/m/bench
            run = latest_run(m, MODEL, bench) if m == "native-noseed" else cell/"native-20261003" if m == "native-seed" else cell/(BACH_TAG if m == "ours" else BASELINE_TAG)
            row[m] = build(bench, m, run, log)
        seed = row["native-seed"]
        for m, c in row.items():  # a returned seed row is complete only if the seed row is
            if c.get("executions_from") and seed["status"] != "complete": c["status"] = "pending"; c["reason"] = "seed row pending"
        cells[bench] = row
        ref[bench] = dict(build(bench, "ours", EX/MODEL/"ours"/bench/REF_TAG, log), not_for_table=True, note="old BACH (v8), not the formal v9; never feeds Table 1")
    man = {"schema": "bach-submission-protocol-v1", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", ""), "model": MODEL, "baseline_tag": BASELINE_TAG, "bach_tag": BACH_TAG,
           "cells": cells, "bach_v8_reference": ref, "dedup_log": log}
    (out/"protocol.json").write_text(json.dumps(man, indent=1))
    pend = [(b, m, c["reason"]) for b, r in cells.items() for m, c in r.items() if c["status"] != "complete"]
    (out/"PROTOCOL.md").write_text(f"""# 提交协议冻结（{man['created_utc']} UTC）

`protocol.json` 固定每个 (bench, row) 的来源 run、summary/history/final 的 sha256、test 的三次 execution（evaluation.json、面板哈希、request-id 签名、缺失/无效计数）、train/dev 面板哈希与预算、模型与 cache 身份。
分析脚本用环境变量 `BACH_SUBMISSION_MANIFEST=<本文件所在目录>/protocol.json` 读取同一快照（`common.current_run` / `common.manifest_reps`）；`analysis/submission_audit.py --manifest FILE` 只读复核，全部完整且一致才返回 0。

- 基线 run 固定为 `{BASELINE_TAG}`；BACH 固定为正式 `{BACH_TAG}`；缺失或未完成的 cell 写 `pending`，不回退到旧 run。旧 `{REF_TAG}` 仅在 `bach_v8_reference` 下，标记 not_for_table。
- 完整 = 有 summary.json 且 test 有重复 0/1/2 三个不同 execution。execution 身份 = 该次评测全部 episode 的 request_id 集合的 sha256；缓存复制（签名相同）记入 `dedup_log`，真实独立重跑即使 success 向量相同也保留。
- 返回 seed 的 skill 复用 seed 行的三次 execution（`returned_seed`、`executions_from`）。run 目录里未记录推理（reasoning）配置，已在 `model_config.reasoning` 注明。
- 当前 pending：{len(pend)} 个 cell（{sum(c['status']=='complete' for r in cells.values() for c in r.values())} 个 complete），原因见 protocol.json 的 `reason`。重新运行本脚本会生成新快照；冻结后不要覆盖。
""")
    print("wrote", out/"protocol.json")
    for b, r in cells.items():
        for m, c in r.items(): print("%-20s %-14s %-9s %s" % (b, m, c["status"], c.get("reason") or c.get("run_name")))
    print("dedup_log entries:", len(log))

if __name__ == "__main__": main()
