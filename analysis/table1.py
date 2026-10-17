"""Table 1 (tab:mech-main): test success Q = avg@3 and the ratios of cost to the Seed skill row, per method and benchmark (DeepSeek-V4.1-Flash and GPT-6 Luna).

Every row is evaluated on three test runs (repetitions 0-2; Q = mean of the per-run success, runs found are in `reps`). Slots whose episodes ran but never
produced a verdict in any of reps 0-2 are adjudicated as failures (listed in `adjudicated_failures`); they count 0 in Q and drop out of the cost ratios for lack of N/X. The Seed row is the seed's three runs
(native-seed run + rep-1, rep-2); a skill that equals the seed text reports them. All ratios are the geometric mean over tasks of (per-task mean over the row's runs) /
(per-task mean over the seed's runs) on one common task set (verified outcome, N and X > 0): rho (c = N + X/1000), rho_inv (tool calls N), rho_trj (input tokens X).
The paper's Table 1 prints Q, rho_trj (as rho) and rho_inv (as rho^calls); the composite rho is not printed.
Bold and underline (rank best / second) are over the five rows of a benchmark; the dagger marks rho > 1.10.
Outputs: assets/cache/analysis/table1/{table1.json (read by fig1, reruns, facts), table1.tex (body rows)} and the audit record paper/assets/data/mech/mech_table1.md (paperout.record).
"""
import argparse, json, math
from analysis.common import *
from src.experiment_config import repetition_dir

MODEL = "deepseek-flash"
MODEL_TEX = {"deepseek-flash": "DeepSeek-V4.1-Flash", "gpt-6-luna": "GPT-6 Luna"}
ROWS = (("No skill", "native-noseed"), ("Seed skill", "native-seed"), ("GEPA", "gepa"), ("SkillOpt", "skillopt"), ("SkillAdam", "skilladam"))
TEX_BENCHES = ("terminal-bench-2-1", "tau3-bench", "automationbench")

def test_rows(run, version):
    r = evaluation(run, "test", version)
    return r[0] if r else None

def rep_files(dirs, version):
    """{repetition: evaluation.json} of the test split for one skill over run dirs (first dir wins a repetition, the newest file within a dir; the repetition is read from the panel rows)."""
    out = {}
    for run in dirs:
        got = {}
        for f in (run/"results/evaluations/test").glob("*/evaluation.json"):
            if not f.parent.name.startswith(version[:12]): continue
            for rep in {t.get("repetition", 0) for t in json.loads(f.read_text())["panel"]}:
                if rep not in got or f.stat().st_mtime > got[rep].stat().st_mtime: got[rep] = f
        for rep, f in got.items(): out.setdefault(rep, f)
    return out

def rep_groups(dirs, version):
    """{repetition: [evaluation.json, ...]} — every file of a repetition (dir order, oldest first within a dir). Files beyond the first are
    unknown-only supplements from repeat.py --execute fills; merged_rows_of/unresolved_of merge them (a known verdict is never overridden)."""
    out = {}
    for run in dirs:
        got = {}
        for f in (run/"results/evaluations/test").glob("*/evaluation.json"):
            if not f.parent.name.startswith(version[:12]): continue
            for rep in {t.get("repetition", 0) for t in json.loads(f.read_text())["panel"]}:
                got.setdefault(rep, []).append(f)
        for rep, fs in got.items(): out.setdefault(rep, []).extend(sorted(fs, key=lambda f: f.stat().st_mtime))
    return out

def merged_episodes(files):
    """One repetition's evaluation files merged into {slot: episode}; repeat.py semantics: an original known verdict is never
    overridden, supplements fill unresolved or absent slots."""
    eps = {}
    for f in files:
        d = json.loads(f.read_text()); ptasks = {t["task_id"] for t in d["panel"]}
        for e in d["episodes"]:
            if e["task_id"] not in ptasks: continue
            k = (e["task_id"], e.get("repetition", 0)); old = eps.get(k)
            if old is not None and type(old.get("success")) is bool and old.get("outcome_valid", True) is True: continue
            eps[k] = e
    return eps

def rows_of(f):
    d = json.loads(f.read_text()); eps = {}
    for e in d["episodes"]: eps.setdefault((e["task_id"], e.get("repetition", 0)), e)
    return {(t["task_id"], t.get("repetition", 0)): task_row(eps.get((t["task_id"], t.get("repetition", 0)), {})) for t in d["panel"]}

def merged_rows_of(files):
    """Merged per-task rows of one repetition over all its evaluation files (union of panels, first file's panel order)."""
    eps = merged_episodes(files); rows = {}
    for f in files:
        for t in json.loads(f.read_text())["panel"]:
            k = (t["task_id"], t.get("repetition", 0))
            if k not in rows: rows[k] = task_row(eps.get(k, {}))
    return rows

def unresolved_of(files):
    """{slot: outcome_reason} for merged episodes (one repetition's files) that ran but produced no valid verdict."""
    return {k: (e.get("outcome_reason") or "unresolved") for k, e in merged_episodes(files).items()
            if not (type(e.get("success")) is bool and e.get("outcome_valid", True) is True)}

def geo(rows, ref):
    """geometric mean per task vs the Seed row; rows/ref: {task: {success,N,X}}."""
    L = {"c": [], "N": [], "X": []}
    for k, r in rows.items():
        b = ref.get(k)
        if b is None or r["success"] is None or b["success"] is None: continue
        if all(x and x > 0 for x in (r["N"], r["X"], b["N"], b["X"])):
            for f in L: L[f].append(math.log(r[f]/b[f]))
    n = len(L["c"])
    return n, {nm: gmean(np.exp(L[f])) for f, nm in (("c", "rho"), ("N", "rho_inv"), ("X", "rho_trj"))}

def as_lists(reps):
    """Normalize {rep: file} (manifest) vs {rep: [files]} (rep_groups) to {rep: [files]}."""
    return {k: ([v] if isinstance(v, Path) else v) for k, v in reps.items()} if reps else reps

def accumulate(allrows):
    """{(task, 0): row} with N, X, c averaged over the per-run rows (tasks with a verified outcome and N, X in a run)."""
    per = {}
    for rows in allrows:
        for (t, _), r in rows.items():
            if r["success"] is not None and r["N"] is not None and r["X"] is not None: per.setdefault(t, []).append(r)
    return {(t, 0): {"success": True, "N": sum(r["N"] for r in v)/len(v), "X": sum(r["X"] for r in v)/len(v), "c": sum(r["N"] + r["X"]/1000 for r in v)/len(v)} for t, v in per.items()}

def mean_rows(reps):
    """Per-run rows and their task means over the runs {rep: file} of a skill."""
    allrows = [rows_of(f) for _, f in sorted(reps.items())]
    return allrows, accumulate(allrows)

def merged_mean_rows(reps):
    """Per-run rows and their task means over the runs {rep: [files]} of a skill, supplements merged (rep_groups)."""
    allrows = [merged_rows_of(fs) for _, fs in sorted(reps.items())]
    return allrows, accumulate(allrows)

def compute(model=MODEL):
    res = {}
    for bench in BENCHES:
        seed_run = current_run("native-seed", model, bench); seed_dirs = [seed_run, repetition_dir(seed_run), seed_run.parent/"rep-1", seed_run.parent/"rep-2"]
        seed_reps = as_lists(manifest_reps(bench, "native-seed", model)) if manifest() else rep_groups(seed_dirs, vid(SEED_TEXT)); ref = merged_mean_rows(seed_reps)[1]
        cell = {}
        for name, m in ROWS:
            run = current_run(m, model, bench)
            if not run or not (run/"summary.json").is_file(): cell[name] = None; continue
            t = final_skill(m, run); seedret = t == SEED_TEXT; asseed = seedret or m == "native-seed"
            own_reps = as_lists(manifest_reps(bench, m, model)) if manifest() else rep_groups([run, repetition_dir(run)], vid(t))
            reps = {**own_reps, **seed_reps} if asseed else own_reps
            if not reps:
                cell[name] = None; continue
            allrows, mean = merged_mean_rows(reps)
            allevid = [unresolved_of(fs) for _, fs in sorted(reps.items())]
            kn = [[r for r in rows.values() if r["success"] is not None] for rows in allrows]
            known_tasks = [{t for (t, _), r in rows.items() if r["success"] is not None} for rows in allrows]
            d = dict(reps=len(reps), succ=sum(r["success"] is True for k in kn for r in k), known=sum(map(len, kn)), total=len(allrows[0]), returned_seed=seedret and m != "native-seed", run=run.name, run_dir=str(run),
                     test_eval=str(reps[min(reps)][0]), files={k: [str(f) for f in fs] for k, fs in sorted(reps.items())}, source="Seed row (native-seed runs, %s)" % seed_run.name if asseed else "own run %s" % run.name)
            d["Q"] = 100*sum(sum(r["success"] is True for r in k)/len(k) for k in kn)/len(kn) if all(kn) else None
            frozen_test = json.loads((run/"native_protocol.json").read_text())["panels"]["test"] if (run/"native_protocol.json").is_file() else json.loads((seed_run/"native_protocol.json").read_text())["panels"]["test"]
            expected = {row if isinstance(row, str) else row["task_id"] for row in frozen_test}
            # A task whose episodes ran but never produced a verdict in any of reps 0-2 is adjudicated
            # as a failure (scored 0 in Q, excluded from the cost ratios for lack of N/X, listed below).
            adjudicated = {}
            if set(reps) == {0, 1, 2}:
                for task in expected:
                    per = [None if task in kt else evid.get((task, rep), False)
                           for rep, evid, kt in zip(sorted(reps), allevid, known_tasks)]
                    if all(type(x) is str for x in per):
                        adjudicated[task] = per
            d["adjudicated_failures"] = adjudicated
            d["avg3_complete"] = (set(reps) == {0, 1, 2} and all(
                {task for task, _ in rows} == expected and all(t in kt or t in adjudicated for t in expected)
                for rows, kt in zip(allrows, known_tasks)))
            if d["avg3_complete"]:
                d["Q"] = 100*sum(sum(r["success"] is True for r in rows.values()) for rows in allrows)/(len(expected)*len(allrows))
            d["n"], rr = geo(mean, ref); d.update(rr)
            cell[name] = d
        qs = sorted({round(v["Q"], 6) for v in cell.values() if v and v["avg3_complete"]}, reverse=True)
        for v in cell.values():
            if v: v["rank"] = ("best" if round(v["Q"], 6) == qs[0] else "second" if len(qs) > 1 and round(v["Q"], 6) == qs[1] else "") if v["avg3_complete"] else ""
        res[f"{model}|{bench}"] = cell
    return res

def heat(v, dag=False):
    return "\\tbd{}" if v is None else f"\\heat{{{v:.2f}}}" + ("$^\\dagger$" if dag and round(v, 2) > 1.10 else "")

def tex(res, models=MODELS):
    L = ["% Body rows of Table 1: returned-seed skills report the Seed row's test episodes; all ratios vs the Seed row per task on one common task set (verified outcome both sides, N,X>0); bold/underline over five rows."]
    for model in models:
        L.append("\\multicolumn{13}{@{}l}{\\emph{%s}} \\\\" % MODEL_TEX[model])
        for name, _ in ROWS:
            if name == "GEPA": L.append("\\addlinespace[2pt]")
            cs = []
            for b in TEX_BENCHES:
                v = res[f"{model}|{b}"][name]
                if v is None or not v.get("avg3_complete"): cs += ["\\tbd{}"]*4; continue
                q = f"{v['Q']:.1f}"; q = f"\\textbf{{{q}}}" if v["rank"] == "best" else f"\\underline{{{q}}}" if v["rank"] == "second" else q
                cs += [q, heat(v["rho"], True), heat(v["rho_inv"]), heat(v["rho_trj"])]
            L.append(name + " & " + " & ".join(cs) + " \\\\")
    return "\n".join(L) + "\n"

def record(res, models=MODELS):
    """paper/assets/data/mech/mech_table1.md: Table 1 as the paper prints it, one block per model."""
    from analysis import paperout as PO
    names = [n for n, _ in ROWS]
    def mark(v, txt): return f"**{txt}**" if v["rank"] == "best" else f"<u>{txt}</u>" if v["rank"] == "second" else txt
    blocks, cells = [], []
    for model in models:
        ranked = {}
        for b in TEX_BENCHES:  # bold / underline over all five rows
            qs = sorted({round(res[f"{model}|{b}"][n]["Q"], 6) for n in names}, reverse=True)
            ranked[b] = {n: ("best" if round(res[f"{model}|{b}"][n]["Q"], 6) == qs[0] else "second" if len(qs) > 1 and round(res[f"{model}|{b}"][n]["Q"], 6) == qs[1] else "") for n in names}
        head = ["Method"] + [f"{lab} {c}" for lab in ("TB2", "tau3", "AutoBench") for c in ("Q", "rho", "rho^calls")]
        rows = []
        for n in names:
            r = [n]
            for b in TEX_BENCHES:
                v = dict(res[f"{model}|{b}"][n], rank=ranked[b][n]); r += [mark(v, f"{v['Q']:.1f}"), f"{v['rho_trj']:.2f}", f"{v['rho_inv']:.2f}"]
            rows.append(r)
        blocks.append(f"**{MODEL_TEX[model]}**\n\n" + PO.mdtable(head, rows))
        cells += [res[f"{model}|{b}"][n] for b in BENCHES for n, _ in ROWS if res[f"{model}|{b}"][n]]
    seedret = [f"{n} on {lab} ({MODEL_TEX[model]})" for model in models for b, lab in zip(BENCHES, ("tau3", "AutoBench", "TB2")) for n, _ in ROWS
               if res[f"{model}|{b}"][n] and res[f"{model}|{b}"][n]["returned_seed"]]
    sent = ("Skills equal to the seed text (" + "; ".join(seedret) + ") report the Seed row's three runs." if seedret
            else "No optimizer row equals the seed text; every row reports its own three runs.")
    txt = ("Table 1 (Q = avg@3 over three test runs per skill, ratios against the Seed skill row on a common task set). Bold / underline: best / second-best Q of a benchmark within a model.\n\n" + "\n\n".join(blocks)
           + "\n\nColumn mapping: paper Q = `Q`; paper rho (input tokens) = `rho_trj`; paper rho^calls (tool calls) = `rho_inv`; `rho` (cost c = N + X/1000) is not printed. " + sent)
    PO.record("mech_table1", "table1", result=txt, source=["bipa/analysis/table1.py"], commands=["/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation/bach-harness/.venv/bin/python -m analysis.table1  # from bipa/"],
              raw=[c["run_dir"] for c in cells] + [c["test_eval"] for c in cells],
              notes=["rho = geometric mean per task of (mean over the runs of the metric) / (the same mean over the seed's runs)."])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(out_dir("table1")))
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)
    res = {}
    for model in MODELS: res.update(compute(model))
    (out/"table1.json").write_text(json.dumps(res, indent=1))
    (out/"table1.tex").write_text(tex(res))
    if all(v and v["avg3_complete"] for cell in res.values() for v in cell.values()):
        record(res)
    else:
        print("Incomplete test repetitions: publication record not refreshed; see avg3_complete in table1.json.")
    for k, c in res.items():
        print(k)
        for n, v in c.items():
            if v: print(n, json.dumps({key: v[key] for key in ("Q", "reps", "avg3_complete", "known", "rho", "rho_inv", "rho_trj", "n")}))

if __name__ == "__main__": main()
