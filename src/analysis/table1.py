"""Table 1: test success Q (avg@3) and cost ratios to the Seed row per method and benchmark, both models.

Q is avg@3: per test run the share of tasks with a verified outcome that succeed, averaged over the three runs; DeepSeek counts
a task with no verdict in any run as a failure, GPT-6 Luna drops provider-failed requests and scores Q over verified outcomes.
rho (input tokens), rho^turns and rho^calls (executed tool calls) are geometric means over the common tasks of a task's mean
over the runs divided by the same mean over the Seed row's runs. A skill equal to the seed text reports the Seed row's runs.
Bold / underline = best / second Q of a benchmark."""
import argparse, json
from pathlib import Path

from src.analysis.common import (accumulate, geo, luna_row, mean_rows, merged_episodes, merged_mean_rows,  # noqa: F401
                                 merged_rows_of, rep_files, rep_groups, rows_of, unresolved_of)
from src.analysis import settings as S
from src.layout import repetition_dir
from src.utils.dataloader import BENCHES, EX, MODELS, SEED_TEXT, final_skill, out_dir, task_row, vid  # noqa: F401
from src.utils import logger

LUNA = "gpt-6-luna"
MODEL_TEX = {"deepseek-flash": "DeepSeek-V4.1-Flash", LUNA: "GPT-6 Luna"}
ROWS = (("No skill", "native-noseed"), ("Seed skill", "native-seed"), ("GEPA", "gepa"), ("SkillOpt", "skillopt"), ("SkillAdam", "skilladam"))
TEX_BENCHES = ("terminal-bench-2-1", "tau3-bench", "automationbench")
LABEL = {"terminal-bench-2-1": "TB2", "tau3-bench": "tau3", "automationbench": "AutoBench"}


def adjudicate(reps, allrows, expected):
    """Tasks with no verdict in any of the three runs: {task: [reason per run]}."""
    if set(reps) != {0, 1, 2}:
        return {}
    known = [{t for (t, _), r in rows.items() if r["success"] is not None} for rows in allrows]
    evid = [unresolved_of(fs) for _, fs in sorted(reps.items())]
    out = {}
    for task in expected:
        per = [None if task in kt else ev.get((task, rep), False) for rep, ev, kt in zip(sorted(reps), evid, known)]
        if all(type(x) is str for x in per):
            out[task] = per
    return out


def row_reader(model):
    """The per-episode reader of a model's test runs. GPT-6 Luna drops provider-failed requests and keeps a success only for a
    verified outcome (luna_row); DeepSeek reads every episode with task_row."""
    return luna_row if model == LUNA else task_row


def run_files(run, text):
    """{repetition: [evaluation.json]} of one skill's test runs: the run's own test evaluations and its test-reps directory."""
    return rep_groups([run, repetition_dir(run)], vid(text))


def seed_files(model, bench):
    """(the Seed row's run, {repetition: [evaluation.json]}): its run, its test-reps directory, rep-1 and rep-2."""
    seed_run = S.run(model, bench, "native-seed")
    dirs = [seed_run, repetition_dir(seed_run), seed_run.parent / "rep-1", seed_run.parent / "rep-2"]
    return seed_run, rep_groups(dirs, vid(SEED_TEXT))


def cell_files(model, bench, m, seed_reps):
    """(run, final skill, {repetition: [evaluation.json]}) of one row, as the pins give it. A skill equal to the seed text
    reports the Seed row's runs."""
    run = S.run(model, bench, m)
    text = final_skill(m, run)
    own = run_files(run, text)
    return run, text, ({**own, **seed_reps} if text == SEED_TEXT or m == "native-seed" else own)


def cell(model, bench, m, seed_run, seed_reps, ref):
    got = cell_files(model, bench, m, seed_reps)
    if not got or not got[2]:
        return None
    run, text, reps = got
    as_seed = text == SEED_TEXT or m == "native-seed"
    allrows, mean = merged_mean_rows(reps, row_reader(model))
    kn = [[r for r in rows.values() if r["success"] is not None] for rows in allrows]
    proto = run / "native_protocol.json"
    proto = proto if proto.is_file() else seed_run / "native_protocol.json"
    expected = {t if isinstance(t, str) else t["task_id"] for t in json.loads(proto.read_text())["panels"]["test"]}
    d = dict(reps=len(reps), succ=sum(r["success"] is True for k in kn for r in k), known=sum(map(len, kn)), total=len(allrows[0]),
             returned_seed=as_seed and m != "native-seed", run=run.name, run_dir=str(run), test_eval=str(reps[min(reps)][0]),
             files={k: [str(f) for f in fs] for k, fs in sorted(reps.items())},
             source=f"Seed row (native-seed runs, {seed_run.name})" if as_seed else f"own run {run.name}")
    d["Q"] = 100 * sum(sum(r["success"] is True for r in k) / len(k) for k in kn) / len(kn) if all(kn) else None
    d["adjudicated_failures"] = adj = adjudicate(reps, allrows, expected)
    known = [{t for (t, _), r in rows.items() if r["success"] is not None} for rows in allrows]
    d["avg3_complete"] = set(reps) == {0, 1, 2} and all(
        {t for t, _ in rows} == expected and all(t in kt or t in adj for t in expected) for rows, kt in zip(allrows, known))
    if d["avg3_complete"] and model != LUNA:  # a task with no verdict in any run counts as a failure
        d["Q"] = 100 * sum(sum(r["success"] is True for r in rows.values()) for rows in allrows) / (len(expected) * len(allrows))
    d["n"], rr = geo(mean, ref, turns=model == LUNA)
    d.update(rr)
    return d


def rank(qs, q):
    qs = sorted(set(qs), reverse=True)
    return "best" if round(q, 6) == qs[0] else "second" if len(qs) > 1 and round(q, 6) == qs[1] else ""


def compute(model="deepseek-flash"):
    res = {}
    for bench in BENCHES:
        seed_run, seed_reps = seed_files(model, bench)
        ref = merged_mean_rows(seed_reps, row_reader(model))[1]
        c = {name: cell(model, bench, m, seed_run, seed_reps, ref) for name, m in ROWS}
        qs = [round(v["Q"], 6) for v in c.values() if v and v["avg3_complete"]]
        for v in c.values():
            if v:
                v["rank"] = rank(qs, v["Q"]) if v["avg3_complete"] else ""
        res[f"{model}|{bench}"] = c
    return res


def record(res):
    """mech_table1: Table 1 as printed, one block per model."""
    blocks, seedret = [], []
    for model in MODELS:
        turns = model == LUNA  # DeepSeek's rho^turns comes from stats
        cols = ("Q", "rho", "rho^turns", "rho^calls") if turns else ("Q", "rho", "rho^calls")
        head = ["Method"] + [f"{LABEL[b]} {c}" for b in TEX_BENCHES for c in cols]
        rows = []
        for n, _ in ROWS:
            r = [n]
            for b in TEX_BENCHES:
                v = res[f"{model}|{b}"][n]
                mark = rank([round(res[f"{model}|{b}"][x]["Q"], 6) for x, _ in ROWS], v["Q"])
                q = f"{v['Q']:.1f}"
                r += [f"**{q}**" if mark == "best" else f"<u>{q}</u>" if mark == "second" else q, f"{v['rho_trj']:.2f}"]
                r += ([f"{v['rho_turns']:.2f}"] if turns else []) + [f"{v['rho_inv']:.2f}"]
            rows.append(r)
        blocks.append(f"**{MODEL_TEX[model]}**\n\n" + logger.md_table(head, rows))
        seedret += [f"{n} on {LABEL[b]} ({MODEL_TEX[model]})" for b in BENCHES for n, _ in ROWS if (res[f"{model}|{b}"][n] or {}).get("returned_seed")]
    txt = ("Q = avg@3 over three test runs; ratios against the Seed skill row on a common task set.\n\n" + "\n\n".join(blocks)
           + "\n\nColumns: rho = `rho_trj` (input tokens), rho^turns = `rho_turns`, rho^calls = `rho_inv` (tool calls). "
           + ("Skills equal to the seed text (" + "; ".join(seedret) + ") report the Seed row's three runs." if seedret
              else "Every row reports its own three runs."))
    logger.record(__spec__.name, "table1", txt, S.runs_text([(model, b, m) for model in MODELS for b in BENCHES for _, m in ROWS]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(out_dir("table1")))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    res = {k: v for model in MODELS for k, v in compute(model).items()}
    (out / "table1.json").write_text(json.dumps(res, indent=1))
    if all(v and v["avg3_complete"] for c in res.values() for v in c.values()):
        record(res)
    else:
        print("Incomplete test repetitions: publication record not refreshed; see avg3_complete in table1.json.")
    for k, c in res.items():
        print(k)
        for n, v in c.items():
            if v:
                print(n, json.dumps({key: v[key] for key in ("Q", "reps", "avg3_complete", "known", "rho", "rho_inv", "rho_trj", "n")}))


if __name__ == "__main__":
    main()
