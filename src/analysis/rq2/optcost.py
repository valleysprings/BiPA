"""Paper 1 (mechanism paper), reviewer X2: what does the optimization itself cost, next to what the learned skill costs per episode?

Role in the argument: the paper says a learned skill carries a hidden input-token tax on every later episode. A reviewer asks for the optimizer-side bill, so that
the tax can be set against it: after how many deployment episodes does the tax alone exceed what the optimization spent?

For each of the nine optimizer runs (GEPA, SkillOpt, SkillAdam x tau3, AutoBench, TB2; run = native-20261008-https as in Table 1) this module reads only recorded data:
  - optimizer LLM calls: <run>/llm/call-*.json (one file per successful call, response.usage = prompt / completion tokens) and llm/attempts.jsonl
    (every attempt incl. errors that were retried). summary.json["llm"] repeats the totals, except GEPA on TB2 where it is empty ({}), so the call logs are the source.
  - agent rollouts of the optimization: <run>/results/evaluations/{train,dev}/*/evaluation.json, one file per (skill version, split). Every episode carries its requests
    (input_tokens, output_tokens per request, de-duplicated by request_id exactly as analysis.common.task_row does) and `collection_mode`; `evaluation_cache_hit`
    marks an episode that was not executed in this run but served from the shared evaluation cache (its tokens were paid by an earlier run of the same skill and task).
    The `test` split of the run directory is the final evaluation of Table 1 and is NOT part of the optimization cost.
  - the seed test episode: the three seed test runs of Table 1 (native-seed run + rep-1, rep-2); median / mean of X per episode over the pooled episodes.
  - the tax per episode: mean over Table 1's common task set of (X_skill - X_seed), with X the per-task mean over the three test runs of each side (stats.task_means,
    stats.common_tasks). Break-even episodes = optimization input tokens / tax (only defined if the tax is positive).

Two bills are reported: "nominal" counts every train/dev episode of the run (what a fresh user without the shared cache would pay), "executed" only the episodes with
collection_mode != evaluation_cache_hit (what this run itself spent). Tokens are input tokens (prompt) unless named output.

No BACH, no model calls. Commands (from bipa):           .venv/bin/python -m src.analysis.rq2.optcost  -> assets/cache/analysis/rq/optcost.json, paper/assets/data/mech/mech_optcost.md
"""
import json
from statistics import median

import numpy as np

from src.analysis import main as T
from src.utils.dataloader import BENCHES, EX, SEED_TEXT, current_run, final_skill, task_row, vid
from src.utils import logger, dataloader
from src.analysis.rq1 import stats as S

MODEL = "deepseek-flash"
METHODS = (("GEPA", "gepa"), ("SkillOpt", "skillopt"), ("SkillAdam", "skilladam"))
SHORT = {"tau3-bench": "tau3", "automationbench": "AutoB", "terminal-bench-2-1": "TB2"}
SPLITS = ("train", "dev")


def llm_calls(run):
    """Optimizer LLM calls of a run: successful calls, their prompt / completion tokens (response.usage), attempts that errored, and the summary.json totals."""
    prompt = completion = 0
    files = sorted((run / "llm").glob("call-*.json"))
    for f in files:
        usage = json.loads(f.read_text())["response"].get("usage", {})
        prompt += usage.get("prompt_tokens", 0)
        completion += usage.get("completion_tokens", 0)
    errors = 0
    attempts = run / "llm/attempts.jsonl"
    if attempts.is_file():
        for line in attempts.read_text().splitlines():
            record = json.loads(line)
            errors += record.get("event") == "result" and record.get("outcome") == "error"
    summary = json.loads((run / "summary.json").read_text())
    return {"calls": len(files), "prompt": prompt, "completion": completion, "errors": errors,
            "summary_llm": summary.get("llm", {}), "summary_episodes": summary.get("episodes", {}),
            "truncated_calls": summary.get("llm", {}).get("truncated_calls", 0)}


def rollouts(run):
    """Agent rollouts of the optimization (train + dev eval files): episode counts and input / output tokens, nominal and executed-only, per split."""
    out = {}
    for split in SPLITS:
        cell = dict(files=0, episodes=0, x_in=0, x_out=0, unknown_x=0, exec_episodes=0, exec_x_in=0, exec_x_out=0, requests=0)
        for f in sorted((run / "results/evaluations" / split).glob("*/evaluation.json")):
            cell["files"] += 1
            for episode in json.loads(f.read_text())["episodes"]:
                row = task_row(episode)
                seen = set()
                x_out = 0
                for request in episode.get("requests", []):
                    if request["request_id"] in seen:
                        continue
                    seen.add(request["request_id"])
                    x_out += request.get("output_tokens") or 0
                x_in = row["X"]
                if x_in is None:
                    # capture incomplete: fall back to the recorded request sizes so the bill is a lower-bound-free sum, and count the episode
                    cell["unknown_x"] += 1
                    x_in = sum(r["input_tokens"] for r in {r["request_id"]: r for r in episode.get("requests", [])}.values()
                               if isinstance(r["input_tokens"], (int, float)))
                cell["episodes"] += 1
                cell["x_in"] += x_in
                cell["x_out"] += x_out
                cell["requests"] += row["nreq"]
                if episode.get("collection_mode") != "evaluation_cache_hit":
                    cell["exec_episodes"] += 1
                    cell["exec_x_in"] += x_in
                    cell["exec_x_out"] += x_out
        out[split] = cell
    return out


def seed_episode_cost(bench):
    """Input tokens per seed test episode over the pooled episodes of the three seed runs: (median, mean, number of episodes), plus the seed's per-task means."""
    seed_run = current_run("native-seed", MODEL, bench)
    reps = T.rep_files([seed_run, seed_run.parent / "rep-1", seed_run.parent / "rep-2"], vid(SEED_TEXT))
    per = S.per_task_runs(reps)
    xs = [r["X"] for runs in per.values() for r in runs]
    return {"median": float(median(xs)), "mean": float(np.mean(xs)), "n": len(xs)}, S.task_means(per)


def tax_per_episode(bench, name, method, run, ref):
    """Mean over the common task set of (X_skill - X_seed), per-task means over the three test runs, and the geometric-mean-free ratio of sums."""
    reps = T.rep_files([run, EX / MODEL / "test-reps" / f"{bench}__{method}__{run.name}"], vid(final_skill(method, run)))
    own = S.task_means(S.per_task_runs(reps))
    tasks = S.common_tasks(ref, own)
    diff = np.array([own[t]["X"] - ref[t]["X"] for t in tasks])
    return {"tax_mean": float(diff.mean()), "n_common": len(tasks), "x_seed_mean": float(np.mean([ref[t]["X"] for t in tasks])),
            "x_own_mean": float(np.mean([own[t]["X"] for t in tasks]))}


def compute():
    """All numbers {bench: {method: {...}}}, plus the seed episode cost per benchmark."""
    res = {}
    for bench in BENCHES:
        seed_cost, ref = seed_episode_cost(bench)
        cell = {"seed_test_episode": seed_cost, "methods": {}}
        for name, method in METHODS:
            run = current_run(method, MODEL, bench)
            llm = llm_calls(run)
            roll = rollouts(run)
            tax = tax_per_episode(bench, name, method, run, ref)
            nominal = sum(roll[s]["x_in"] for s in SPLITS)
            executed = sum(roll[s]["exec_x_in"] for s in SPLITS)
            total_nominal = nominal + llm["prompt"]
            total_executed = executed + llm["prompt"]
            cell["methods"][name] = {
                "run": run.name, "llm": llm, "rollouts": roll, "tax": tax,
                "rollout_in_nominal": nominal, "rollout_in_executed": executed,
                "total_in_nominal": total_nominal, "total_in_executed": total_executed,
                "seed_episodes_nominal": total_nominal / seed_cost["median"],
                "seed_episodes_executed": total_executed / seed_cost["median"],
                "breakeven_nominal": total_nominal / tax["tax_mean"] if tax["tax_mean"] > 0 else None,
                "breakeven_executed": total_executed / tax["tax_mean"] if tax["tax_mean"] > 0 else None,
            }
        res[bench] = cell
    return res


def m(x):
    """Millions of tokens, one decimal."""
    return f"{x / 1e6:.1f}M"


def record(res):
    """Write mech_optcost.md: one table per run and the summary tables."""
    calls = sum(c["methods"][n]["llm"]["calls"] for c in res.values() for n, _ in METHODS)
    cut = {n: sum(c["methods"][n]["llm"]["truncated_calls"] for c in res.values()) for n, _ in METHODS}
    head_a = ["run", "optimizer calls", "of which cut at 512k", "failed attempts", "opt. in tok", "opt. out tok", "summary.json llm.calls"]
    rows_a = []
    for bench in BENCHES:
        for name, _ in METHODS:
            r = res[bench]["methods"][name]
            l = r["llm"]
            rows_a.append([f"{name} {SHORT[bench]}", l["calls"], l["truncated_calls"], l["errors"], m(l["prompt"]), m(l["completion"]), l["summary_llm"].get("calls", "empty ({})")])
    rows_a.append(["**all nine**", calls, f"GEPA {cut['GEPA']}, SkillOpt {cut['SkillOpt']}, SkillAdam {cut['SkillAdam']}", "", "", "", ""])
    head_b = ["run", "train eps (nominal / executed)", "dev eps (nominal / executed)", "rollout in tok nominal", "rollout in tok executed", "rollout out tok nominal", "eps with incomplete capture"]
    rows_b = []
    for bench in BENCHES:
        for name, _ in METHODS:
            ro = res[bench]["methods"][name]["rollouts"]
            rows_b.append([f"{name} {SHORT[bench]}",
                           f"{ro['train']['episodes']} / {ro['train']['exec_episodes']}", f"{ro['dev']['episodes']} / {ro['dev']['exec_episodes']}",
                           m(sum(ro[s]["x_in"] for s in SPLITS)), m(sum(ro[s]["exec_x_in"] for s in SPLITS)), m(sum(ro[s]["x_out"] for s in SPLITS)),
                           sum(ro[s]["unknown_x"] for s in SPLITS)])
    head_c = ["run", "seed test ep. median X", "total in tok (rollouts + optimizer calls), nominal", "= seed test eps", "executed only", "= seed test eps",
              "tax / episode (tok)", "tax / seed X", "break-even episodes (nominal)", "break-even (executed)"]
    rows_c = []
    for bench in BENCHES:
        sc = res[bench]["seed_test_episode"]
        for name, _ in METHODS:
            r = res[bench]["methods"][name]
            tax = r["tax"]["tax_mean"]
            be_n, be_e = r["breakeven_nominal"], r["breakeven_executed"]
            rows_c.append([f"{name} {SHORT[bench]}", f"{sc['median']:,.0f}", m(r["total_in_nominal"]), f"{r['seed_episodes_nominal']:.0f}",
                           m(r["total_in_executed"]), f"{r['seed_episodes_executed']:.0f}", f"{tax:+,.0f}", f"{100 * tax / r['tax']['x_seed_mean']:+.0f}%",
                           f"{be_n:,.0f}" if be_n else "never (tax <= 0)", f"{be_e:,.0f}" if be_e else "never (tax <= 0)"])
    table = "\n\n".join([
        "**(a) Optimizer LLM calls** (llm/call-*.json; usage = prompt / completion tokens)", logger.md_table(head_a, rows_a),
        "**(b) Agent rollouts of the optimization** (results/evaluations/train+dev; test split excluded)", logger.md_table(head_b, rows_b),
        "**(c) Cost against the seed test episode and the per-episode tax**", logger.md_table(head_c, rows_c)])
    logger.write_record(
        "optcost",
        "Reviewer X2: the optimizer-side cost of each of the nine optimizer runs (optimizer LLM calls, training and development rollouts, input tokens), in units of one seed test episode, and the number of "
        "deployment episodes after which the learned skill's input-token tax exceeds that cost.",
        "Optimizer runs native-20261008-https (GEPA, SkillOpt, SkillAdam x tau3-bench, automationbench, terminal-bench-2-1), DeepSeek-V4.1-Flash in OpenCode; seed = native-seed run + rep-1, rep-2; tax from the test runs of Table 1 (native-20261008-https + test-reps).",
        "Optimizer calls: llm/call-*.json files. Rollouts: every episode of results/evaluations/{train,dev}; nominal = all episodes, executed = collection_mode != evaluation_cache_hit; input tokens = sum of request input_tokens (de-duplicated). "
        "Seed test episode = median input tokens over the pooled episodes of the three seed runs. Tax = mean over Table 1's common task set of (skill X - seed X), per-task means over three runs. Break-even = (rollout + optimizer-call input tokens) / tax. "
        "Quantities not recorded: the optimizer's wall-clock and dollar cost; the agent episodes' cost of cache-hit episodes in this run (their tokens are in the episode record but were paid by an earlier run).",
        table, ".venv/bin/python -m src.analysis.rq2.optcost")


def main():
    res = compute()
    dataloader.write_json("optcost", res)
    record(res)


if __name__ == "__main__":
    main()
