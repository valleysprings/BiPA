"""The second-model table of the appendix (tab:luna): GPT-6 Luna without a skill and with the seed skill, one test run per arm.

Same columns as Table 1 (analysis/table1.py): Q = test success over the tasks with a known outcome, rho and rho^calls = the geometric
mean over the common tasks of the per-task ratio of input tokens and of tool calls to the Seed skill row (analysis/report.py ratios).
Luna's episodes contain provider-failed requests (connection resets, rate limits, no capacity) that return no completion and carry no
token count; the graph allocator (graph/episode.py) drops every episode that has one. These columns need no graph, so this table reads each
episode's outcome, input tokens and tool calls straight from evaluation.json and sums the tokens over the completed requests.
Inputs: the native seed and noseed runs of gpt-6-luna (run id native-20261011, src/configs/experiments.yaml).
Outputs: the LaTeX body rows on stdout and the audit record paper/assets/data/mech/appC_luna.md (analysis/paperout.record).

    /mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation/bach-harness/.venv/bin/python -m analysis.luna_table   # run from bipa/
"""

import glob
import json
from pathlib import Path
from types import SimpleNamespace

from analysis import report
from analysis import paperout as PO

H = Path("/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation")
MODEL = "gpt-6-luna"
RUN = "native-20261011"
BENCHES = [("terminal-bench-2-1", "TB2"), ("tau3-bench", r"\taubench{}"), ("automationbench", "AutoBench")]
ARMS = [("native-noseed", "No skill"), ("native-seed", "Seed skill")]
EXECUTION = H / "assets/artifacts/execution" / MODEL


def episode(record: dict) -> SimpleNamespace:
    """One task's episode with the fields report.stats / report.ratios read: task, success (None when the outcome is not verified),
    X = input tokens summed over the episode's distinct completed requests, N = the tool calls the harness executed."""
    # a provider-failed request has no usage (input_tokens None) and no completion; the client retried it
    requests = [r for r in {r["request_id"]: r for r in record["requests"]}.values() if r["input_tokens"] is not None]
    calls = [c for r in requests for c in r.get("tool_calls", []) if c.get("executed") is not False]
    return SimpleNamespace(task=record["task_id"],
                           success=record["success"] if record["outcome_valid"] else None,
                           X=sum(r["input_tokens"] for r in requests),
                           N=len(calls))


def test_episodes(arm: str, bench: str) -> tuple[list, str]:
    """The test episodes of one arm on one benchmark (its single test run) and the evaluation file they come from."""
    path = sorted(glob.glob(str(EXECUTION / arm / bench / RUN / "results/evaluations/test/*/evaluation.json")))[0]
    return [episode(r) for r in json.load(open(path))["episodes"]], path


def main():
    cells = {}
    raw = []
    for bench, _ in BENCHES:
        seed_eps, _ = test_episodes("native-seed", bench)
        for arm, _ in ARMS:
            eps, path = test_episodes(arm, bench)
            raw.append(path)
            run = report.stats([e.success for e in eps], eps)
            # the ratios of the arm's run against the seed's run on the tasks both have (1.00 for the seed itself)
            ratio = report.ratios([eps], seed_eps)
            cells[(arm, bench)] = {"Q": 100 * run["success"], "known": run["known"], "tasks": run["tasks"],
                                   "rho": ratio["input"], "rho_calls": ratio["calls"], "common": ratio["tasks"]}

    # LaTeX body rows in Table 1's column order
    for arm, label in ARMS:
        parts = [label]
        for bench, _ in BENCHES:
            c = cells[(arm, bench)]
            parts += [f"{c['Q']:.1f}", f"\\heat{{{c['rho']:.2f}}}", f"\\heat{{{c['rho_calls']:.2f}}}"]
        print(" & ".join(parts) + r" \\")

    rows = [[label, name, f"{cells[(arm, b)]['Q']:.1f}", f"{cells[(arm, b)]['known']}/{cells[(arm, b)]['tasks']}",
             f"{cells[(arm, b)]['rho']:.3f}", f"{cells[(arm, b)]['rho_calls']:.3f}", cells[(arm, b)]["common"]]
            for arm, label in ARMS for b, name in BENCHES]
    PO.record("appC_luna", "table",
              result=PO.mdtable(["Arm", "Benchmark", "Q", "known/tasks", "rho", "rho^calls", "common tasks"], rows),
              source=["bipa/analysis/luna_table.py", "bipa/analysis/report.py"],
              commands=["/mnt/shared-storage-gpfs2/safewt2-share/proj_harness/harness-mitigation/bach-harness/.venv/bin/python -m analysis.luna_table  # from bipa/"],
              raw=raw,
              notes=["One test run per arm (run native-20261011); ratios against the Seed skill run on the tasks with a verified outcome and nonzero tokens and calls in both.",
                     "Tokens and calls are read from evaluation.json: input_tokens summed over the distinct completed requests (provider-failed requests carry no usage and are retried), executed tool calls."])


if __name__ == "__main__":
    main()
