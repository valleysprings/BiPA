"""mech_transfer: the skills the three optimizers learned with DeepSeek-V4.1-Flash on tau3-bench, run unchanged with GPT-6 Luna on tau3-bench's
test split (one run each), next to Luna's own No skill and Seed skill runs (paper 1, single-run cross-model check).  The transfer job's own Seed
reference (ood-seed) is a second, independent Seed skill run: its row against the first is the seed's run-to-run spread on Luna.

Per row: Q over the row's verified tasks (and their count), Q of the row and of the Seed skill on the tasks both verified, and rho, rho^calls:
the geometric mean over those common tasks of the per-task ratio of input tokens and of tool calls to the Seed skill run (analysis/report.ratios,
the exact logic of bach/report.py that Table 1 uses).  Episodes are read from evaluation.json with analysis.luna_table.episode
(provider-failed requests carry no usage).
A transferred skill whose run has not finished is listed as pending.

  .venv/bin/python -m analysis.mech.transfer        -> paper/assets/data/mech/mech_transfer.md
"""
import glob
import json

from analysis import report
from analysis.luna_table import episode
from analysis.mech import common as M

LUNA = M.H / "assets/artifacts/execution/gpt-6-luna"
BENCH = "tau3-bench"
ROWS = [("No skill", f"native-noseed/{BENCH}/native-20261011"), ("Seed skill", f"native-seed/{BENCH}/native-20261011"),
        ("Seed skill, second run (noise reference)", f"native-seed/{BENCH}/ood-seed"),
        ("GEPA (DeepSeek-trained)", f"gepa/{BENCH}/ood-from-deepseek-flash"),
        ("SkillOpt (DeepSeek-trained)", f"skillopt/{BENCH}/ood-from-deepseek-flash"),
        ("SkillAdam (DeepSeek-trained)", f"skilladam/{BENCH}/ood-from-deepseek-flash")]


def test_episodes(run: str):
    """(episodes, evaluation path) of a run's test evaluation, or (None, None) while it has none."""
    files = sorted(glob.glob(str(LUNA / run / "results/evaluations/test/*/evaluation.json")))
    if not files:
        return None, None
    return [episode(r) for r in json.load(open(files[0]))["episodes"]], files[0]


def main():
    """Write the record with one row per arm."""
    seed, _ = test_episodes(ROWS[1][1])
    seed_y = {e.task: e.success for e in seed if e.success is not None}
    rows, raw = [], []
    for label, run in ROWS:
        eps, path = test_episodes(run)
        if eps is None:
            rows.append([label, "pending", "", "", "", "", ""])
            continue
        raw.append(path)
        own = report.stats([e.success for e in eps], eps)
        common = [e for e in eps if e.success is not None and e.task in seed_y]
        q_common = 100 * sum(e.success for e in common) / len(common)
        q_seed_common = 100 * sum(seed_y[e.task] for e in common) / len(common)
        ratio = report.ratios([eps], seed)
        rows.append([label, f"{100 * own['success']:.1f} ({own['known']}/{own['tasks']})", f"{q_common:.1f} vs {q_seed_common:.1f}",
                     len(common), f"{ratio['input']:.2f}", f"{ratio['calls']:.2f}", ratio["tasks"]])
    table = M.md_table(["arm (GPT-6 Luna, tau3-bench test)", "Q, own verified tasks", "Q vs Seed on common verified tasks", "common verified tasks",
                        "rho (input tokens)", "rho^calls", "tasks in the ratios"], rows)
    M.write_record(
        "transfer",
        "Cross-model transfer: the final skills of GEPA, SkillOpt and SkillAdam trained with DeepSeek-V4.1-Flash on tau3-bench (the Table 1 runs), "
        "evaluated unchanged with GPT-6 Luna on the tau3-bench test split, one run each, against Luna's own Seed skill and No skill runs.",
        "Luna runs: native-seed and native-noseed native-20261011 (test), native-seed ood-seed (the transfer job's Seed reference, a second "
        "independent run); transferred skills: <optimizer>/tau3-bench/ood-from-deepseek-flash "
        "(bipa/scripts/jobs/transfer_model.py, trained runs native-20261008-https). Evaluation files: " + "; ".join(raw) + ".",
        "Q: success over verified outcomes. Common verified tasks: verified in both the row and the Seed skill run. rho, rho^calls: geometric mean "
        "over the tasks both runs have with a verified outcome and positive input tokens and calls of the per-task ratio to the Seed skill run "
        "(analysis/report.ratios). Input tokens sum over completed requests. One test run per row.",
        table,
        ".venv/bin/python -m analysis.mech.transfer")


if __name__ == "__main__":
    main()
