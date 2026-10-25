"""Write mech_attribution3.md from attribution.json (written by the attribution compute step).
Run:  python -m aux.token_attribution.attribution_draw"""
import json
import math

from src.utils import dataloader, logger

TERMS = ["more_actions", "turn_splitting", "content", "skill_text"]
TERM_LABEL = dict(more_actions="more actions", turn_splitting="turn-splitting", content="content per turn", skill_text="skill text")
PREFIX = ["system", "tool_schemas", "task", "skill"]


def f(x, nd=3):
    return f"{x:+.{nd}f}"


def record(cells):
    """OUT/data/mech/mech_attribution3.md."""
    rowsA = []
    for c in cells:
        a = c["A"]
        rowsA.append([c["method"], c["short"], f"{c['n']} / {c['n_table1']}"] + [f(a[k]) for k in TERMS] + [f(a["total"]), f(c["table1_log_rho_usable"]), f(a["residual_mean"]), f"{a['residual_max_abs']:.3f}", TERM_LABEL[a["largest"]]])
    rowsB = []
    for c in cells:
        for b in c["B"]:
            rowsB.append([c["method"], c["short"], b["run"]] + [f(b[k]) for k in TERMS] + [f(b["total"]), f(b["residual_mean"]), TERM_LABEL[b["largest"]]])
    rowsS = [[c["method"], c["short"], " / ".join(TERM_LABEL[b["largest"]] for b in c["B"]), "yes" if c["same_largest_all_runs"] else "no"] for c in cells]
    rowsD = [[c["method"], c["short"], c["n_table1"] - c["n"], "; ".join(f"{k}: {v}" for k, v in c["dropped"].items() if v) or "-"] for c in cells]
    rowsC = [[c["method"], c["short"]] + [f"{c['C'][p]['delta']:+.0f} (seed {c['C'][p]['seed']:.0f})" for p in PREFIX] for c in cells]
    rowsT = [[c["method"], c["short"], f"{c['table1_rho_full']:.3f}", f"{math.exp(c['table1_log_rho_usable']):.3f}", f"{math.exp(c['A']['total']):.3f}"] for c in cells]
    what = ("Four-term split of the change in log input tokens of each baseline skill against the seed, on the same data as Table 1: per task, R, N, X, F are averaged over the three test runs of each side "
            "before the split, so the total is Table 1's log rho_trj (mean over tasks of log(Xbar_skill / Xbar_seed)). Terms per task: more actions e_R dlogN, turn-splitting -e_R dlog(N/R), "
            "content per turn s_H dlogh, skill text s_F dlogF (midpoint shares, exact e_R = 1 + s_H R/(R-1)). The residual is total minus the sum of the four terms.")
    runs = "Skills: native-20261008-https (rep 0) + test-reps rep 1/2; seed: native-seed native-20261003-trace-final + rep-1, rep-2; DeepSeek-V4.1-Flash; GEPA, SkillOpt, SkillAdam x tau3-bench, automationbench, terminal-bench-2-1."
    agg = ("Tasks: Table 1's common task set restricted to tasks with a verified outcome and N, X in all three runs of both sides and a usable graph episode in all six runs (R>=2, an executed call, positive history). "
           "A) means over tasks of the per-task terms on 3-run means. B) the same terms on single run pairs (skill rep k vs seed rep k). C) mean over tasks of the per-request tokens of each fixed-prefix column, "
           "skill minus seed (per-task mean over the three runs).")
    table = ("### A. 3-run means (log units; n = tasks used / Table 1 common tasks)\n\n"
             + logger.md_table(["Skill", "Bench", "n", "actions", "turn-split", "content", "skill text", "total", "Table 1 log rho (same tasks)", "resid mean", "resid max |.|", "largest"], rowsA)
             + "\n\n### B. Per run pair (skill rep k vs seed rep k)\n\n"
             + logger.md_table(["Skill", "Bench", "run", "actions", "turn-split", "content", "skill text", "total", "resid mean", "largest"], rowsB)
             + "\n\nStability of the largest term (runs 0 / 1 / 2):\n\n" + logger.md_table(["Skill", "Bench", "largest per run", "same in all three (and in A)"], rowsS)
             + "\n\n### C. Fixed-prefix columns, change per request in tokens (skill minus seed)\n\n" + logger.md_table(["Skill", "Bench", "system", "tool_schemas", "task", "skill"], rowsC)
             + "\n\n### Table 1 consistency (rho_trj)\n\n" + logger.md_table(["Skill", "Bench", "Table 1 rho (full common set)", "rho on used tasks (Table 1 rows)", "exp(total) of A"], rowsT)
             + "\n\n### Dropped tasks (Table 1 common set minus used)\n\n" + logger.md_table(["Skill", "Bench", "dropped", "reason"], rowsD))
    logger.write_record("attribution3", what, runs, agg, table,
                        "python -m src.analysis.attribution compute\npython -m aux.token_attribution.attribution_draw")


def main():
    record(json.loads((dataloader.CACHE / "attribution.json").read_text())["cells"])


if __name__ == "__main__":
    main()
