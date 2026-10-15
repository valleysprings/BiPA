"""Numbers the Pareto figure needs that main_table.evaluation/pair must compute (run with the harness-bench / slim venv python).

Writes assets/artifacts/results/pareto-inputs.json:
  skilladam[model][bench] = {Q, rho, n, train_episodes}  (finished SkillAdam runs only)
  run_seed_Q[model][bench][arm] = test Q of the seed as evaluated inside that arm's own run (arm in skillopt, gepa, skilladam, zo)
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import main_table as M
from src.layout import SKILLS, current_run

seed_vid = M.version((SKILLS / "seed/SKILL.md").read_text())
ours = json.loads((ROOT / "bach-harness/config/ours_runs.json").read_text())
out = {"skilladam": {}, "run_seed_Q": {}}
q = lambda rows: M.pair(rows, None)["Q"]
for model in M.MODELS:
    for bench, _ in M.BENCHES:
        seeds = out["run_seed_Q"].setdefault(model, {}).setdefault(bench, {})
        ref = current_run("seed", model, bench)
        if ref and (ref / "summary.json").is_file() and M.evaluation(ref, seed_vid):
            seeds["native-seed"] = q(M.evaluation(ref, seed_vid))
        for arm in ("skillopt", "gepa", "skilladam"):
            run = current_run(arm, model, bench)
            if run and (run / "summary.json").is_file() and M.evaluation(run, seed_vid):
                seeds[arm] = q(M.evaluation(run, seed_vid))
                if arm == "skilladam":
                    sm = json.loads((run / "summary.json").read_text())
                    f = M.evaluation(run, M.version(sm["final_skill"]))
                    if f:
                        p = M.pair(f, M.evaluation(run, seed_vid))
                        out["skilladam"].setdefault(model, {})[bench] = {"Q": p["Q"], "rho": p["rho"], "n": p["rho_n"], "train_episodes": sm["arm_episodes"]["train"]}
        zo = current_run("zo", model, bench)
        if (zo / "summary.json").is_file():
            seeds["zo"] = 100 * json.loads((zo / "summary.json").read_text())["test"]["seed"]["success"]
dest = ROOT / "assets/artifacts/results/pareto-inputs.json"
dest.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
