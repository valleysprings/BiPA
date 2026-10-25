"""Copy the seed and the 15 returned skills into OUT/skills/ with a README table, each checked against the skill delivered at test.

Run:  python -m aux.rule_coding.release_skills   (then compare with the tracked skills/ and copy what changed)
"""
import json, re

from src.layout import EXECUTION, OUT as OUT_ROOT, SEED_FILE

EX = EXECUTION
OUT = OUT_ROOT / "skills"
RUNS = {"deepseek-v4.1-flash": ("deepseek-flash", "native-20261008-https"), "gpt-6-luna": ("gpt-6-luna", "native-20261012")}
OPTS = {"gepa": "GEPA", "skillopt": "SkillOpt", "skilladam": "SkillAdam"}
BENCHES = {"tau3-bench": "tau3-bench", "automationbench": "automationbench", "terminal-bench-2-1": "terminal-bench-2.1"}
SUSPECT = re.compile(r"/mnt/|/home/|sk-[A-Za-z0-9]{16,}|api[_-]?key|token\s*[:=]|password|[\w.+-]+@[\w-]+\.[a-z]{2,}|BEGIN [A-Z ]*PRIVATE", re.I)


def delivered(run, split):
    """The SKILL.md bytes delivered to the agent in the run's evaluations of a split."""
    return {p.read_bytes() for p in (run / "results/skills/deliveries/evaluations" / split).glob("*/native/SKILL.md")}


def main():
    """Write the copies and the README table; print the table and any problem found."""
    seed = SEED_FILE.read_bytes()
    rows = []
    problems = []
    (OUT / "seed").mkdir(parents=True, exist_ok=True)
    (OUT / "seed/SKILL.md").write_bytes(seed)
    rows.append(("—", "—", "Seed (hand-written)", len(seed.decode().split()), "skills/seed", "yes (every run's seed)"))
    for model_out, (model, tag) in RUNS.items():
        for bench, bench_out in BENCHES.items():
            for opt, name in OPTS.items():
                run = EX / model / opt / bench / tag
                final = (run / "final/SKILL.md").read_bytes()
                summary = json.loads((run / "summary.json").read_text())["final_skill"]
                if final.decode().strip() != summary.strip():
                    problems.append(f"{model} {opt} {bench}: final/SKILL.md != summary.final_skill")
                test_skills = delivered(run, "test")
                verified = "yes" if final in test_skills else f"NO ({len(test_skills)} test deliveries)"
                if m := SUSPECT.search(final.decode()):
                    problems.append(f"{model} {opt} {bench}: suspicious text {m.group(0)!r}")
                if final == seed:
                    rows.append((model_out, bench_out, name, len(final.decode().split()), tag, verified + "; returned the seed, see seed/"))
                    continue
                d = OUT / model_out / bench_out / opt
                d.mkdir(parents=True, exist_ok=True)
                (d / "SKILL.md").write_bytes(final)
                rows.append((model_out, bench_out, name, len(final.decode().split()), tag, verified))
    if SUSPECT.search(seed.decode()):
        problems.append("seed: suspicious text")

    L = ["# Skills", "",
         "The hand-written seed skill and the skill each optimizer returned in the paper's runs (OpenCode harness; three optimizers x three benchmarks x two agent models). "
         "Each `SKILL.md` is byte-identical to the run's `final/SKILL.md` and to the skill delivered to the agent in the test evaluations (checked against the delivered `native/SKILL.md`).", "",
         "GEPA rewrites the whole skill, including its `name` and `description` frontmatter; SkillOpt and SkillAdam edit the body and keep the seed's name and description. "
         "With GPT-6-Luna on Terminal-Bench 2.1 all three optimizers return the seed unchanged (no dev-set gain separated a candidate from it), so those cells point to `seed/` instead of a copy.", "",
         "| model | benchmark | optimizer | words | source run tag | delivered at test |", "|---|---|---|---|---|---|"]
    for r in rows:
        path = "seed/SKILL.md" if r[2].startswith("Seed") or "returned the seed" in r[5] else f"{r[0]}/{r[1]}/{[k for k, v in OPTS.items() if v == r[2]][0]}/SKILL.md"
        L.append(f"| {r[0]} | {r[1]} | [{r[2]}]({path}) | {r[3]} | {r[4]} | {r[5]} |")
    L += ["", "Words are whitespace-separated tokens of the whole file (frontmatter included).", ""]
    (OUT / "README.md").write_text("\n".join(L))
    print("\n".join(L))
    print("PROBLEMS:", problems or "none")


if __name__ == "__main__":
    main()
