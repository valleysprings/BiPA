# Skills

The hand-written seed and the skill each optimizer returned in the recorded runs, exactly as the agent received them at test time.

GEPA rewrites the whole skill, including its `name` and `description`; SkillOpt and SkillAdam edit the body and keep the seed's name and description. With GPT-6-Luna on Terminal-Bench 2.1 GEPA and SkillOpt return the seed unchanged, so those cells have no separate file.

| benchmark | DeepSeek-V4.1-Flash | GPT-6-Luna |
|---|---|---|
| τ3-bench | [GEPA](deepseek-v4.1-flash/tau3-bench/gepa/SKILL.md) · [SkillOpt](deepseek-v4.1-flash/tau3-bench/skillopt/SKILL.md) · [SkillAdam](deepseek-v4.1-flash/tau3-bench/skilladam/SKILL.md) | [GEPA](gpt-6-luna/tau3-bench/gepa/SKILL.md) · [SkillOpt](gpt-6-luna/tau3-bench/skillopt/SKILL.md) · [SkillAdam](gpt-6-luna/tau3-bench/skilladam/SKILL.md) |
| AutomationBench | [GEPA](deepseek-v4.1-flash/automationbench/gepa/SKILL.md) · [SkillOpt](deepseek-v4.1-flash/automationbench/skillopt/SKILL.md) · [SkillAdam](deepseek-v4.1-flash/automationbench/skilladam/SKILL.md) | [GEPA](gpt-6-luna/automationbench/gepa/SKILL.md) · [SkillOpt](gpt-6-luna/automationbench/skillopt/SKILL.md) · [SkillAdam](gpt-6-luna/automationbench/skilladam/SKILL.md) |
| Terminal-Bench 2.1 | [GEPA](deepseek-v4.1-flash/terminal-bench-2.1/gepa/SKILL.md) · [SkillOpt](deepseek-v4.1-flash/terminal-bench-2.1/skillopt/SKILL.md) · [SkillAdam](deepseek-v4.1-flash/terminal-bench-2.1/skilladam/SKILL.md) GEPA and SkillOpt return the [seed](seed/SKILL.md) · [SkillAdam](gpt-6-luna/terminal-bench-2.1/skilladam/SKILL.md) |

Seed: [seed/SKILL.md](seed/SKILL.md). SkillAdam gates on dev batches under DeepSeek-V4.1-Flash and on its training batches under GPT-6-Luna (the runs pinned in `configs/experiments.yaml`).

To evaluate a skill, see Reproduction in the [top-level README](../README.md): `python -m src.run repeat` runs the three test repetitions of a returned skill, and `python -m src.run transfer` runs one unchanged under another model or on another benchmark.
