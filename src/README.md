# BiPA baselines

第一篇 draft2mech 的三个上游优化器：SkillOpt、SkillAdam、GEPA。第二篇的成本感知候选在 [../bach](../bach/README.md)。本目录从 `candidate-skill` 更名而来。

| 层 | 文件 | 负责什么 |
|---|---|---|
| 调度 | `run.sh` → `launch.py`，`configs/experiments.yaml` | 选择模型、run、并发；默认预览；完整套件包含五种方法的重复测试 |
| 单格运行 | `run.py` | 冻结协议，调用 adapter，评测最终 skill，保存 summary，调用离线审计 |
| 上游接口适配 | `adapters/skillopt.py`、`skilladam.py`、`gepa.py` | 转换任务、轨迹、verifier 分数与 LLM 接口；不实现另一套搜索算法 |
| 外部评测接口 | `utils/executor.py` | 封装 `harness_bench.eval.run` 及返回轨迹；执行、缓存和重试归 harness-bench |
| 提案传输 | `utils/llm.py` | 上游 optimizer 的 LLM 调用及调用记录 |
| 任务设置 | `panels.py`、`configs/tasks.yaml` | 固定面板和训练预算 |
| 上游源码 | `optimizers/skillopt`、`skilladam`、`gepa` | 各自原版优化循环，由 adapter 调用 |
| 重复测试 | `repeat.py` | 冻结最终 skill 和原测试面板，执行 rep 1、2，逐次检查覆盖 |
| 证据 | `native_protocol.py`、`native_trace.py`、`../scripts/analysis/` | 冻结设置、逐轮见证、完整性检查 |

优化循环与返回规则：

| adapter | 上游入口 | 返回 |
|---|---|---|
| SkillOpt | `ReflACTTrainer.train` | D_sel 最佳 skill |
| SkillAdam | `FeedbackLoop.run`，原 proposer/tracker 和 acceptance gate | 最后接受版本 |
| GEPA | `optimize_anything` / `GEPAEngine.run` | D_sel 候选池最佳版本 |

上游源码随 BiPA 维护在 [optimizers/](optimizers/README.md)，从原 `assets/optimizers/` 整体迁入；外部执行和模型接口统一在 [utils/](utils/README.md)，以 `src.utils` 导入，避免与 BACH 的 `utils` 包重名。适配并不等于论文原始设置的完整复现：本项目变更了 benchmark、预算、起始 skill、执行接口、缓存和传输。具体差异以每个 adapter 的 `DEVIATIONS` 和 run 的协议为准。当前 native-v1 向 SkillOpt / GEPA 传完整反馈；SkillAdam 使用上游的轨迹压缩及 prompt。训练或 dev 的 unknown 不转成失败分数；最终测试保留 unknown 并标明 known/total。

当前 full-train 的 τ³ / AutoB 训练预算为 160，D_sel 为 30；SkillOpt / GEPA / SkillAdam 分别 4 / 20 / 16 步。TB2 的训练池为 62，D_sel 为全部 9 题，分批后分别 5 / 20 / 18 步。实际冻结的 `native_protocol.json.batches` 是审计依据，不存在统一十轮要求。公共 seed 只有 `assets/skills/seed/SKILL.md` 一份。

从项目根目录：

```bash
export BASELINE_PYTHON="$PWD/bach-harness/.venv/bin/python"
# 单格优化、单格重复评测、整个模型的实验套件：均只预览。
bash bipa/src/run.sh run tau3 gpt-6-luna --arm skillopt
bash bipa/src/run.sh repeat tau3 gpt-6-luna --arm skillopt
bash bipa/src/run.sh suite gpt-6-luna --benches tau3 autob tb2
```

加 `--execute` 执行，或 `--detach` 后台顺序执行。suite 覆盖 Seed、No-skill、SkillOpt、GEPA、SkillAdam 的优化/控制运行及重复测试。已有 summary 的优化过程跳过；同一输出目录用锁防止调度器重复启动。单独使用 `run.py` 会直接执行，默认 native-v1，需要显式 cache root；历史 legacy 选项仅用于解释旧协议。

`BASELINE_PYTHON` 默认取当前环境的 `python`（依赖环境使用 Python 3.12）。已有 venv 可以复用，也可以使用安装了 harness-bench 和上游依赖的独立 Python；BiPA 不导入 `bach`、`graph` 或 `utils.experiments`。主机代理设置在 `../host.env`。原来借用 BACH 的 native / replicate / test_reps 入口已归档，不保留兼容入口。

重复评测读取原 run 的 `summary.json`、`final/SKILL.md`、`native_protocol.json`，使用原测试任务和缓存。即使返回 seed，也检查真实重复测试；相同 skill 的同一 repetition 由 harness-bench 缓存复用。输出为 `execution/<model>/test-reps/<bench>__<method>__<run>/results.json`。episode 已执行但在全部请求的 repetition 中都无判决的槽位（如持续 verifier 超时）判负：计 0、列入 `adjudicated_failures`、不再触发补跑；仍有槽位连 episode 证据都没有时不给 `avg`，保留每次 `Q_known` 和覆盖率。它是 Q 的重复测试报告，不替代请求/工具轨迹审计。已知 rep 0 不重跑；rep0 的未知或缺失槽位通过单独补充 evaluation 恢复，原 source 不改写。报告记录所有补充证据路径。论文 Table 1 的读取器按同一判负规则检查 avg3_complete，缺失的格保留待办。

`summary.json` 表示优化结束，`repetitions_complete` 表示重复评测覆盖，`completion_audit.json` 检查逐轮对象和完整轨迹，三者不能相互替代。[Luna 调查](docs/luna-20261014.md) 记录当前缺口。

离线测试：

```bash
"$BASELINE_PYTHON" -m pytest bipa/src/tests -q -p no:cacheprovider
```

测试使用伪造的模型/任务结果，覆盖上游循环的 unknown、resume、失败提案、审计和调度，不启动真实模型。跨模型/跨 benchmark 的评测脚本在 [../scripts/jobs](../scripts/README.md)，报告与论文分析由 [analysis](analysis/README.md) 负责。
