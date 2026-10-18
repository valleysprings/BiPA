# Shell entry points

`scripts/` 只有 shell 入口；Python 实现在 `src/dynamics/`（`python -m src.dynamics.<module>`，从 bipa 根目录运行）。解释器为 bach-harness venv（`bach-harness/.venv/bin/python`），可用 `BASELINE_PYTHON` 覆盖。除 `refresh.sh` 外全部默认预览，不传 `--execute` 不会调用模型。

**baselines.sh** — 基线调度（原 `src/run.sh` → `src/launch.py`）：五个 arm（native-noseed、native-seed、skillopt、skilladam、gepa）× 模型 × bench 的训练与重复测试；默认只打印计划：

```bash
bash scripts/baselines.sh run tau3 gpt-6-luna --arm skillopt      # 单格优化（预览）
bash scripts/baselines.sh repeat tau3 gpt-6-luna --arm skillopt   # 重复测试（预览）
bash scripts/baselines.sh suite gpt-6-luna                        # 整个模型的套件（预览）
```

加 `--execute` 执行，`--detach` 后台。`baselines.sh` 会 source 仓库根的 `host.env`（代理等本机设置）。

**rq4_transfer.sh** — RQ4（generality）迁移评测驱动（`src/dynamics/transfer_*.py`；默认 dry run，缓存可断点续跑，`--execute` 才调模型）：

```bash
bash scripts/rq4_transfer.sh model --help     # 跨模型：A 的 final skill 原样用 B 评测
bash scripts/rq4_transfer.sh bench --help     # 跨 benchmark：同模型，A 上优化的 skill 在 B 的 test 面板评测
bash scripts/rq4_transfer.sh tb2auto          # TB2 ↔ AutoB 两方向扫描（rep0，dry run）
```

**audit.sh** — 运行证据审计（离线、只读原始 run）：

```bash
bash scripts/audit.sh native-run --run RUN_DIR [--output FILE]   # completion certificate（summary 存在 ≠ 审计通过）
bash scripts/audit.sh trace [--audit-dir DIR]                    # 选中 test episode 的轨迹完整性
bash scripts/audit.sh skill-history [--output-dir DIR]           # 从优化器日志推导证据链逐轮 skill 历史
```

**refresh.sh** — 一次离线刷新 `assets/artifacts/results/`：逐轮 skill 导出（`render_native_skill_history`）→ 五臂主表（`main_table`）→ 状态块（`weekly_status`），串行，不调 API：

```bash
bash scripts/refresh.sh
```

**analysis.sh** — 论文 RQ 分析的 composition（全离线，不调模型）：每个 target 内先 compute（venv python）再画图（系统 python3，可用 `FIG_PYTHON` 覆盖）；跨 target 有 cache 依赖（rq3 要 rq2 的 `appc/attr.json` 与 `mech/attribution.json`，cheatsheet 还要 rq4 的产物），`all` 按 `rq1 rq2 rq4 rq3 paper` 的顺序跑，满足全部约束：

```bash
bash scripts/analysis.sh rq1          # rq2 rq3 rq4 paper table1，可多个
bash scripts/analysis.sh all          # 全部，按依赖顺序
```

RQ1–3 中 manifest 驱动的模块（`mechanism/{nullmodel,paired,power}`）读 `BACH_SUBMISSION_MANIFEST`，脚本默认指向 `assets/cache/analysis/submission-20261008/protocol.json`（不存在时由 `src/analysis/submission_protocol.py` 重新冻结生成）。

论文表格与图的逐模块说明见主 README 的 RQ mapping。所有表格必须区分重复次数、结局覆盖和轨迹完整性。
