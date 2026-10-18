# Scheduling and RQ-test drivers

BiPA 的训练与重复测试入口在 [../src](../src/README.md)。本目录是扁平结构，只放调度/RQ 驱动和运行完整性工具；论文表格与图的生成在 [../src/analysis](../src/analysis/README.md)。

**迁移 RQ 驱动**（从原 BACH 目录迁入，使用 BiPA 配置和 harness-bench；默认预览，`--execute` 才真正调用模型；缓存可断点续跑）：

- `transfer_model.py` — 跨模型迁移：A 模型优化的 skill 原样用 B 模型在同一 benchmark 的 test 面板上评测。
- `transfer_bench.py` — 跨 benchmark 迁移：在 A 上优化的 skill 原样在 B 的 test 面板上评测（同一模型）。
- `transfer_bench_tb2auto.py` — TB2 ↔ AutoB 两个方向迁移扫描的子集驱动（rep0）。

**实时状态与主表**（离线、只读 run 目录、不调 API）：

- `main_table.py` — 按各 cell 当前 run 直接打分的主结果表（五个 arm：native-noseed、native-seed、skillopt、skilladam、gepa），写 `assets/artifacts/results/main-table.{md,csv}` 和 `per_task.csv`；也是迁移驱动共用的配对打分库。
- `weekly_status.py` — 比较网格的实时状态块（进行中轮次/评测进度、异常标记）。
- `render_native_skill_history.py` — 导出各 cell 当前 run 到 `results/<model>/<method>/<bench>/`（summary、逐轮 skill、final）。
- `refresh_results.sh` — 一次离线刷新：逐轮 skill 导出 + 主表 + 状态块，串行执行上面三项。

**运行证据审计**：

- `audit_native_run.py` — run 结束时生成 completion certificate；summary 存在和审计通过是不同状态。
- `audit_skill_history.py` — 从持久化的优化器日志推导证据链逐轮 skill 历史，不改优化器状态。
- `audit_trace_integrity.py` — 对选中 episode 的轨迹与原始证据做只读完整性审计。

在 bipa 根目录、使用 bach-harness venv 的解释器（`bach-harness/.venv/bin/python`）：

```bash
python scripts/transfer_model.py --help
python scripts/transfer_bench.py --help
python scripts/main_table.py
bash scripts/refresh_results.sh
```

前两条只显示帮助；`main_table.py` 会刷新主表文件。所有表格必须区分重复次数、结局覆盖和轨迹完整性。
