# External interfaces

- `executor.py` 封装 `harness_bench.eval.run`、任务结果及轨迹转换。执行、缓存、sandbox 和基础设施重试仍由 harness-bench 负责。
- `llm.py` 封装 optimizer 的模型传输、调用记录与重试；优化器决定何时提案、如何接受和返回候选。

使用 `src.utils.executor`、`src.utils.llm` 导入，`PYTHONPATH` 包含 `bipa/`。不使用顶层 `utils` 或 `executor` 名称，避免与 BACH 的 `src/utils` 混淆。BiPA 启动入口和迁移脚本已设置包路径。

上游算法在 [../optimizers](../optimizers/README.md)，接口适配在 `../adapters/`；面板、冻结协议和审计仍由各自模块负责。
