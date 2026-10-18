# bipa/src — 代码地图

BiPA 实验 harness 的源码树。详细的实验文档（调度、协议、重复测试、审计、预算）见[仓库根 README](../README.md)。

| 路径 | 负责什么 |
|---|---|
| `run.py` | 唯一的直接入口：单格优化运行（`scripts/baselines.sh` → `core/launch.py` → `run.py`），冻结协议、调用 adapter、评测最终 skill、保存 summary |
| `core/` | 内部机制：`launch.py`（调度）、`repeat.py` / `repeat_transfer.py`（冻结 skill 的重复评测）、`panels.py`（固定面板与训练批次）、`native_protocol.py` / `native_trace.py`（协议冻结与逐轮见证）、`layout.py`（共享路径与历史路径迁移）、`experiment_config.py`（run 选择与协议环境） |
| `adapters/` | 上游优化器接口适配（skillopt、skilladam、gepa）；不实现另一套搜索算法 |
| `analysis/` | 离线分析：Table 1、论文图表与数字（只读 `assets/artifacts`，不调用模型） |
| `dynamics/` | 运行时驱动：RQ4 迁移评测、run 证据审计、结果刷新 |
| `graph/` | episode 的请求×成分二分图（分配词汇在 `../configs/graph.yaml`） |
| `prompts/` | 各 adapter 的 prompt 文件 |
| `utils/` | 外部评测接口（`executor.py`）与 LLM 传输（`llm.py`），以 `src.utils` 导入 |

测试在仓库根的 `tests/`：

```bash
"$BASELINE_PYTHON" -m pytest tests -q -p no:cacheprovider   # 从 bipa/ 运行
```
