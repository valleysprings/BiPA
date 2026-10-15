# Shared analysis and evaluation jobs

BiPA 的训练与重复测试入口在 [../src](../src/README.md)；第二篇的新候选在 [../../bach-harness/baselines](../../bach-harness/baselines/README.md)。

`analysis/` 保存 `main_table.py`、`weekly_status.py`、`render_native_skill_history.py` 和离线审计。`audit_native_run.py` 在 run 结束时生成 completion audit；summary 和审计通过是不同状态。旧产物保留原状，换代码不自动覆盖历史证书。

`jobs/transfer_model.py` 和 `jobs/transfer_bench.py` 从原 BACH 目录迁入，使用 BiPA 配置和 harness-bench；默认预览，`--execute` 才执行。它们也支持读取我们方法的最终 skill 做同样的迁移评测。

在项目根目录、装好依赖的 Python 环境中：

```bash
python bipa/scripts/jobs/transfer_model.py --help
python bipa/scripts/jobs/transfer_bench.py --help
python bipa/scripts/analysis/main_table.py
```

前两条只显示帮助；最后一条会刷新主表文件。已有 `refresh_results.sh` 可刷新逐轮 skill、主表和周报。所有表格必须区分重复次数、结局覆盖和轨迹完整性；Luna 目前只有单次 test，不能标为 avg@3。
