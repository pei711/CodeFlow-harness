# Langfuse 评测改进交付

已接入失败分类、独立验证细项、实验版本、缓存用量、费用估算与预算记账分离，以及按场景对比和失败排查线索。

原实验记录保留；导出历史证据没有重新调用模型。

## 查看结果

- 最新轮次：[study15contract2 报告](study15contract2-v1/REPORT.md)，两组各通过 15/15。
- 失败复盘：[study15contract1 报告](study15contract1-v1/REPORT.md)，两组各通过 14/15，保留两个最新决策未执行的案例。
- 两份报告旁的 `evidence.json` 包含全部结构化指标与版本摘要。
- [最新轮次 Langfuse 数据集](https://us.cloud.langfuse.com/project/cmvxlhwre00c5ad0k5d5c3nwb/datasets/cmvxtlh1n00hgad0kc6ujvr3x)。
- [失败案例 Langfuse 数据集](https://us.cloud.langfuse.com/project/cmvxlhwre00c5ad0k5d5c3nwb/datasets/cmvxtqgle00kcad0kgodxg372)。

两轮共 60 条 Trace 均已通过远端回读验证：观察记录 ID、评分 ID 与值、导出版本元数据、输入 Token 总量。

最新轮次的历史运行时费用估算：baseline 约 0.012935 USD、governed 约 0.012304 USD；平均耗时约 4.29 秒和 7.91 秒。跨轮依赖场景治理组的费用估算及耗时更高，不能用整体输入减少代表所有场景都获益。

历史价格版本、实际账单、人工等待时间及主调用/摘要的费用拆分仍未知。预算账本与运行时估算不是实际账单；失败分类不是已证明的因果关系。每任务一次的小样本不能证明质量等价或统计显著性。

## 验证

11 项适配器与证据边界检查通过，主环境的 2 项无 OpenTelemetry 检查通过。两轮各 30 条本地记录的输入和输出用量均与历史模型 Span 对账一致。源码编译检查通过。

使用方式和数据口径见 [langfuse-runtime.md](../langfuse-runtime.md)。
