# Langfuse 评测证据复盘

实验：c-study15contract1
版本：2564978d7769a41a2032d9b9c118e82b00c0126e8379306a943b4dd12d99b558

费用为历史运行时记录的估算，不是实际账单；价格版本未知。预算账本单独列示，失败尝试保留。
未知或不完整的费用不按零计算。每任务一次的小样本不能证明质量等价或统计显著性。

历史 Span 未记录调用角色，主调用与摘要的费用拆分未知；分别保留摘要 Token 与耗时，不能按 Token 比例分摊费用。

| 场景 | 条件 | 通过/总数 | 未缓存输入 | 缓存读取 | 输出 | 均耗时 ms | 估算 USD | 预算记账 CNY |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| overall | baseline | 14/15 | 75359 | 145792 | 7778 | 4565.224667 | 0.013136 | 0.248542 |
| overall | governed | 14/15 | 60495 | 92416 | 16665 | 8140.931333 | 0.013394 | 0.195553 |
| compression_degradation | baseline | 5/5 | 26859 | 51456 | 2515 | 4416.938000 | 0.004609 | 0.087512 |
| compression_degradation | governed | 5/5 | 13882 | 25984 | 2169 | 4312.498000 | 0.002624 | 0.046414 |
| cross_turn_dependency | baseline | 5/5 | 26735 | 52096 | 2875 | 4986.302000 | 0.004694 | 0.088810 |
| cross_turn_dependency | governed | 4/5 | 25044 | 25728 | 12085 | 16001.780000 | 0.006962 | 0.078689 |
| long_session | baseline | 4/5 | 21765 | 42240 | 2388 | 4292.434000 | 0.003834 | 0.072220 |
| long_session | governed | 5/5 | 21569 | 40704 | 2411 | 4108.516000 | 0.003809 | 0.070450 |

## 失败排查

判定来自独立产物检查，不代表已证明摘要导致失败。人工等待时间未采集。

### cross_turn_dependency-3 / governed
- 分类：artifact_mismatch, latest_decision_applied_failed
- 验证依据：["artifact_mismatch"]
- 产物检查：{"early_constraint_retained": true, "active_constraint_applied": true, "latest_decision_applied": false, "artifact_valid_json": true, "forbidden_paths_clean": true}
- 本地证据：['.cf/c-study15contract1/t-0958c2a02612']
- 排查顺序：manifest 原始历史 → 保存的摘要 → 最终模型输入 → 实际产物。缺失环节标 unknown。
- 下一步：核对关键决策在各环节的保留情况；固定数据与评分规则，每次只改一个因素再评测。

### long_session-3 / baseline
- 分类：artifact_mismatch, latest_decision_applied_failed
- 验证依据：["artifact_mismatch"]
- 产物检查：{"early_constraint_retained": true, "active_constraint_applied": true, "latest_decision_applied": false, "artifact_valid_json": true, "forbidden_paths_clean": true}
- 本地证据：['.cf/c-study15contract1/t-ae302c1e5c12']
- 排查顺序：manifest 原始历史 → 保存的摘要 → 最终模型输入 → 实际产物。缺失环节标 unknown。
- 下一步：核对关键决策在各环节的保留情况；固定数据与评分规则，每次只改一个因素再评测。
