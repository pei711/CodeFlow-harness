# Context 任务的公开输出契约

上一轮治理组严格 Verifier 为 13/15。两个失败产物的必需字段值均正确，但分别
多写了 `deployment_window_utc` 和 `recorded_at`。任务原文要求 “Include ...”，
独立 Verifier 实际采用完整 JSON 相等，因此额外字段也会失败。

现在 `ContextTask.output_schema` 从任务已声明的 `constraint_keys` 和
`decision_keys` 生成公开契约：

```json
{
  "type": "object",
  "properties": {"approver": {}, "day": {}, "release_id": {}, "timezone": {}},
  "required": ["approver", "day", "release_id", "timezone"],
  "additionalProperties": false
}
```

`RuntimeContextTrialRunner` 将契约原样附到两组最终请求，要求 Agent 自行读回
产物，检查完整键集合。Schema 不读取 `expected_path`；字段类型和枚举只来自
任务的公开 `output_properties`，正确选项仍需从历史推导。契约同时写入 task spec 和
实验 manifest，并随数据集上传，保证可复核实际给了模型哪些信息。

这次没有增加自动删字段、自动改产物、Verifier 答案反馈或按失败追加运行。
独立 Verifier、禁止路径检查、预期答案文件保持不变。若模型仍生成额外字段
或错误值，任务仍会失败。获得单批次 100% 也不代表任意新任务都能达到 100%。

本改动修正的是任务输出协议，不能当作新的上下文压缩算法收益。
新实验与上一轮相比提示词更明确；只有新一轮内部两组使用同一契约，适合做
成对比较。旧的 13/15 和更早失败记录保留，不覆盖或重新评分。

回归覆盖：两组请求文本一致、Schema 不访问预期文件、不泄露预期答案值，
以及额外字段仍被原 Verifier 判失败。

字段契约 v1 的工件目录为 `.codeflow/lc/study15contract1/`，两组均为 14/15；
失败产物把 strategy 写成了 `blue-green migration`，而严格协议要求 `blue-green`。
全部 30 条记录和对应源码快照保留。

v2 在迁移任务的公开定义中声明 `strategy` 是字符串枚举
`["rolling", "blue-green"]`。旧决策和新决策都是合法 Schema 值，因此该约束
不指定当前正确答案。回归额外确认：模型若选已被替代的 rolling，仍会被独立
Verifier 拒绝；修改返回 Schema 也不会污染任务原始定义。相关回归共 137 项。

v2 工件目录为 `.codeflow/lc/study15contract2/`。以该目录中的 `summary.json`、
`REPORT.md`、逐次记录及后续证据审计为实际结果，不预设通过率或 Token 降幅。
这是输出协议的两次明确修订，旧轮次结果保留，不混合计算通过率。

## v2 真实复测结果（2026-11-01）

新一轮完整执行 15 对、30 次，两个条件均为 15/15 严格 Verifier 通过。
使用相同历史、模型、主 Agent 500 Token 和摘要 2048 Token 预算，独立 Verifier
及预期答案不变。没有增加自动修剪字段或重新运行失败任务的路径。

| 场景 | 完整历史对照 | 治理组 | 输入 Token 减少 | 平均耗时：对照 → 治理 |
|---|---:|---:|---:|---:|
| 长会话 | 5/5 | 5/5 | 2.97% | 4.56 → 4.08 秒 |
| 跨轮次依赖 | 5/5 | 5/5 | 34.40% | 4.08 → 15.66 秒 |
| 压缩降级（故障注入） | 5/5 | 5/5 | 48.91% | 4.24 → 4.00 秒 |
| 合计 | 15/15 | 15/15 | 30.41% | 4.29 → 7.91 秒 |

实际输入为 220413 → 153395，包含缓存输入和 11192 个摘要输入 Token。
95 次 Provider 请求均结算，输入合计 373808、输出合计 24032 Token；与每次
运行记录和原始模型 trace 汇总一致。137 项相关测试通过。

本轮 100% 仅适用于这组公开契约下的固定任务集，不等同生产可靠性。
跨轮次组仍有 4 次摘要达到上限后进入 fallback，但原始约束保留机制使任务
通过。平均治理耗时更长，因此不据此宣称提速。

代码运行期间，报告脚本有证据元数据和诊断输出更新。已按启动 manifest 的
SHA-256 保存逐字节一致的源码快照；Provider 设置及整个 Trial 执行循环的 AST
与当前脚本一致，变化只在证据构建和报告部分。详见本轮 evidence-audit.json。
