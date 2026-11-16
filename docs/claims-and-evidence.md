# CodeFlow-harness 简历主张与证据状态索引 (Claims and Evidence)

本文档逐项对照个人简历中针对 **CodeFlow-harness** 项目提出的量化技术主张与当前开源仓库中的工程证据状态。

遵循严谨工程原则，所有主张均如实标注其证据状态：
- **`仓库内可复现`**：仓库内包含完整测试命令与自动化流水线，任何人克隆本仓库即可本地复现，产物已入库。
- **`内部实验记录，未入库`**：主张基于真实开展的工程评测与实验数据，但原始庞大数据集、带付费 API Key 的评测日志、敏感环境快照或离线运行产物未随开源仓库公开发布。

---

## 简历量化主张对照表

| # | 简历量化主张 | 证据状态 | 核心数字与指标 | 仓库对应文件 / 复现命令 / 状态说明 |
|---|---|:---:|---|---|
| 1 | **多会话调度降低队头阻塞 (P95)** | **`内部实验记录，未入库`** | P95 13.57s → 11.66s (-13.53%) | 评测方案与统计方法见 [docs/evaluation/runtime-scheduler-experiments.md](evaluation/runtime-scheduler-experiments.md)。基于 commit `00fd757f` 跑完 80 组配对重复、4,160 个真实 Agent Turn。离线复验命令为 `CODEFLOW_LIVE_PERF_EVIDENCE=<file> make codeflowbench-runtime-live-verify`。因数据量极大且包含 Provider 会话记录，原始 4,160 条 Turn 产物未入库 Git。 |
| 2 | **任务自动验收通过率** | **`内部实验记录，未入库`** | 自动验收 99.95% (4,158 / 4,160) | 记录于 [docs/evaluation/runtime-scheduler-experiments.md](evaluation/runtime-scheduler-experiments.md) 第 122 行。4,160 次真实 Turn 中 4,158 次通过 Verifier 验证；2 次因模型回复遗漏 marker（`marker_missing`）失败。全量 4,160 次请求无任何调度器丢失或异常。因预注册门禁要求 100% 全通，基准测试标为 `claim_eligible: false`，原始记录未入库。 |
| 3 | **并发读取性能优化** | **`内部实验记录，未入库`** | 8 路读取 185.31ms → 51.50ms | 仓库内包含单进程资源锁协调器与便携跨进程文件锁实现（`codeflow/agent/tools/`），但针对特定 8 路并发读取实验的测试脚本、冷热缓存测量及原始耗时日志属于内部性能评测记录，未入库。 |
| 4 | **Token 治理与缓存降耗** | **`内部实验记录，未入库`** | Token 消耗降低 -12% ~ -13.44% | 参见 [docs/evaluation/tokenwise-cost.md](evaluation/tokenwise-cost.md)。验证了稳定 Prefix 使 DeepSeek 自动缓存命中率达 74.05%、单任务成本下降 72.3%；历史探索性实验中观察到输入 Token 的降幅。具体 -12% / -13.44% 的多任务基准对比记录属于内部实验归档，未入库。 |
| 5 | **SWE-bench 解决率提升** | **`内部实验记录，未入库`** | SWE-bench 32% → 46% | 仓库包含 SWE-bench 适配器与轨迹压缩实现（`codeflow/evolver/compressor/trajectory.py`、`codeflow/evolver/orchestrator/`）。50 题对比实验涉及高额模型调用及长时间环境执行，原始生成的补丁及评测详细日志属于内部实验记录，未入库。 |
| 6 | **LLM Judge 综合评分** | **`内部实验记录，未入库`** | LLM Judge 评分绝对分 +0.8 | 仓库内置 LLM Judge 评分框架、Rubric 与提示词模板（`codeflow/evolver/judge/prompts.py`、`codeflow/evolver/judge/schema.py`）。+0.8 分对应的评测样本打分记录、专家盲测比对日志属于内部实验记录，未入库。 |
| 7 | **沙箱命令隔离安全评测** | **`内部实验记录，未入库`** | 安全评测 90 次 (30 场景 × 3 轮) | 仓库实现了 Direct / SRT / BoxLite 三种隔离后端及完备单测（`codeflow/sandbox/`）。30 个危险命令场景各 3 轮共 90 次安全评测的专项汇总报告属于内部测试记录，未入库。 |
| 8 | **故障恢复与漂移识别** | **`内部实验记录，未入库`** | 10 个恢复场景 / 漂移识别 100% | 仓库包含 Checkpoint 快照与 Shadow Git 工作区恢复机制（`codeflow/agent/loop/checkpoint.py`）。涵盖部分写入、进程崩溃、外部文件漂移等 10 个测试场景与 100% 漂移识别的独立汇总矩阵属于内部专项评测记录，未入库。 |

---

## 仓库内已可独立复现的门禁 (Deterministic Benchmark Gate)

除上述需要大量真实外部模型调用或特定环境的内部实验外，仓库自带完整的**本地免凭据冒烟门禁（CodeFlowBench Smoke Gate）**，任何克隆本仓库的用户均可直接在本地一键运行并验证：

- **验证命令**：`make codeflowbench-smoke`
- **已入库验证报告**：[docs/evidence/REPORT.md](evidence/REPORT.md)
- **已入库原始证据**：[docs/evidence/report.json](evidence/report.json)（Evidence Digest: `11527da52c80b3e9d5e17fe275f284e3b3bffd05685a997686b39aceee13fdfb`）
- **覆盖轨道**：
  1. **MCP Smoke**：64 个 MCP 工具注册与 `catalog_probe` 调用，验签闭环；
  2. **R0 Scheduler 轨道**：2,000 次请求、64 会话，验证 0 lost requests、0 lifecycle contradictions，P95 分发开销 0.56ms，P95 队列等待 601.33ms；
  3. **R1 Runtime 轨道**：100-Turn `AgentTurnRunner` 全流程闭环测试，0 状态冲突，资源正常释放；
  4. **Report Rebuild**：确定性产物与摘要重算校验。
