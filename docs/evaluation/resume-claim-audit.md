# CodeFlow-harness 简历主张核对与补证计划

本文把简历项目中的工程主张与当前仓库可定位的源码、测试和实验记录对应起来。它用于避免把代码存在、单元测试通过和真实模型实验结果混为一谈。没有找到对应原始记录的数字保持“待补证”，不从相邻实验推导。

## 已能在仓库中定位

| 主张 | 当前证据 | 可支持的范围 | 仍需说明 |
| --- | --- | --- | --- |
| 多会话调度降低队头阻塞 | `docs/evaluation/runtime-scheduler-experiments.md` | 冻结的真实 Provider 对照包含 4,160 个 Turn；P95 中位数由 13,567.997 ms 降至 11,663.193 ms；Verifier 为 4,158/4,160（99.95%）。 | 所有任务通过的预注册门槛未通过，实验标为 `claim_eligible: false`；这是特定突发负载，不是生产 SLO。 |
| 定时任务多进程 claim | `codeflow/proactive_engine/schedulers/cron/service.py`、`tests/test_cron_service_claim.py` | 共享 jobs.json 的 sidecar 文件锁保护 claim 和持久化；新回归测试以两个独立 OS 进程竞争同一到期任务。 | claim 使用 30 分钟 TTL。超时后可接管，长任务可能重复触发；外部副作用需要幂等。不能将此描述成跨崩溃 exactly-once。 |
| 开发任务 Worktree 隔离 | `docs/examples/concurrency-and-worktrees.md`、`codeflow/agent/subagent/worktree.py` | 显式开发模式下为任务提供独立 Git 工作区及 Review / Verifier / Promote 流程。 | Worktree 是目录和版本隔离，不是 OS 安全沙箱。简历中 10 个场景、100% 的表述属于任务恢复，不能归到 Worktree 隔离结果。 |
| 上下文压缩与跨轮约束保持 | `docs/evaluation/context-management-fix.md` | 15 对探索性真实模型试验中，治理组 13/15 严格通过，输入减少数字按文档的指定子集统计；所有失败仍保留。 | 这是小样本探索性结果，存在两次严格 Verifier 失败；无法映射到简历所写 Judge +0.8、约束保留 +26%、Token -13.44% 的同一实验口径。 |
| 上下文阈值对比 | `benchmark/context_threshold_ablation/summary.md` | 文档保留正式试验中断原因和部分运行记录。 | 正式矩阵不完整，不能选出最佳阈值，也不能把部分样本作为完整对照结论。 |

## 尚待补证的简历数字

- Checkpoint / Resume 的 10 个场景、100% 漂移识别：仓库有恢复实现及专项测试，但本次审查没有找到按 10 个场景逐项汇总、可追溯到原始结果的报告。
- 安全评测 30 个场景重复 3 轮：仓库有沙箱和安全测试，但没有找到与 90 次运行对应的完整记录、策略快照和汇总脚本。测试代码存在不等于该数字已经复现。
- 8 路读取 185.31 ms → 51.50 ms：没有找到对应的基线、样本定义、重复次数和原始计时结果。
- SWE-bench 50 题对照按单独约定排除在本轮范围之外。

## 简历相关的相近功能及边界

| 相近名称 | 各自职责 | 结论 |
| --- | --- | --- |
| Spine Scheduler / CronService | Spine 管理进程内会话队列；CronService 管理持久化定时任务并用文件锁协调领取。 | 调度对象与存储边界不同，不应当合并或将 Cron claim 描述为 Spine 的跨进程能力。 |
| ContextBuilder / ContextAssembler | 前者保留共享资源、估算和兼容接口；后者通过分段构建器负责每轮实际上下文组装。 | 不是两个并行生产组装入口，旧接口仍有调用者，删除前需要迁移和验证。 |
| Worktree / Checkpoint | Worktree 隔离开发任务的 Git 工作目录；Checkpoint 使用独立 shadow Git 保存工作区快照。 | 分别服务并发隔离和恢复证据；Checkpoint 不保存对话状态，也不是完整崩溃恢复。 |
| ResourceLockManager / portable file lock | 前者协调单进程内工具对重叠路径的读写；后者通过相邻锁文件协调参与者跨进程写入。 | 锁范围不同，不能互相替换；外部不遵守锁的进程仍可修改文件。 |
| Direct / SRT / BoxLite 执行器 | 三者实现同一个执行接口，但隔离方式和能力不同。 | 是可配置后端，不是重复实现；安全测试应按实际选中的后端分别报告。 |

## 后续执行顺序

1. 给 Resume / 安全 / 读取并发三项补一个冻结清单：场景 ID、前置状态、动作、通过条件、平台、版本与配置摘要。先映射已有测试，避免重复造场景。
2. 对 Checkpoint / Resume 增加或整理部分写入、进程退出、外部工作区漂移、重复副作用和损坏状态场景；记录“恢复、拒绝恢复、人工介入”的明确结果。
3. 对安全场景记录策略版本、工具类型、工作区边界和每次拒绝 / 允许结果；把 SRT、BoxLite 与直接执行分开汇总，不合并成一个安全率。
4. 对读取并发固定文件大小、并发度、冷 / 热缓存、平台与重复次数，报告中位数及 P95，并校验写入 barrier 与锁语义。
5. 对 LLM Judge 指标保留 rubric、模型标识、原始评分和人工抽查样本；明确 5 分制绝对分变化与相对提升的计算公式。
6. 将以上结果写成带源版本、依赖摘要、命令、逐场景结果和汇总程序的报告。只把完整、可复算且门槛通过的数字放回简历。

## 本次变更

- 更新并发设计说明，区分进程内 Spine 队列和 CronService 的多进程 claim，并写清 TTL 过期接管的重复触发边界。
- 在 `tests/test_cron_service_claim.py` 增加双进程抢占测试。
- 其余未找到原始结果的 Resume、安全和性能数字列为待补证，不凭空生成实验数据或改写成已验证结果。
