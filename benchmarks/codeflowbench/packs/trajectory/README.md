# CodeFlowBench: 轨迹质量与上下文评测包 (`packs/trajectory/`)

## 1. 评测目标与设计原则

本评测包用于定量评估 CodeFlow-harness 在多轮 Agent 执行轨迹中的质量表现，重点评测：
1. **LLM Judge 质量评分 Harness**：按「工具调用合理性、错误恢复质量、约束遵守度」等维度，采用 5 分制量表对 Agent 执行轨迹打分；
2. **上下文治理三类典型场景**：
   - **长会话场景 (`long_session`)**：40-60 轮长对话交互，累积高输入 Token；
   - **跨轮次依赖场景 (`cross_turn_dependency`)**：早期声明全局约束，跨越 30+ 轮次后依然要求准确继承与应用；
   - **压缩降级场景 (`compression_degradation`)**：上下文逼近模型窗口上限时触发历史治理；
3. **对照策略**：
   - **基线对照组 (Control)**：阈值触发式尾部截断（`fifo_tail`），直接滑动丢弃早期上下文；
   - **治理组 (Treatment)**：CodeFlow 分层治理与上下文策展（`curator`），保护首尾关键槽位、动态提炼工作状态并修剪中间无用噪音。

---

## 2. 评测指标定义

| 指标 | 英文标识 | 含义 |
|---|---|---|
| 工具调用合理性 | `tool_rationality` | 工具选择是否合理、参数是否准确、是否存在重复调用或循环死锁 (1.0 - 5.0) |
| 错误恢复质量 | `error_recovery` | 工具报错后 Agent 是否具备错误归因能力并自适应切换备选策略 (1.0 - 5.0) |
| 关键约束保留率 | `key_constraint_retention_rate` | 跨轮次交互中早期全局约束、活跃规则与最新决策在最终输出中的保留比例 (0.0 - 1.0) |
| 输入 Token 统计 | `input_tokens` | 多轮推理累积喂给 LLM 的上下文 Token 总数 |
| Token 节省比例 | `token_reduction_pct` | 策展治理组相比于未治理对照组节省的输入 Token 比例 (%) |
| 综合评分 | `composite_judge_score` | LLM Judge 加权综合质量评分 (1.0 - 5.0) |

---

## 3. 运行方式

### 模式 A：免凭据离线评测与干跑校验 (Dry-Run / Offline Comparison)
完全无需外部 API Key，通过 `DeterministicJudge` 与 `DeterministicTrajectoryRunner`：
- 对 6 个代表性场景执行轨迹的结构特征进行客观度量；
- 对比计算 FIFO 与 Curator 策略下的 Token 消耗与约束保留情况；
- 明确标注 `is_synthetic=True`，杜绝伪造远程商业模型评分。

运行命令：
```bash
# 1. 运行单元测试验证评测包协议
pytest tests/test_codeflowbench_trajectory_pack.py -v

# 2. 重新计算上下文质量对比并输出报告
python benchmarks/codeflowbench/packs/trajectory/scripts/run_comparison.py
```

### 模式 B：正式全量 LLM Judge 运行 (Live LLM Judge)
正式运行需要配置 Judge 模型：
1. **环境配置**：
   - 导出 Judge 模型凭据，如 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY` 或 `DEEPSEEK_API_KEY`；
   - 在 `~/.codeflow/config.toml` 中配置默认 Judge 模型（推荐 `gpt-4o` 或 `claude-3-5-sonnet`）。
2. **运行命令**：
   ```bash
   codeflow benchmark run --pack trajectory --track formal --model gpt-4o
   ```

---

## 4. 评测结果证据存储
- 结构化数据：`docs/evaluation/trajectory-eval-results.json`
- 详细对比报告：`docs/evaluation/trajectory-eval-report.md`
