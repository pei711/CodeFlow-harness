# CodeFlowBench: SWE-bench Verified 回归评测包 (`packs/swebench/`)

## 1. 评测目标与背景

本评测包基于权威软件工程基准 **SWE-bench Verified**，为 CodeFlow-harness 提供真实可运行的代码缺陷修复回归评测。
对比的核心治理策略为：
- **基线策略 (Control)**：阈值触发式滑动窗口/尾部截断（`fifo_tail`）。当会话接近上下文限制时，简单舍弃早期历史或暴力压缩。
- **治理策略 (Treatment)**：CodeFlow 分层治理与上下文策展（`curator`）。通过语义分段、受保护的早期约束锚定、元信息提炼，实现高保留率与低输入 Token 消耗。

评测指标包括：
1. **解决率 (Resolution Rate)**：补丁是否使全部 `FAIL_TO_PASS` 测试通过，且未引起任何 `PASS_TO_PASS` 测试回归。
2. **输入 Token 统计 (Input Tokens)**：各策略在多轮推理中喂给 LLM 的输入 Token 总数与节省比例。
3. **补丁统计 (Patch Statistics)**：生成的 Unified Diff 语法合法性、行变更量（`+added/-removed`）。

---

## 2. 冻结的 50 个 SWE-bench Verified 任务清单

清单文件存储在 `benchmarks/codeflowbench/tasks/swebench/manifest/`（单一事实源）：
- `meta.json`（清单元数据：schema / 版本 / 选取说明）
- `tasks-01.json` … `tasks-05.json`（50 个任务分片，每片 10 个，顺序合并即完整清单）

正式轨道用全部 50 个任务，冒烟轨道用清单前 2 个任务，均由 `tasks.py` 从清单直接派生，不再另存副本。

### 任务库分布 (覆盖官方 Verified 500 集全部 12 个仓库)
- `django/django`: 11 任务
- `sympy/sympy`: 8 任务
- `sphinx-doc/sphinx`: 5 任务
- `matplotlib/matplotlib`: 5 任务
- `scikit-learn/scikit-learn`: 5 任务
- `astropy/astropy`: 4 任务
- `pydata/xarray`: 4 任务
- `pytest-dev/pytest`: 3 任务
- `pylint-dev/pylint`: 2 任务
- `psf/requests`: 1 任务
- `pallets/flask`: 1 任务
- `mwaskom/seaborn`: 1 任务
**总计：50 任务**

每个任务冻结字段：
- `instance_id`：SWE-bench 唯一任务标识
- `repo`：目标仓库路径
- `base_commit`：基线提交哈希
- `version`：软件版本
- `fail_to_pass`：缺陷暴露必须修复的测试列表
- `pass_to_pass`：不得破坏的既有测试列表
- `problem_statement`：真实 Issue 描述
- `gold_patch_sha256`：官方参考修复补丁哈希
- `gold_patch_lines`：官方补丁行数

---

## 3. 数据集获取方式

大体积数据文件（>5MB）严格遵守红线**不入库**，仅冻结清单与获取脚本。

### 获取脚本
运行内置获取脚本：
```bash
# 1. 仅校验本地清单有效性（无需外部网络）
python -m benchmarks.codeflowbench.packs.swebench.scripts.fetch_dataset --verify-only

# 2. 从官方/镜像端获取全量数据或指定任务数据
python -m benchmarks.codeflowbench.packs.swebench.scripts.fetch_dataset --output /tmp/swebench_verified.json
```

支持两种下载方式：
1. **HuggingFace `datasets` 库**：若环境中已安装 `datasets`，脚本自动调用 `load_dataset("princeton-nlp/SWE-bench_Verified")`。
2. **Datasets Server API**：免安装额外依赖，通过官方 API 分批次拉取。

---

## 4. 评测运行模式

评测包提供双轨执行架构：

### 模式 A：免凭据离线干跑与冒烟 (Dry-Run / Smoke)
在缺少远程商业模型 API Key 或隔离 Docker 环境的开发与 CI 环境中，使用 `DeterministicSwebenchRunner` 执行：
- 校验 50 任务清单完整性与哈希一致性；
- 模拟生成 Unified Diff 补丁并走通 `SwebenchVerifier`；
- 基于 CodeFlow 的上下文规则评估 FIFO 与 Curator 策略的输入 Token 消耗与约束保留情况；
- 真实生成结构化评测结果对象（`TrialExecution`），不编造远程模型执行分数。

运行方式：
```bash
pytest tests/test_codeflowbench_swebench_pack.py -v
```

### 模式 B：正式全量评测 (Live Docker & LLM Evaluation)
正式执行需要配置以下运行环境：
1. **模型凭据**：
   - 导出对应 Provider 凭据（如 `DEEPSEEK_API_KEY`、`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`）；
   - 或在 `~/.codeflow/config.toml` 中配置 Provider。
2. **环境依赖**：
   - Docker 服务（用于拉取 SWE-bench 官方镜像 `swebench/sweb.eval.x86_64.*` 并在隔离容器内运行 `pytest` 测试）；
   - 本地 `git` 与网络（克隆对应 commit 的目标源码）。
3. **执行命令**：
   ```bash
   codeflow benchmark run --pack swebench --track formal --model deepseek/deepseek-chat
   ```
