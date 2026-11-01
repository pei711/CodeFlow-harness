# 并发与开发工作区改造

## 改造前审计（2026-10-25）

实际 Git 根是当前目录。仓库原有的 context budget 等未提交改动不属于本次修改。

| 模块 | 已实现 | 缺口 / 本次范围 |
| --- | --- | --- |
| `spine/scheduler.py` | asyncio；按 conversation 的 Lane；APPEND/INJECT/INTERRUPT；worker 独占 Future 完成权；shield；取消与四阶段 shutdown | 保留机制；验证 4/2 上限、顺序、取消、异常；拒绝零容量池 |
| `spine/runner.py`、`turn.py` | Runner 协议、不可变请求、TurnOutcome | 无需重构；Turn 不等于线程、进程或工作区 |
| Gateway / REPL / TUI | Gateway 默认 USER=4、SYSTEM=2；交互入口此前为 1/1 | 统一入口默认 4/2，仍允许显式配置 |
| `agent/tools/registry.py` | schema、转换、timeout、错误归一化；连续安全 READ 并行，写入是批内 barrier | 保留；跨 registry 共用资源锁 |
| `agent/tools/resource_locks.py` | 同路径及父子目录重叠；共享读、独占写；写者公平；等待取消清理 | 补自动化回归；不是 OS 锁或外部写入保护 |
| 文件 / 搜索工具 | workspace 路径解析和权限检查 | 同步 I/O 阻塞事件循环；整文件覆盖没有旧版本检查；原地写可能留下半文件 |
| Shell / checkpoint | asyncio 子进程 | Shell 和 checkpoint 取消时需回收直接子进程；共享 checkpoint 的 add/diff/commit 事务需串行 |
| AgentLoop 共享资源 | executor 启动锁、MCP connecting 标志 | Turn 错误会关闭其他 Turn 的 executor；后来的 Turn 会跳过尚未完成的 MCP 初始化。本次增加使用计数和初始化等待 |
| SubagentManager | asyncio task、独立 registry/executor、独立 gate、限频、结果重注入 | 所有开发任务共享代码目录；完成映射被清理，不保留工作区状态 |
| Session / 共享存储 | `utils/atomic_io.py`、`portable_lock.py` 已使用 portalocker、epoch、读检查写事务；Gateway 单实例保护 | 没有跨进程任务领取队列，不新增通用 claim 锁或租约 |

必须修复：文件 I/O、取消期间锁的生命周期、版本校验与原子写、checkpoint 事务和子进程取消清理。

建议且本次实现：显式 `spawn(..., development=True)` 的隔离开发工作区，以及单独的 Review → 临时集成 → Verifier → 目标引用提升 API。普通 spawn/聊天不创建 worktree。不猜测任务文本是否需要独立开发。

暂不实现：分布式调度、持久化任务领取、租约/重试/幂等、进程崩溃自动恢复、模型自动审代码/解决冲突、自动提交全部修改。现有子 Agent 的自然语言完成状态不是代码验收证据。

## 验证与边界

测试和实际结果在本文件末尾记录。原有 R0（每次 2,000 请求，默认 5 次合计 10,000）、P95 调度实验保留；其指标独立于文件锁和 Worktree。运行期间仓库含未提交改动，测量只作为本地回归，不作为发布性能声明。

## 实现与使用

### 文件一致性

`ResourceLockManager` 仍只协调本进程里共用管理器的调用。读锁可共享；同文件、父子目录的读写和写写互斥；不同目录的写入不互相阻塞。写者排在后续重叠读者之前，取消的等待者不会继续挡住读者，重复取消也必须完成释放。

`read_file` 返回原始字节的 SHA256（分页读取也返回整文件版本）。`write_file` 和 `edit_file` 增加可选 `expected_sha256`。写入时在同一独占锁内读取当前版本、比较、计算变更并原子替换；不匹配返回 failed ToolResult，要求重新读取。新文件可传 `missing`，防止覆盖别人刚创建的文件。

```json
{"path": "src/example.py", "content": "new content", "expected_sha256": "<read_file 返回的 SHA256>"}
```

保留旧调用兼容性：不传版本的整文件覆盖仍采用串行的最后一次写入生效语义，不能防止调用方基于陈旧读取覆盖内容。共享开发应传版本，或用独立 Worktree。Edit 还保留原有 old_text 检查。

文件读写、目录扫描、Python 搜索回退在 asyncio 默认线程池中执行。线程池容量与 USER/SYSTEM 池独立，不为每个 Agent 建线程。不能强制终止已启动的 Python 文件 I/O，因此 timeout / cancel 请求会等待正在执行的 I/O 收尾，再释放路径锁；它不是硬实时期限。临时文件与目标同目录，写完 fsync 后 os.replace，失败清理临时文件并保留原文件。保留已有文件的 mode 和 Edit 的 CRLF；不保证 ACL、扩展属性、硬链接关系或断电后的目录元数据持久性。

Shell 继续使用 asyncio 子进程，不假装能解析任意命令的读写集合，因此不套文件工具的路径锁。Direct/SRT 为命令建立可终止进程组；POSIX 用 killpg，Windows 用 taskkill /T /F，随后回收直接进程。外部独立进程、脱离进程树的守护进程和远程副作用仍超出这个边界。

### 开发工作区

主 Agent 通过现有 `spawn` 工具显式传入 `development: true`，或调用 `SubagentManager.spawn(..., development=True)`。普通 spawn 不创建 Worktree。工作区必须是 Git 仓库根并有有效的已提交 HEAD；失败会作为子任务失败回报，不回退到共享目录写代码。

Worktree 位于仓库旁的 `.<仓库目录名>.codeflow-worktrees/<task_id>`，分支为 `codeflow/task/<task_id>-<随机后缀>`。放在仓库外，避免主工作区的 `git add -A` 意外加入嵌套仓库。主工作区未提交修改不复制、不暂存、不丢弃。每个任务复用自己的工作区，多个 Worktree 仍由一个 Python 进程管理。

ContextVar 为子任务绑定工作区，文件路径、allowed_dir、Shell cwd、executor、提示词使用同一路径；不修改共享的 `self.workspace`。文件工具限制在该目录内，Shell 仍受原有配置的 sandbox 和路径守卫控制。Worktree 是版本与目录隔离，不是安全沙箱。SRT 默认禁止 Git 元数据写入，Boxlite 可能无法访问工作区外的 Git common dir；本次不放宽这些策略。不能提交时保留修改并报告 pending commit，不能宣称完成整合。

开发状态记录默认位于 `.codeflow/development_tasks/<task_id>.json`；若 Runtime 提供独立 state root，则位于其 `development_tasks/`。记录包含路径、分支、base、HEAD、dirty、执行状态、错误；集成记录还包含目标引用和已 Review 的提交。完成、异常和取消都不自动删除分支或工作区。进程被强制终止时，可能留下 preparing/running；这些是需要人工检查的记录，不是自动恢复的任务租约。

### Review 与整合

当前 Runtime 没有可直接复用的通用编译/测试 Verifier 服务。本次提供必须注入的异步 `Verifier(Path) -> bool` 接口；主 Agent / Host 应在其中运行本项目适合的编译、测试和约束检查。接口只接受显式 True，不把模型文字或任意 truthy 对象当作验收。

整合 API 位于 `subagents.worktrees`，没有暴露给子 Agent 的自动合并工具：

1. 主 Agent 拆分任务并决定哪些任务需要独立开发目录。
2. 子 Agent 修改、检查 Diff、按任务文件提交自己的功能分支；提交不是无条件自动 `git add -A`。
3. 主 Agent 调用 `review(task_id)` 获取固定 HEAD 与 Diff，检查任务约束和改动，批准后才传给 integrate。dirty 或已切换分支的工作区不能通过 Review。
4. `integrate(approved_reviews, target="目标分支", verifier=...)` 再检查 Review 是否过期，在新的临时集成分支上合并固定提交。任何冲突都保留 MERGE_HEAD 和文件现场，并通过 WorktreeError 交回主 Agent / 人工处理，不自动解决。
5. 合并后调用 Verifier。失败、异常或取消只记录状态，目标引用不变。Verifier 若改动 HEAD 或留下未提交文件也拒绝候选。
6. 验收后得到候选。再次 Review 有效性由 Host 负责；只有显式 `promote(candidate)` 才允许执行最后的 Git 引用比较并交换（CAS）。候选被改动或目标已前进会拒绝。
7. 已在任意 Worktree 检出的目标分支一律拒绝自动提升，保留候选，交给主 Agent / 用户在该目标工作区显式整合，避免使真实 index / 工作区与分支引用不同步。短 CAS 发布一旦开始，会在响应取消前完成状态记录。

API 的 approved_reviews 是受信任协调者传入的批准结果，不是自动审代码或授权系统。主 Agent 的 Review 质量与 Verifier 覆盖面仍决定业务正确性；文本无冲突不代表没有逻辑冲突。

### 共享状态边界

不新增跨进程任务领取锁：Scheduler / Lane 的队列和 Future 都在本进程内，SubagentManager 也没有磁盘任务 claim API。已有 Session 的跨进程 portalocker / epoch / 原子写事务继续保留。Checkpoint 新增的是单个 AgentLoop 持有的 CheckpointService 的事务锁；Git 自身的 index.lock 不等于 add/diff/commit 事务锁。

同时启动多个独立 Runtime 写同一代码目录、同一 shadow.git 不在新增保证内，应使用独立工作区和 state root。Worktree 管理与提升 API 也要求单个 Host 协调，不支持外部 Git 管理命令同时切换目标分支。若未来增加多消费者任务队列，应单独设计持久化领取事务、租约、超时回收、幂等和崩溃恢复，并用两个独立进程验证；本次不声称跨进程任务 exactly-once。

## 修改文件索引

| 目的 | 文件 |
| --- | --- |
| 保留调度并校验容量；统一交互默认 4/2 | `codeflow/spine/scheduler.py`、`codeflow/cli/_repl_spine.py`、`codeflow/tui_rpc/spine.py` |
| 共享资源生命周期与 checkpoint 事务 | `codeflow/agent/loop/main.py`、`codeflow/agent/loop/checkpoint.py` |
| 文件并发、版本检查、线程池与原子替换 | `codeflow/agent/tools/resource_locks.py`、`filesystem.py`、`file_search.py`；`codeflow/utils/async_io.py`、`atomic_file.py` |
| 显式开发模式与整合 API | `codeflow/agent/subagent/worktree.py`、`manager.py`；`codeflow/agent/tools/spawn.py` |
| 子进程取消与 Windows / Linux 路径边界 | `codeflow/utils/subprocess.py`；`codeflow/sandbox/direct_executor.py`、`srt_executor.py`、`boxlite_executor.py`、`config.py` |
| 新增回归 | `tests/test_concurrency_safety.py`、`test_development_worktrees.py`、`test_agent_shared_lifecycle.py`、`test_subprocess_cancellation.py` |
| 更新已有预期与平台可执行性 | `tests/test_cli_repl_spine.py`、`test_tui_rpc_spine.py`、`test_sandbox_unit.py`、`test_runtime_checkpoint_bug2_deep.py`、`test_atomic_io.py`、`test_codeflowbench_runtime_experiments.py` |
| 原有评测的源码对照入口 | `scripts/check_concurrency_performance.py` |

## 本地调度性能记录

命令：`uv run --no-sync python -X utf8 scripts/check_concurrency_performance.py`。

结果保存在 `.codeflow/evidence/concurrency-validation/20261025T084319404589Z/` 的 `baseline.json`、`current.json`、`comparison.json`。对照只替换 HEAD 中的 Scheduler 源码，共用当前 benchmark 与依赖，运行时关闭 tracing；不是两个完整干净 checkout 的发布对照。

| 指标 | HEAD 调度源码 | 当前调度源码 |
| --- | ---: | ---: |
| R0 接受请求数 | 10,000 | 10,000 |
| 丢失 / 重复 / 未完成 Handle | 0 / 0 / 0 | 0 / 0 / 0 |
| P95 Lane 相对全局 FIFO 降幅（成对重复的中位数） | 52.83% | 55.35% |
| 后台饱和时共享池 P95 / 空闲 P95 | 5.41 倍 | 4.76 倍 |
| 后台饱和时隔离池 P95 / 空闲 P95 | 1.05 倍 | 1.51 倍 |

当前 HOL 的汇总 P95 为 FIFO 30.6229 ms、Lane 14.1232 ms。所有原有正确性与性能方向门禁通过。比例存在计时波动，隔离池倍率本次也变高，不能据此声称改造带来新的性能提升或证明没有任何退化。Worktree、文件锁、线程 I/O 不在这些延时 Runner 的测量路径中，其测试结果单独记录。

原有小配置性能单元测试的 1/3 ms 延迟在 Windows 粗粒度定时器上出现方向翻转；将单元测试的因果间隔提高为 20/80/160 ms，保留原来的方向断言。正式评测默认参数与 10,000 请求配置没有更改。

## 自动化验收命令

下面的 PowerShell 命令只运行本地测试，不调用模型或外部账号服务：

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
$modules = @(
  'tests/test_agent_shared_lifecycle.py',
  'tests/test_sandbox_unit.py',
  'tests/test_subagent_manager.py',
  'tests/test_spine_scheduler.py',
  'tests/test_spine_scheduler_lane.py',
  'tests/test_spine_scheduler_pools.py',
  'tests/test_tool_registry_execution.py',
  'tests/test_tool_registry_timeout.py',
  'tests/test_search_tools.py',
  'tests/test_concurrency_safety.py',
  'tests/test_development_worktrees.py',
  'tests/test_subprocess_cancellation.py',
  'tests/test_runtime_checkpoint_bug2.py',
  'tests/test_runtime_checkpoint_bug2_deep.py',
  'tests/test_cli_repl_spine.py',
  'tests/test_cli_gateway_spine.py',
  'tests/test_tui_rpc_spine.py',
  'tests/test_spine_runner.py',
  'tests/test_atomic_io.py',
  'tests/test_session_manager.py',
  'tests/test_cli_gateway_lock.py',
  'tests/test_codeflowbench_runtime_experiments.py',
  'tests/test_codeflowbench_runtime_track.py',
  'tests/test_agent_loop_run_emit.py',
  'tests/test_mcp_tools.py'
)
uv run --no-sync python -X utf8 -m pytest @modules -q --basetemp=.pytest-tmp/concurrency-repeat --junitxml=.codeflow/evidence/concurrency-validation/repeat.xml
```

开发环境是 Python 3.12 / Windows。本次执行沙箱无法创建 asyncio 初始化所需的本机 socketpair，改在沙箱外运行这些本地测试；pytest 临时目录和 uv 缓存放在工作区，使用 UTF-8 模式。不能把环境初始化挂起算成产品功能测试通过。

新增四个测试模块共 34 个用例，分别验证：Lane 提交顺序、跨会话推进、4/2 额度、Future 单次完成、取消和异常后的名额释放；真实线程并行读、带旧版本的写入冲突、CRLF 编辑合并、取消收尾与重复取消释放；checkpoint 事务；真实 Git Worktree 的目录/分支隔离、脏目录保留、Review 过期、合并冲突、真实编译/测试、验证失败不更新目标、目标前进/被检出时拒绝提升、集成和子任务取消；共享 executor/MCP 的并发初始化、失败与重连；Shell 直接进程回收及真实子进程取消后不能继续延迟写文件。

跨进程任务领取、进程被杀后的任务恢复测试不适用：本次没有增加这两种机制。既有 Session / Atomic I/O / Gateway 文件锁测试已包含在验收范围中。真实 Linux、SRT OS 沙箱和 Boxlite VM 没有在本机启动验证；SRT 的取消路径通过 mock 验证，不能据此声称在真实 VM 或生产环境验收完成。

## 最终结果（2026-10-25）

| 验收项 | 实际结果 |
| --- | --- |
| 上述 25 个相关模块 | 517 passed、2 skipped、0 failed；65.88 秒 |
| 新增四个并发 / Worktree 测试模块 | 34 个用例全部通过，包含真实 Git 和真实 Shell 子进程测试 |
| 既有 R0 原配置 | 对照与当前各 10,000 请求，全部正确性门禁通过 |
| 既有 P95 原配置 | 对照与当前均通过原有性能方向门禁；Lane P95 从 14.0715 到 14.1232 ms，不声称新增性能收益 |
| Ruff lint / format | 本次修改的 29 个 Python 文件全部通过 |
| Git diff --check | 通过 |

最终机器可读结果：`.codeflow/evidence/concurrency-validation/final-acceptance.xml`。两项跳过是当前 Windows 账号没有 symlink 特权、Windows chmod 无法表达 POSIX executable bit；不是静默忽略功能失败。符号链接用例在具有权限的平台仍执行，可执行位用例在非 Windows 平台仍执行。

验收期间曾出现的默认 pool 预期、Windows 路径映射、VM 路径校验、测试 Shell 语法、目录读取异常类型以及小配置定时器问题均已处理，最终结果没有残留失败。原有用户未提交的 context-budget 改动未纳入本次实现，也没有以本次测试结果声称那部分功能已被全面验收。
