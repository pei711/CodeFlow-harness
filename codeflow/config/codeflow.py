"""CodeFlow Feature Configuration，在 Base `Config` 上组合独立 Feature Blocks。

Usage：

    from codeflow.config import CodeFlowConfig, load_codeflow_config

    cfg = load_codeflow_config()
    if cfg.context.engine == "curator":
        ...

Design 上，``CodeFlowConfig`` Compose Base ``Config`` 而非 Subclass，使 Base Schema 不变，Feature Blocks 可
增删而不破坏 Base Loader。每个 Feature Block 有独立 Pydantic Model；Behavior-changing Path 默认 Off 或
Observe-only，Fresh Install 不会自动 Rewrite Requests。类型化加载成功不证明 Feature External Dependency
可用。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_camel

from codeflow.config.loader import (
    EXTENSION_KEYS,
    _migrate_config,
    get_config_path,
)
from codeflow.config.loader import load_config as load_base_config
from codeflow.config.schema import Config as BaseConfig


class _Base(BaseModel):
    """接受 ``camelCase`` 与 ``snake_case`` Keys 的 Feature Config Base。

        ``extra='forbid'`` 在 Startup 捕获 Typos。Known Legacy Retired Fields 在 Pydantic Validation 前由
    ``loader._migrate_config`` 显式 Strip；未列出的 Unknown Keys 仍 Raise，避免配置静默不生效。
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


# ---------------------------------------------------------------------------
# 功能 1：上下文管理（Curator）
# ---------------------------------------------------------------------------


class ContextConfig(_Base):
    """Context Assembly 与 Curator History Path 的 Selection/Tuning。

    当前 Runtime 只有统一 `ContextAssembler`；保留 `engine` 仅为旧配置兼容。其余参数控制 Fast/Slow Path、
    Relevance、Protected Head 与 Archive Location，不直接改变 Provider Context Window。
    """

    engine: str = "unified"
    """Deprecated：现在只有一个 :class:`ContextAssembler`。

    Historical ``"legacy"`` / ``"curator"`` / ``"default"`` Split 已合并；每 Turn 在同一 Engine 运行
    Curator History、Memory、Local Skill Lanes。Field 保留为 Free String，使旧 YAML ``engine: legacy`` 等
    仍可加载；``build_context_engine`` 忽略其 Value。
    """

    # Curator 历史路径参数。
    fast_path_threshold: float = 0.60
    """Curator Fast Path Cutoff。低于 Budget 此比例时 Zero-LLM Pass-through。"""

    curator_model: str = "gemini-2.5-flash"
    """Curator Agent Loop Slow Path 使用的 Model，默认 Small & Fast；配置不验证 Provider 是否支持。"""

    curator_timeout_seconds: float = 30.0
    """一次 Curator Slow-path Invocation 在 Fallback 前允许的 Max Wall Time。"""

    relevance_decay: float = 0.95
    """Non-recent Message Relevance 的 Per-turn Decay Factor。"""

    relevance_reference_boost: float = 0.15
    """Assistant Response References Older Message Content 时应用的 Relevance Boost。"""

    protect_first_n: int = 3
    """Context 中始终保留的 Head Exchanges 数量，防止初始约束被 Curator 丢弃。"""

    archive_dir: str = "memory/.curator/archive"
    """Workspace 下 Lossless Message Archives 的 Relative Path；目录创建与写入由 Context Engine 负责。"""


# 功能 2：调用效率
# ---------------------------------------------------------------------------


class BudgetPolicyConfig(_Base):
    """Per-session / Per-day Spend Limits 与 Warning Thresholds。

    Token/Cost Tracker 使用这些值做预算观测；仅配置 Limit 不等于 Provider 已强制硬停，具体 Enforcement
    取决于 Runtime Strategy 是否启用。
    """

    warn_at_usd: float = 0.50
    hard_limit_usd: float = 2.00
    warn_at_input_tokens: int = 500_000
    track_per_session: bool = True
    track_global_daily: bool = True


class SmartRoutingConfig(_Base):
    """SmartRouter 的 Tier 与 Fallback Configuration。

    默认 Disabled。启用后可按 Light/Medium/Heavy 候选列表选择模型；列表只是配置身份，不验证模型在线。
    """

    enabled: bool = False
    tiers: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "light": ["gemini-2.5-flash", "claude-haiku-4-5"],
            "medium": ["claude-sonnet-4-6", "gpt-4.1-mini"],
            "heavy": ["claude-opus-4-6", "gpt-4.1"],
        }
    )
    default_tier: Literal["light", "medium", "heavy"] = "heavy"
    """Routing Uncertain 时使用的 Fallback Tier，默认 Heavy 以保守保证能力。"""


class ToolResultLifecycleConfig(_Base):
    """Tool Result Lifecycle Management，即 Three-phase Pruner。

    Disabled 时不改结果；启用后按 Full-retention Turns、Summary-retention Turns 与 Placeholder 管理旧 Tool
    Output。Summary Model 只在需要压缩时使用，配置不代表历史结果已可检索。
    """

    enabled: bool = False
    full_retention_turns: int = 3
    summary_retention_turns: int = 10
    placeholder_text: str = "[Tool result archived — retrievable via Curator]"
    summary_model: str = "gemini-2.5-flash"


class CallEfficiencyConfig(_Base):
    """Provider Call Cache、Normalized Usage 与 Estimated Cost Policy。

    `mode` 决定 Off/Observe/Optimize；Legacy TokenWise Fields 保留兼容。Observe 只记录不 Rewrite Request，
    Optimize 才允许 Cache Planning。Usage Record、Cost Estimate 与预算 Enforcement 是不同职责。
    """

    mode: Literal["off", "observe", "optimize"] = "observe"
    """Default Observe；Request Rewriting 必须显式选择 ``optimize`` Mode。"""

    enabled: bool = True
    """Legacy TokenWise Switch；`False` 把 Effective Mode 映射为 ``off``。"""

    usage_tracking: bool = True
    """记录 Per-call Token Usage，Cheap and Informative，Default On；记录成功不等于 Cost 可计算。"""

    cache_optimization: bool = True
    """Legacy TokenWise Field；Canonical Request Behavior 只由 ``mode`` 选择。"""

    max_cache_breakpoints: int = 4
    """Anthropic API Breakpoint Limit，保留 Configurable 以便 Forward Compatibility；有效范围由 Runtime 验证。"""

    skill_lazy_loading: bool = False
    """只注入与 Current Message Relevant 的 Skill Summaries；Legacy Efficiency Knob。"""

    tool_result_lifecycle: ToolResultLifecycleConfig = Field(default_factory=ToolResultLifecycleConfig)
    smart_routing: SmartRoutingConfig = Field(default_factory=SmartRoutingConfig)
    budget: BudgetPolicyConfig = Field(default_factory=BudgetPolicyConfig)

    @property
    def effective_mode(self) -> Literal["off", "observe", "optimize"]:
        return self.mode if self.enabled else "off"


TokenWiseConfig = CallEfficiencyConfig


# ---------------------------------------------------------------------------
# 功能 3：SkillForge
# ---------------------------------------------------------------------------
#
# SkillForge 负责 Local Skill 的检索与执行。
#
# 配置有意保持扁平。组件级参数
# （嵌入模型、BM25 参数等）位于
# 暂时留在 ``skill_forge/`` 内的脚手架 dataclass 中并使用默认值；
# 需要向用户开放时，再由负责人把相应字段提升到这里。


class LocalDirConfig(_Base):
    """一条 Local Skill Directory Entry（R1）。

    Path、Enablement、Display Name 与 Always-injection Permission 共同决定 Catalog 如何 Mount 该 Source。
    """

    path: str
    """Absolute 或 ``~``-relative Path，在 Startup Expand；不存在目录会被 Catalog Skip/Warning。"""

    enabled: bool = True
    """`False` 表示 Directory Completely Skipped，不参与 Scan/Retrieval。"""

    name: str | None = None
    """Logs/Source 使用的 Display Name；`None` 时从 Path Basename Derive。"""

    always_enabled: bool = True
    """`False` 时该目录中 ``always: true`` Skills 不进入 Always Injection，但仍可经 ``select`` Retrieval。"""


class SkillForgeConfig(_Base):
    """SkillForge Configuration，当前聚焦 Operator-managed Local Skills。

    Active Runtime 不发 Provider Call 即 Resolve Local Skills；Repository Memory 与本 Subsystem Independent。
    Legacy Remote Retrieval、Reranker、Query Rewrite 与 LLM Gate Fields 仍可为旧 Evaluation Config 加载，但
    不参与 Active Runtime Path。配置项存在不能被当作这些远端能力已启用的证据。
    """

    # --- 总开关与位置 ---
    enabled: bool = True
    """发现 Configured Local Skill Directories 时使用的 Compatibility Master Switch。"""

    router: "SkillForgeRouterConfig" = Field(
        default_factory=lambda: SkillForgeRouterConfig(),
    )
    """Local BM25 Routing Policy，Config Key 为 ``skillForge.router``。

    Router 是 SkillForge Component，因此 Nested Here 而非 Sibling Top-level Block。使用 Forward-ref + 下方
    ``model_rebuild``，因为 ``SkillForgeRouterConfig`` 在本 Module 更后定义。
    """

    local_dirs: list[LocalDirConfig] = Field(default_factory=list)
    """要 Mount 的 Local Skill Directories（R1）。List Order 表示 Priority：Name Collision 时 Later Entry
    Override Earlier。Legacy ``skills_dir`` 经 Model Validator（R5）Auto-migrate。"""

    scan_max_depth: int = 5
    """扫描 ``SKILL.md`` 的 Maximum Directory Depth（R2）。Layer Root 下更深 Path Silently Skip，防止
    Huge Mirrors 上 Unbounded Filesystem Walk。"""

    # --- 检索/重排参数 ---
    embedding_model: str = "default"
    """Legacy Dense Embedding Model Identifier。

    若旧 Evaluation 使用 ``mass_library_db``，它 **MUST** Match 生成 Stored Vectors 的 Model，否则 Query
    Vector 位于 Different Space，Dense Retrieval 返回 Garbage。Active Local-only Runtime 忽略该远端路径。
    """

    embedding_url: str = "http://localhost:1357"
    """Legacy Remote Embedding Service Base URL。

    旧 Retrieval 调用 ``POST <embedding_url>/embed``；Hosted Service 可用 ``REMOTE_EMBEDDING_URL`` 或 User
    Config Override。Active Local Runtime 不发此 Request。
    """

    reranker_enabled: bool = True
    """Legacy Dense Retrieval 后是否运行 Reranker。旧路径 Default On，每 Query 增加约 200-500ms
    Cross-encoder GPU Inference，但提高 Mass-pool Precision；Latency 更重要时可关闭。Active Path 忽略。"""

    reranker_model: str = "default"
    """用于 Legacy Configuration 与 Observability 的 Reranker Model Label。"""

    reranker_url: str = "http://localhost:1357"
    """Legacy Remote Reranker Service Base URL。

    旧 Reranking 使用 ``POST <reranker_url>/score``，Body ``{"prompts": [...]}``，读取
    ``{"scores": [...]}``；Hosted Service 可用 ``REMOTE_RERANKER_URL`` 或 User Config Override。
    """

    embedding_api_key: str | None = None
    """Configured Embedding Service 的 Optional Bearer Token；加载成功不验证 Token 有效。"""

    reranker_api_key: str | None = None
    """Configured Reranker Service 的 Optional Bearer Token；应避免写入 Trace/Log。"""

    embedding_dimensions: int | None = None
    """为支持该能力的 Model 请求 Specific Embedding Dimensions；Legacy Remote Field。"""

    top_k: int = 5
    """Legacy ``select()`` 返回的 Skill 数；Active Router 使用 Nested Router Top-K。"""

    # --- 双池融合权重（R6）---
    local_pool_top_k: int = 10
    """每 Query 从 Local BM25 Pool 取得的 Candidate Count；Legacy Dual-pool Evaluation Field。"""

    mass_pool_top_k: int = 10
    """每 Query 从 Mass Dense Pool 取得的 Post-rerank Candidate Count；Active Local Path 不使用。"""

    local_weight: float = 1.3
    """Local-pool Candidates 的 RRF Weight，Mass 隐式为 1.0。Recommended Range ``[1.2, 1.5]``；小于
    1.0 或大于 2.0 会被 Validator 拒绝。"""

    mass_reranker_overfetch: int = 20
    """Legacy Reranker Enabled 时，Mass Pool 为 Rescoring Over-fetch 的数量，之后在 RRF 前截到
    ``mass_pool_top_k``。"""

    # --- 旧版查询改写参数 ---
    rewrite_enabled: bool = False
    """为 Config Compatibility 保留的 Legacy Evaluation Knob。

    Active Runtime Never 用 Provider Rewrite Local Skill Query；Resolution Local-only，因此不会 Delay Main
    Model Call。字段为 True 也不激活旧路径。
    """

    rewrite_max_tokens: int = 8192
    """Legacy Rewriter LLM Call 的 Output Token Budget。

    Default 8192 为 Qwen3-style Reasoning Traces（约 3-4K Tokens）及 Actual Rewrite 留 Headroom。旧 1024
    频繁产生 ``finish_reason=length`` + Empty Visible Content，触发
    ``Failed to parse rewrite response as JSON`` Fallback。Active Runtime 不调用 Rewriter。
    """

    mass_library_db: str | None = None
    """Deprecated Compatibility Field；Local Skill Retrieval 忽略，并在加载时 Warning。"""

    # --- 技能注入模式（full_body 或 summary）---
    injection_mode: str = "full_body"
    """Selected Skills 如何 Surface 给 Agent。

    ``"full_body"`` 默认 Inline 至多 ``inject_max`` 个 Explicit Local Matches，并把 Ambiguous Matches 暴露
    为 Compact References。``"summary"`` 把所有 Relevant Matches 作为 References，Agent Loop 再调用
    ``skill_read`` 加载 Selected Body。配置模式不保证 Agent 实际读取引用。
    """

    inject_max: int = 2
    """``injection_mode='full_body'`` 时 Inline 的 Maximum Skills；每个 Body 通常增加 1-5K Tokens。"""

    disable_always: bool = False
    """为 True 时 ``get_always_skills()`` 返回 ``[]``，Select 过滤 ``always:true`` Skills。R8 Default False，
    即 Always Skills Inject。"""

    always_max: int = 5
    """每 Turn Inject 的 Maximum Always Skills（R3）。超出时按 `local_dirs` List Order + Alphabetical
    Truncate，并以 WARN 列出 Dropped Skill Names。"""

    # --- 旧版 LLM 门控选择器 ---
    llm_gate_enabled: bool = False
    """为 Config Compatibility 保留的 Legacy LLM Gate Knob。

    Active Runtime 使用 Deterministic Local Confidence，不调用 LLM Gate；Ambiguous Candidates 成为 Main
    Agent Loop 可用 ``skill_read`` 检查的 Compact References。字段为 True 也不改变 Active Path。
    """

    llm_gate_max_select: int = 2
    """Legacy Gate 可 Select Skill 的 Upper Bound，与 ``inject_max`` 对齐。"""

    llm_gate_pool_size: int = 10
    """RRF 后交给 Legacy Gate 的 Candidate Pool Size，与 Local+Mass Dedup 后输出规模对齐。"""

    llm_gate_model: str | None = None
    """Legacy Gate Call 的 Optional Model Override；`None` 使用 Provider Default Chat Model，通常是 Agent
    Main Model。"""

    llm_gate_temperature: float = 0.0
    """Legacy Gate Sampling Temperature。0.0 用于 Deterministic Filtering；Reasoning Model 可能需 0.6
    才触发 ``<think>``。"""

    llm_gate_max_tokens: int = 8192
    """Legacy Gate LLM Call 的 Output Token Budget。

    Default 8192 为 Qwen3-style Reasoning（约 3-4K）+ JSON Answer 留余量。旧 4096 在 27B Model 约 50%
    Calls 出现 Empty Content（``finish_reason=length``），强制 Legacy Top-N Fallback 返回 5 Skills，而非
    Configured ``llm_gate_max_select``。
    """

    stats_tracking: bool = True
    """记录 Per-skill Invocation Stats；Cheap，可支持 Future Features，但统计不等于 Skill Success。"""

    # --- 校验器 ---

    @model_validator(mode="before")
    @classmethod
    def _migrate_skills_dir(cls, data: dict) -> dict:
        """R5：Auto-convert Legacy ``skills_dir`` → ``local_dirs``。

        同时接受 CamelCase，只有 New Field 未显式提供时才迁移，并发出 DeprecationWarning；还验证
        ``local_weight`` 在 ``[1.0, 2.0]``。输入非 Dict 原样返回。
        """
        if not isinstance(data, dict):
            return data
        for old_key in ("skills_dir", "skillsDir"):
            old_val = data.pop(old_key, None)
            if old_val and "local_dirs" not in data and "localDirs" not in data:
                data["local_dirs"] = [{"path": old_val}]
                warnings.warn(
                    f"skill_forge.{old_key} is deprecated, use local_dirs "
                    f"instead. Auto-converted to local_dirs=[{{path: {old_val!r}}}]. "
                    f"This field will be removed in a future release.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        lw = data.get("local_weight") or data.get("localWeight")
        if lw is not None:
            lw = float(lw)
            if lw < 1.0 or lw > 2.0:
                raise ValueError(f"local_weight={lw} out of valid range [1.0, 2.0]")
        return data


# ---------------------------------------------------------------------------
# CFG-1：插件 / 记忆后端 / SkillForgeRouter
# ---------------------------------------------------------------------------


class PluginsConfig(_Base):
    """Plugin-system Top-level Config。

    ``disabled`` 是按 Plugin ID（匹配 ``codeflow-plugin.toml`` 的 ``id``）索引的 User Opt-out List。``config``
    是 Registry 通过 :class:`PluginContext.config` 交给 Factory 的 Per-plugin Slice；Shape 由各 Manifest
    ``config_schema`` 决定，Host 视为 Free-form Dict，不替 Plugin 深度验证。
    """

    disabled: list[str] = Field(default_factory=list)
    """用户显式 Opted Out 的 Plugin IDs；Discovery 可见但 Activation 会 Skip。"""

    config: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """按 Plugin ID Keyed 的 Per-plugin Configuration。每个 Factory 接收
    ``ctx.config = plugins.config.get(<id>, {})``；Missing ID 得到 Empty Dict。"""


class MemoryConfig(_Base):
    """选择 Active Memory Backend，并配置 Recall Identity/Top-K。

    ``backend`` 是 Activated ``memory_backend`` Contribution Name；设 `None` 会禁用 Implicit Memory Recall、
    Persistence、Personalization 与 Curator Memory Tools，同时保留 Sessions/Local Skills。``user_id`` 是
    User Recall Track 的 Public Interface Identity。Backend 成功 Resolve 不代表远端 Store 可访问。
    """

    backend: str | None = "myna"
    """Activated Backend Contribution Name。`None` 禁用 Implicit Memory Path，但保留 Sessions、Curator
    State 与 Local Skills。"""

    user_id: str = "default"
    """在 ``ContextAssembler.assemble`` User-track Recall 中传给
    ``backend.recall(user_id=...)`` 的 Bare User Identity。"""

    memory_top_k: int = 5
    """每 Turn 为 ``# Recalled memory`` Block 传给 ``backend.recall(user_id=user_id)`` 的 Top-K Upper Bound。"""


class SkillForgeRouterConfig(_Base):
    """Active Runtime 的 Local Skill BM25 Routing Policy。

    该 Block 真正参与 Local-only Router：Master Switch、Minimum Score 与 Final Top-K。
    """

    enabled: bool = True
    """Master Switch。`False` 让 Host Entirely Bypass `SkillForgeRouter`，用于 Tests/Restricted Deployments。"""

    local_min_score: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
    )
    """Local Skill Source 可 Emit 的 Minimum BM25 Score；Bool 被 Validator 明确拒绝。"""

    @field_validator("local_min_score", mode="before")
    @classmethod
    def _reject_boolean_local_min_score(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("local_min_score must be a number")
        return value

    top_k: int = 5
    """``SkillForgeRouter.select`` 返回的 Final Top-K。"""


# ``SkillForgeRouterConfig`` 已存在于模块作用域，此处解析前向引用
# 字段声明：``SkillForgeConfig.router: "SkillForgeRouterConfig"``。
SkillForgeConfig.model_rebuild()


# ---------------------------------------------------------------------------
# 功能 4：Runtime 约束
# ---------------------------------------------------------------------------


class CheckpointConfig(_Base):
    """Workspace 的 Per-turn ``shadow-git`` Checkpoint Policy。

    Active 时，Agent Loop 在每 Turn 末尾把 Workspace Commit 到 Out-of-band Shadow Git Repo，覆盖 Normal 与
    Max-iteration Exit。它是 Bug2 Safety Net：Truncated Multi-file Edit 留下 Recoverable Snapshot，Next Turn
    得到列出 Interrupted Changes 的 Recovery Prompt。

    Activation 同时受 ``policy`` 与 AgentLoop ``interactive`` Flag 控制，后者由 CLI/TUI/Gateway Call Site
    设置。``"always"`` 覆盖所有 Loop，包括 ``-m`` One-shot；``"interactive"`` 只覆盖 REPL/TUI/Gateway
    Multi-turn Session，因为 One-shot 没有 Next Turn 可注入 Recovery；``"never"`` 完全关闭，Loop 与
    Pre-Bug2 Baseline Byte-identical，无 Commit、Interrupt Reclassification、Recovery Injection。

    Default ``"interactive"`` 对齐 Claude Code/Cursor 的 Long-session Transparent Checkpoint。Checkpoint
    Commit 成功只证明代码快照可恢复，不证明修改正确或任务完成。
    """

    policy: Literal["always", "interactive", "never"] = "interactive"
    """控制 Per-turn Shadow-git Snapshot 何时 Active；与 AgentLoop ``interactive`` Flag 的关系见 Class
    Docstring。"""

    shadow_dir: str = ".codeflow/shadow.git"
    """Shadow Git-dir。Project-local Foreground Run 相对 Workspace State，Colocated Legacy/Service Run 相对
    Workspace。Real Workspace 是 Work-tree，User Own ``.git`` **Never Touched**。"""


class RuntimeConfig(_Base):
    """Runtime Discipline，即 4th Feature Pillar。

     它承载 Opt-in Runtime Safety Nets。Bug2 已提供 ``checkpoint``；历史 Roadmap 计划加入 ``journal`` /
     ``verifier`` / ``done_gate`` / ``loop_detection``（Bug3）与 ``session``（Bug1）Sibling Config。当前只声明
    已实现字段，不能把 Roadmap 名称当成 Runtime 已交付能力。
    """

    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)


class TracingConfig(_Base):
    """In-tree ``codeflow.tracing`` Observability Config。

    Default On；每个 ``codeflow`` Command 在 AgentLoop 构造前安装 Non-invasive Instrumentation。
    ``CODEFLOW_TRACING=0`` 是覆盖此 Block 的 Explicit Env Kill-switch。``codeflow tracing`` 或 TUI ``/tracing``
    查看 Captured Traces。Enabled 不保证 Viewer Running 或 Span Write 成功。
    """

    enabled: bool = True
    port: int = 4318
    preview_len: int = 500


# ---------------------------------------------------------------------------
# 根配置
# ---------------------------------------------------------------------------


class CodeFlowConfig(_Base):
    """CodeFlow Root Config，把 Base `Config` 与 Feature Extensions Compose 在一起。

    Root 拥有 Context、Call Efficiency、SkillForge、Runtime、Tracing、Plugins、Memory 与完整 Base Config。
    Pydantic 统一处理 Camel/Snake Aliases 与 Legacy TokenWise Migration；对象是 Process Configuration
    Snapshot，不自动热更新 Disk Changes。
    """

    # 功能配置块
    context: ContextConfig = Field(default_factory=ContextConfig)
    call_efficiency: CallEfficiencyConfig = Field(default_factory=CallEfficiencyConfig)
    # SkillForge 子系统：其 RRF 路由策略嵌套在
    # ``skill_forge.router``（配置键 ``skillForge.router``），不再是
    # 而不是独立的顶层 ``skillRouter`` 块。
    skill_forge: SkillForgeConfig = Field(default_factory=SkillForgeConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)

    # CFG-1：插件系统与记忆后端。
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    # 完整基础配置（agents、channels、providers、tools、routing）。
    # 保持为嵌套字段，以便与基础 loader 往返转换 YAML。
    base: BaseConfig = Field(default_factory=BaseConfig)

    @model_validator(mode="before")
    @classmethod
    def _migrate_token_wise_block(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        migrated = dict(data)
        legacy_camel = migrated.pop("tokenWise", None)
        legacy_snake = migrated.pop("token_wise", None)
        legacy = legacy_camel if legacy_camel is not None else legacy_snake
        if "callEfficiency" in migrated:
            migrated.pop("call_efficiency", None)
        if "callEfficiency" not in migrated and "call_efficiency" not in migrated and legacy is not None:
            migrated["callEfficiency"] = legacy
        return migrated

    @property
    def token_wise(self) -> CallEfficiencyConfig:
        """为 Historical Code 与 Frozen Benchmarks 提供 `call_efficiency` 的 Compatibility View。

        返回同一 Config Object，不维护第二份 TokenWise State。
        """
        return self.call_efficiency


def load_codeflow_config(config_path: Path | None = None) -> CodeFlowConfig:
    """从同一 JSON File 加载 Base Config 与 CodeFlow Extension Blocks。

    包括 ``context``、``call_efficiency``、``skill_forge``、Runtime/Tracing/Plugins/Memory。`config_path` 为
    Optional，缺失时使用 Active Default。先调用 Base Loader，再以 ``pop_extension_keys=False`` 运行同一
    Migration，并提取 `EXTENSION_KEYS`。

    JSON 无 Entry 或显式 ``null`` 的 Extension Block 使用 Dataclass Default。Extension File Parse/IO Error
    当前降级 Empty Overrides；Base Loader 已按其独立策略处理。返回对象表示 Config Merge 完成，不验证
    Feature Backends。
    """
    base = load_base_config(config_path)

    overrides: dict = {}
    actual_path = config_path or get_config_path()
    if actual_path.exists():
        try:
            with open(actual_path, encoding="utf-8") as f:
                data = json.load(f) or {}
        except (json.JSONDecodeError, OSError):
            data = {}
        # 提取扩展块前，执行与基础 loader 相同的迁移。
        data = _migrate_config(data, pop_extension_keys=False)
        # 用户仍保留已忽略的旧配置时只警告一次。
        # ``skill_forge.mass_library_db`` 字段。
        _warn_mass_library_db_deprecated(data)
        for key in EXTENSION_KEYS:
            if key in data and data[key] is not None:
                overrides[key] = data[key]

    return CodeFlowConfig(base=base, **overrides)


def _warn_mass_library_db_deprecated(data: dict) -> None:
    """为 ``skill_forge.mass_library_db`` 发出 Single-shot Deprecation Warning。

    Legacy SQLite Field 为 Config Compatibility 保留，但 Local Skill Retrieval 忽略。函数兼容 Snake/Camel
    Block/Field Names；无非空 Legacy Value 时 No-op。
    """
    legacy = None
    for skill_forge_key in ("skill_forge", "skillForge"):
        block = data.get(skill_forge_key)
        if isinstance(block, dict):
            legacy = block.get("mass_library_db") or block.get("massLibraryDb")
            if legacy:
                break
    if not legacy:
        return
    warnings.warn(
        "skill_forge.mass_library_db is deprecated and ignored. Local skills "
        "are discovered from configured filesystem sources; remove this field.",
        DeprecationWarning,
        stacklevel=2,
    )
