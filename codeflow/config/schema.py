"""CodeFlow Base Agent Runtime 的 Pydantic Configuration Schema。

这里定义 Channels、Agent Defaults、Providers、Routing、Gateway、Tools、Cron 与 Root `Config`。Schema
同时接受 CamelCase/Snake_case，提供 Defaults，并把 Unknown Root/Feature Fields 交给 Loader/Config
Validation 处理。模型构造成功只证明字段类型合法，不验证 Credentials、Network、Executable 或 MCP Server。
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings

from codeflow.product import DEFAULT_WORKSPACE_SPEC, get_default_workspace
from codeflow.sandbox.config import SandboxConfig


class Base(BaseModel):
    """同时接受 ``camelCase`` 与 ``snake_case`` Keys 的 Base Pydantic Model。

    Alias Generator 统一 File/CLI 与 Python Naming，`populate_by_name=True` 让两种拼写都能 Validate。
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class FeishuConfig(Base):
    """使用 WebSocket Long Connection 的 Feishu/Lark Channel Configuration。

    包含 App Credentials、Event Verification、Sender Allowlist、Reaction 与 Group Policy。`enabled=True` 只
    表示 Runtime 应尝试启动，连接与鉴权仍在 Channel Adapter 验证。
    """

    enabled: bool = False
    app_id: str = Field(default="", json_schema_extra={"required": True})  # 飞书开放平台 App ID
    app_secret: str = Field(default="", json_schema_extra={"required": True})  # 飞书开放平台 App Secret
    encrypt_key: str = ""  # 事件订阅 Encrypt Key
    verification_token: str = ""  # 事件订阅 Verification Token
    allow_from: list[str] = Field(default_factory=lambda: ["*"])  # 允许的用户 open_id；['*'] 表示任何人
    react_emoji: str = "THUMBSUP"  # 消息表态类型，如 THUMBSUP、OK、DONE、SMILE
    group_policy: Literal["open", "mention"] = "mention"  # "mention" 仅在被 @ 时响应，"open" 响应全部消息


class QQConfig(Base):
    """使用 Botpy SDK 的 QQ Channel Configuration。

    App ID/Secret 与 Sender OpenID Allowlist 由 Adapter 消费；Defaults 公开访问但仍需平台鉴权。
    """

    enabled: bool = False
    app_id: str = Field(default="", json_schema_extra={"required": True})  # q.qq.com 机器人 AppID
    secret: str = Field(default="", json_schema_extra={"required": True})  # q.qq.com 机器人 AppSecret
    allow_from: list[str] = Field(default_factory=lambda: ["*"])  # 允许的用户 openid；['*'] 表示公开访问


class WecomConfig(Base):
    """WeCom（Enterprise WeChat）AI Bot Channel Configuration。

    包含 Bot ID/Secret、Allowlist 与 Enter-chat Welcome Message。Config 不持有 Socket/Delivery State。
    """

    enabled: bool = False
    bot_id: str = Field(default="", json_schema_extra={"required": True})  # 企业微信 AI Bot 平台 Bot ID
    secret: str = Field(default="", json_schema_extra={"required": True})  # 企业微信 AI Bot 平台 Bot Secret
    allow_from: list[str] = Field(default_factory=lambda: ["*"])  # 允许的用户 ID；['*'] 表示任何人
    welcome_message: str = ""  # enter_chat 事件的欢迎消息


class ChannelsConfig(Base):
    """Chat Channels 的 Root Configuration。

    `send_progress`/`send_tool_hints` 控制 Runtime 中间信息，Per-channel Blocks 决定启用与凭据。Channel
    Enabled 与 Message Delivered 必须通过运行时证据区分。
    """

    send_progress: bool = True  # 向渠道流式发送 Agent 文本进度
    send_tool_hints: bool = False  # 流式发送工具调用提示，如 read_file("…")
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)
    qq: QQConfig = Field(default_factory=QQConfig)
    wecom: WecomConfig = Field(default_factory=WecomConfig)


class AgentDefaults(Base):
    """Agent Runtime 的 Default Model、Workspace、Budget、Tool-loop 与 Subagent Safety Configuration。

    这些值作为每个 AgentLoop 的构造基线。Concurrency/Hourly Spawn Limits 防止 Prompt-injected Agent 无界
    Spawn；Empty-response Recovery Budgets 限制额外 Nudges/Retry；Deprecated Fields 仅为旧 Config Load。
    """

    workspace: str = DEFAULT_WORKSPACE_SPEC
    model: str = "anthropic/claude-opus-4-5"
    provider: str = "auto"  # Provider 名称，如 "anthropic"、"openrouter"；"auto" 表示自动检测
    max_tokens: int = 8192
    context_window_tokens: int = 65_536
    temperature: float = 0.1
    max_tool_iterations: int = 40
    # 同时运行的 subagent VM 上限；超额 spawn 会排队。ge=1：
    # 上限为 0 或负数会让所有子智能体死锁（Semaphore(0)）。
    max_concurrent_subagents: int = Field(default=4, ge=1)
    # 每个 session 在滚动一小时内的 spawn 速率限制。单靠并发门禁无法阻止
    # 被 prompt 注入的 Agent 无限 spawn：每个任务结束后都会释放槽位，跨 Turn
    # 重注入循环也不需要用户输入。滚动窗口把失控上限限制在 N 次/小时，且能
    # 自动恢复，不会永久阻止高强度的合法使用。按 session 分别计数，避免繁忙
    # session 限流其他 session。
    max_subagent_spawns_per_hour: int = Field(default=30, ge=1)
    # 空响应恢复：挽救模型未产生可见文本便结束、但实际上仍有内容可给出的 Turn，
    # 包括工具调用后为空或只有思考的情况，避免暴露无效的“无响应”结果。预算按 Turn 计算。
    empty_recovery_enabled: bool = True
    post_tool_empty_max_nudges: int = 1
    thinking_prefill_max_retries: int = 2
    empty_content_max_retries: int = 3
    # 已弃用的兼容字段：接受旧配置，但 Runtime 会忽略。
    memory_window: int | None = Field(default=None, exclude=True)
    reasoning_effort: str | None = None  # low / medium / high，用于启用 LLM thinking mode
    enable_personalization: bool = False  # 受 PAHF 启发的四步个性化流程：分类、询问、执行、学习

    @property
    def should_warn_deprecated_memory_window(self) -> bool:
        """旧 ``memoryWindow`` 存在且未显式提供 ``contextWindowTokens`` 时返回 `True`。

        供 CLI 发出迁移 Warning；不会自动把旧值映射成新 Context Window。
        """
        return self.memory_window is not None and "context_window_tokens" not in self.model_fields_set


class AgentsConfig(Base):
    """Agent Configuration Root，目前承载共享 `defaults`。

    Wrapper 为未来 Per-agent Overrides 保留稳定层级；当前所有 AgentLoop 从同一 `AgentDefaults` 构造基线。
    """

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class CronConfig(Base):
    """Cron Scheduler 的 Trigger-time Delivery Configuration。

    **Only** 在 Cron Job TRIGGER Time 读取，Creation Time 不使用。Ephemeral CLI/TUI，即不在
    `ChannelManager.enabled_channels` 的 Channel，在 Host Process 退出后无法自行 Deliver；
    `forward_channels` 决定哪些 Real Channels 接收 Reminder。
    """

    forward_channels: list[str] = Field(default_factory=lambda: ["*"])
    """Ephemeral-origin Reminder 的 Forward Channels。``["*"]`` Broadcast 到每个 Enabled Channel；
    ``["feishu", "qq"]`` 等 Specific Names 限制范围。Non-ephemeral Channel 忽略该 List，始终按 Per-job
    Binding Pass-through。"""

    default_timezone: str = "Asia/Shanghai"
    """Cron Expression 未显式 ``--tz`` 时使用的 Default IANA Timezone。"""


class ProviderConfig(Base):
    """通用 LLM Provider Configuration。

    包含 API Key、Optional Base URL/Headers 与 User-curated Model Names。Secret 只是配置值，不应进入
    Diagnostics；字段非空也不代表鉴权成功。
    """

    api_key: str = ""
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None  # 自定义 header，如 AiHubMix 的 APP-Code
    models: list[str] = Field(default_factory=list)  # 用户为选择器整理的模型名称


class GeminiProviderConfig(ProviderConfig):
    """支持 Vertex AI 与 Multi-key Rotation 的 Gemini Provider Config。

    Example YAML：
        gemini:
          vertex: true
          api_key_list:
            - "key1"
            - "key2"
    `vertex=True` 由 Provider Setup 设置 Vertex Environment；`api_key_list` 按 Round-robin 轮换。Schema 不
    联网验证 Keys。
    """

    vertex: bool = False  # 为 True 时设置 GOOGLE_GENAI_USE_VERTEXAI=True，以使用 Vertex AI
    api_key_list: list[str] = Field(default_factory=list)  # 用于轮换的多个 API key

    def next_api_key(self) -> str:
        """使用 Round-robin Rotation 返回 Next API Key。

        `api_key_list` Empty 时回退 Single ``api_key``；完全无 Key 返回空字符串。Cycle 在实例上 Lazy
        Create，后续调用持续轮换，不探测 Key 是否可用。
        """
        import itertools

        if not hasattr(self, "_key_cycle"):
            keys = self.api_key_list or ([self.api_key] if self.api_key else [])
            object.__setattr__(self, "_key_cycle", itertools.cycle(keys) if keys else None)
        cycle = getattr(self, "_key_cycle", None)
        if cycle is None:
            return self.api_key or ""
        return next(cycle)

    @property
    def effective_api_key(self) -> str:
        """返回 Current Effective API Key：List First Item 优先，否则 Single Key。

        该 Property 不推进 Round-robin Cycle，适合环境初始化 Preview。
        """
        if self.api_key_list:
            return self.api_key_list[0]
        return self.api_key

    @property
    def all_keys(self) -> list[str]:
        """返回所有 Configured API Keys 的 List Copy。

        Multi-key List 优先，只有 Single Key 时包装成 One-item List，完全无配置时为空；不会推进 Rotation。
        """
        if self.api_key_list:
            return list(self.api_key_list)
        return [self.api_key] if self.api_key else []


class ProvidersConfig(Base):
    """所有 Builtin LLM Providers 的 Configuration Registry。

    每个字段使用 `ProviderConfig` 或 Specialized Gemini Config。Provider Selection 由 Root Config 与
    `codeflow.providers.registry` 完成；声明 Block 不自动 Import/Connect Provider。
    """

    custom: ProviderConfig = Field(default_factory=ProviderConfig)  # 任意 OpenAI-compatible 端点
    azure_openai: ProviderConfig = Field(default_factory=ProviderConfig)  # Azure OpenAI，model 为部署名称
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)  # 阿里云通义千问
    vllm: ProviderConfig = Field(default_factory=ProviderConfig)
    gemini: GeminiProviderConfig = Field(default_factory=GeminiProviderConfig)  # 提供商：Google Gemini / Vertex AI
    moonshot: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax: ProviderConfig = Field(default_factory=ProviderConfig)
    aihubmix: ProviderConfig = Field(default_factory=ProviderConfig)  # AiHubMix API 网关
    ollama: ProviderConfig = Field(default_factory=ProviderConfig)  # Ollama 本地模型
    siliconflow: ProviderConfig = Field(default_factory=ProviderConfig)  # 提供商：SiliconFlow
    volcengine: ProviderConfig = Field(default_factory=ProviderConfig)  # 提供商：VolcEngine
    openai_codex: ProviderConfig = Field(default_factory=ProviderConfig)  # 提供商：OpenAI Codex（OAuth）
    github_copilot: ProviderConfig = Field(default_factory=ProviderConfig)  # 提供商：Github Copilot（OAuth）


class ModelEndpoint(Base):
    """一个 Routable Model 与服务它的 OpenAI-compatible Endpoint。

    KNN Router 用该记录建立候选与请求地址；Default ``api_key="EMPTY"`` 适合不鉴权 Local Endpoint，
    不是生产 Secret。
    """

    model: str = ""
    api_base: str = ""
    api_key: str = "EMPTY"


class RoutingConfig(Base):
    """Model Routing Configuration。

    ``backend`` 选择 ``ecoclaw``（Original PinchBench Benchmark Scores）或 ``knn``（Per-model Rewards 上的
    Task-level KNN）。KNN Fields 只有 ``backend == 'knn'`` 时读取。Similarity/Neighbour/Memory/Margin Gates
    让证据不足时保留 Default Model；Routing Enabled 不保证候选 Endpoint 在线。
    """

    enabled: bool = False
    backend: str = "ecoclaw"  # 可选 ecoclaw 或 knn
    profile: str = "balanced"  # 可选 best、balanced 或 eco
    # 用于 embedding 的 OpenRouter API key（ecoclaw 后端；默认取 providers.openrouter.api_key）
    api_key: str = ""
    # knn 后端：可路由模型及其端点
    models: list[ModelEndpoint] = Field(default_factory=list)
    # knn 后端：预构建 KNN 记忆（embedding + 各模型奖励/成本）
    memory_path: str = ""
    k: int = 30  # 检索宽度：拉取多少个最近邻
    lambda_cost: float = 0.0  # 分数 = 奖励 - lambda_cost * 成本
    embedding_endpoint: str = ""  # 入站任务使用的 embedding 服务
    # knn 后端安全门禁：只有证据充分时才离开默认模型。
    # 在“相似”邻居（cosine >= min_similarity）上为候选模型评分。
    min_similarity: float = 0.6  # cosine 达到此值才将邻居视为相似
    min_similar_neighbors: int = 4  # 至少需要这么多相似邻居才路由
    min_memory_size: int = 10  # 至少需要这么多记忆条目才允许路由
    min_margin: float = 0.0  # 候选分数至少领先默认模型此值才切换


class GatewayLogConfig(Base):
    """Gateway Logging、Rotation 与 Retention Configuration。

    ``rotation`` / ``retention`` 接受 Loguru Vocabulary：按 Size ``"10 MB"``、Wall-clock ``"00:00"`` Daily、
    Interval ``"1 week"`` 轮换；Retention 可为 File Count ``7`` 或 Duration ``"14 days"``。``level`` 过滤
    Persisted ``gateway.log``，``console_level`` 过滤 Foreground Gateway Live Stderr Mirror。
    """

    rotation: str = "10 MB"
    retention: int | str = 7
    level: str = "INFO"
    console_level: str = "INFO"


class GatewayConfig(Base):
    """Long-running Gateway/Server Configuration。

    Host/Port 决定 Listener，User/System Pools 控制 Lane Capacity，Send Retries 控制 Delivery Retry，Log Block
    管理文件与 Console。配置不执行 Bind 或健康检查。
    """

    host: str = "0.0.0.0"
    port: int = 18790
    user_pool: int = 4
    system_pool: int = 2
    send_max_retries: int = 3
    log: GatewayLogConfig = Field(default_factory=GatewayLogConfig)


class WebSearchConfig(Base):
    """Web Search Tool Configuration。

    包含 Serper API Key 与 Per-query Maximum Results；Schema 不验证 Credential 或 Search Endpoint 可达性。
    """

    api_key: str = ""  # Serper API 密钥
    max_results: int = 5


class WebToolsConfig(Base):
    """Web Tools 的 Shared Configuration。

    包含 Optional HTTP/SOCKS5 Proxy、Jina Reader API Key 与 Search Sub-config；每个 Tool 按需消费字段。
    """

    proxy: str | None = None  # HTTP/SOCKS5 代理 URL，如 "http://127.0.0.1:7890" 或 "socks5://127.0.0.1:1080"
    jina_api_key: str = ""  # Jina Reader API 密钥
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(Base):
    """Shell Exec Tool 的 Default Timeout 与 PATH Append Configuration。

    该 Block 不决定 Host/Sandbox Backend，隔离策略位于 `ToolsConfig.sandbox`。
    """

    timeout: int = 60
    path_append: str = ""


class MCPServerConfig(Base):
    """MCP Server Connection Configuration，支持 Stdio、SSE、Streamable HTTP。

    Type 省略时 Runtime Auto-detect。Stdio 使用 Command/Args/Env；HTTP/SSE 使用 URL/Headers；
    `tool_timeout` 限制 Tool Call。Schema 合法不代表 Server 可启动或 Handshake 成功。
    """

    type: Literal["stdio", "sse", "streamableHttp"] | None = None  # 省略时自动检测
    command: str = ""  # Stdio：要运行的命令，如 "npx"
    args: list[str] = Field(default_factory=list)  # Stdio：命令参数
    env: dict[str, str] = Field(default_factory=dict)  # Stdio：额外环境变量
    url: str = ""  # HTTP/SSE：端点 URL
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP/SSE：自定义 header
    tool_timeout: int = 30  # 工具调用取消前等待的秒数


class ToolSearchConfig(Base):
    """Progressive Tool Disclosure Configuration。

    Live Tool Catalog（Builtins + Plugins + MCP）超过 ``compaction_threshold`` 时，大多数 Tool Schemas 不再
    随每个 Request 发送，而通过 ``tool_search`` / ``tool_call`` Meta-tools On-demand 访问，使 Context Cost
    不再随 Tool Count 线性增长，Per-turn Tool List 与 Prompt Cache 更稳定。Threshold 以下全部 Directly
    Exposed，Meta-tools Omitted，保持旧行为。
    """

    enabled: bool = False
    compaction_threshold: int = 50
    """触发 Compaction 的 Tool-catalog Size。At or Below 时 Direct Exposure，Above 时 Withhold Schemas。"""
    search_result_limit: int = 10
    """``tool_search`` 每 Query 返回的 Default Hit Count。"""
    always_visible: list[str] = Field(default_factory=list)
    """除 Core Set 外，每 Turn 仍 Direct Exposed 的 Extra Tool Names。"""


class ToolsConfig(Base):
    """Agent Tool Surface 的 Root Configuration。

    聚合 Web、Exec、Workspace Restriction、MCP、Sandbox、Progressive Search 与 Disabled Tools。各子项只配置
    Admission/Behavior，实际 Registration/Connection 由 Runtime Stack 完成。
    """

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False  # 为 True 时把所有工具访问限制在 workspace 目录
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    tool_search: ToolSearchConfig = Field(default_factory=ToolSearchConfig)
    disabled_tools: list[str] = Field(default_factory=list)
    """Default-tool Registration 与 MCP Connect 后要 Unregister 的 Tool Names。

    Eval Harnesses 如 BrowseComp-Plus 用它把 Agent 限制到 Specific Subset。Names 必须匹配
    ``ToolRegistry``，例如 ``read_file``、``web_search``、``mcp_bcp-search_search``。Unknown Name 通常无
    效果，实际移除需运行时验证。
    """


class Config(BaseSettings):
    """CodeFlow Base Agent Runtime 的 Root Configuration。

    组合 Agents、Channels、Providers、Gateway、Tools、Routing、Cron 与 UI Language，并支持 ``CODEFLOW_`` Env
    Prefix + ``__`` Nested Delimiter。CodeFlow Feature Extensions 通过 `load_codeflow_config` 另行 Compose。
    """

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    cron: CronConfig = Field(default_factory=CronConfig)
    # onboarding 时选择的 UI 语言，控制向导/CLI 文案和 Agent 回复语言
    # （注入 system prompt）。取值为 "en" | "zh"。
    language: Literal["en", "zh"] = "en"

    @property
    def workspace_path(self) -> Path:
        """返回 Expand 后的 Workspace Path。

        Default Spec 映射到 Product Default Workspace；Custom Value 只 Expanduser，不创建目录。
        """
        workspace = self.agents.defaults.workspace
        return get_default_workspace() if workspace == DEFAULT_WORKSPACE_SPEC else Path(workspace).expanduser()

    def _match_provider(self, model: str | None = None) -> tuple["ProviderConfig | None", str | None]:
        """匹配 Provider Config 与 Registry Name，返回 ``(config, spec_name)``。

        Forced Provider 优先；Auto 模式先匹配 Explicit Model Prefix，再按 Registry Keywords，之后尝试已配置
        Local Provider，最后按 Registry 顺序回退 Non-OAuth Credential Provider。OAuth Provider 不能隐式
        Fallback，必须显式 Model Prefix。无可用配置返回 ``(None, None)``。
        """
        from codeflow.providers.registry import PROVIDERS

        forced = self.agents.defaults.provider
        if forced != "auto":
            p = getattr(self.providers, forced, None)
            return (p, forced) if p else (None, None)

        model_lower = (model or self.agents.defaults.model).lower()
        model_normalized = model_lower.replace("-", "_")
        model_prefix = model_lower.split("/", 1)[0] if "/" in model_lower else ""
        normalized_prefix = model_prefix.replace("-", "_")

        def _kw_matches(kw: str) -> bool:
            kw = kw.lower()
            return kw in model_lower or kw.replace("-", "_") in model_normalized

        # 显式 Provider 前缀优先，避免 `github-copilot/...codex` 匹配 openai_codex。
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and model_prefix and normalized_prefix == spec.name:
                if spec.is_oauth or spec.is_local or p.api_key:
                    return p, spec.name

        # 按关键字匹配，顺序与 PROVIDERS registry 一致。
        for spec in PROVIDERS:
            p = getattr(self.providers, spec.name, None)
            if p and any(_kw_matches(kw) for kw in spec.keywords):
                if spec.is_oauth or spec.is_local or p.api_key:
                    return p, spec.name

        # 回退：已配置的本地 Provider 可路由不含 Provider 专属关键字的模型，
        # 例如 Ollama 上的纯 "llama3.2"。
        for spec in PROVIDERS:
            if not spec.is_local:
                continue
            p = getattr(self.providers, spec.name, None)
            if p and p.api_base:
                return p, spec.name

        # 回退：先网关，再按 registry 顺序尝试其他 Provider。
        # OAuth Provider 不能作为回退，必须显式选择模型。
        for spec in PROVIDERS:
            if spec.is_oauth:
                continue
            p = getattr(self.providers, spec.name, None)
            if p and p.api_key:
                return p, spec.name
        return None, None

    def get_provider(self, model: str | None = None) -> ProviderConfig | None:
        """返回 Matched Provider Config，包括 ``api_key``、``api_base``、``extra_headers``；必要时回退 First Available。

        返回对象不执行鉴权或模型可用性检查。
        """
        p, _ = self._match_provider(model)
        return p

    def get_provider_name(self, model: str | None = None) -> str | None:
        """返回 Matched Provider Registry Name，例如 ``"deepseek"`` / ``"openrouter"``；无匹配为 `None`。"""
        _, name = self._match_provider(model)
        return name

    def get_api_key(self, model: str | None = None) -> str | None:
        """返回给定 Model 的 API Key，必要时回退 First Available Key；无 Provider 返回 `None`。"""
        p = self.get_provider(model)
        return p.api_key if p else None

    def get_api_base(self, model: str | None = None) -> str | None:
        """返回给定 Model 的 API Base URL，并为 Gateway/Local Provider 应用 Default URLs。

        Explicit Config 优先。Standard Provider 的 Base 通常由 Environment Setup 管理，避免污染 Global
        LiteLLM Base；无适用 URL 返回 `None`。
        """
        from codeflow.providers.registry import find_by_name

        p, name = self._match_provider(model)
        if p and p.api_base:
            return p.api_base
        # 此处仅为网关设置默认 api_base。标准 Provider
        # Moonshot 等提供商会在 _setup_env 中通过环境变量设置 base URL。
        # 以避免污染全局 litellm.api_base。
        if name:
            spec = find_by_name(name)
            if spec and (spec.is_gateway or spec.is_local) and spec.default_api_base:
                return spec.default_api_base
        return None

    @property
    def skill_forge(self):
        """返回 Default `SkillForgeConfig` Compatibility View。

        Extension Blocks 由 ``load_codeflow_config`` 加载，不通过 Base `Config`。该 Property 只为访问 Plain
        ``config.skill_forge`` / ``Config.skill_forge`` 的 Older Code 保留；每次返回 Default Object，不代表 User Extension Values。
        """
        from codeflow.config.codeflow import SkillForgeConfig

        return SkillForgeConfig()

    model_config = ConfigDict(
        env_prefix="CODEFLOW_",
        env_nested_delimiter="__",
        extra="forbid",
    )
