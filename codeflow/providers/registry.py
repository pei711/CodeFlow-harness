"""Provider Registry 是 LLM Provider Metadata 的 Single Source of Truth。

新增 Provider 只需两步：在下方 PROVIDERS 增加 ProviderSpec，再在 config/schema.py 的
ProvidersConfig 增加 Field。Env Vars、Model Prefix、Config Matching 与 Status Display 均从这里
推导，不能在多个模块复制 Provider Table。

Registry Order 控制 Match Priority 与 Fallback，Gateway 必须优先。每项显式写全 Field，既作为
可 Copy-paste Template，也让 Default/Capability 差异可审查。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderSpec:
    """描述一个 LLM Provider 的匹配、配置、路由与能力 Metadata。

    Identity Field 决定 Config Name/Display/Keyword；Prefix Field 规范 LiteLLM Model ID；Gateway/Local
    Detection 可按 Provider Name、Key Prefix、Base URL；Capability 说明 OAuth、Direct、Prompt Cache、
    Reasoning Replay 与 Default Model。具体实例见下方 PROVIDERS。

    ``env_extras`` Value 支持 ``{api_key}``（User API Key）与 ``{api_base}``（Config Base 或 Spec
    ``default_api_base``）Placeholder。Frozen Spec 是 Registry Fact，不持有真实 Client 或 Secret
    Lifecycle。
    """

    # 身份
    name: str  # 配置字段名，如 "dashscope"
    keywords: tuple[str, ...]  # 用于匹配的模型名关键字（小写）
    env_key: str  # LiteLLM 环境变量，如 "DASHSCOPE_API_KEY"
    display_name: str = ""  # 显示在 `codeflow status` 中

    # 模型前缀
    litellm_prefix: str = ""  # "dashscope" 表示模型变为 "dashscope/{model}"
    skip_prefixes: tuple[str, ...] = ()  # 模型已有这些前缀时不再添加

    # 额外环境变量，如 (("ZHIPUAI_API_KEY", "{api_key}"),)
    env_extras: tuple[tuple[str, str], ...] = ()

    # 网关/本地检测
    is_gateway: bool = False  # 可路由任意模型，如 OpenRouter、AiHubMix
    is_local: bool = False  # 本地部署，如 vLLM、Ollama
    detect_by_key_prefix: str = ""  # 匹配 api_key 前缀，如 "sk-or-"
    detect_by_base_keyword: str = ""  # 匹配 api_base URL 中的子串
    default_api_base: str = ""  # 回退 base URL

    # 网关行为
    strip_model_prefix: bool = False  # 重新添加前缀前移除 "provider/"

    # 各模型参数覆盖，如 (("kimi-k2.5", {"temperature": 1.0}),)
    model_overrides: tuple[tuple[str, dict[str, Any]], ...] = ()

    # 基于 OAuth 的 Provider（如 OpenAI Codex）不使用 API key。
    is_oauth: bool = False  # 为 True 时使用 OAuth 流程而非 API key

    # 直连 Provider 完全绕过 LiteLLM，如 CustomProvider。
    is_direct: bool = False

    # Provider 支持 content block 上的 cache_control，如 Anthropic prompt caching。
    supports_prompt_caching: bool = False

    # Provider 要求重放的 assistant 工具调用消息包含 reasoning_content。
    requires_reasoning_content_replay: bool = False

    # /v1/models 为空时，onboard 向导为 agents.defaults.model 使用的回退值。
    default_model: str = ""

    @property
    def label(self) -> str:
        return self.display_name or self.name.title()


# ---------------------------------------------------------------------------
# PROVIDERS registry，顺序即优先级；新增项可复制任一条目作为模板。
# ---------------------------------------------------------------------------

PROVIDERS: tuple[ProviderSpec, ...] = (
    # === Custom（直连 OpenAI-compatible 端点，绕过 LiteLLM）======
    ProviderSpec(
        name="custom",
        keywords=(),
        env_key="",
        display_name="Custom",
        # 作为通用 OpenAI-compatible 网关通过 LiteLLM 路由：`openai/` 前缀加
        # 配置的 api_base 可访问任意兼容端点，LiteLLM 还提供旧版直连
        # CustomProvider 缺少的 streaming、重试和工具调用。仅在显式选择
        # `provider: custom` 时匹配，keywords 为空。
        litellm_prefix="openai",
        is_gateway=True,
        default_api_base="http://localhost:8000/v1",
    ),
    # === Azure OpenAI（直连 API，版本 2024-10-21）=====
    ProviderSpec(
        name="azure_openai",
        keywords=("azure", "azure-openai"),
        env_key="",
        display_name="Azure OpenAI",
        litellm_prefix="",
        is_direct=True,
    ),
    # === 网关（按 api_key / api_base 检测，而不是模型名）=========
    # 网关可路由任意模型，因此在 fallback 中优先。
    # OpenRouter：全局网关，key 以 "sk-or-" 开头。
    ProviderSpec(
        name="openrouter",
        keywords=("openrouter",),
        env_key="OPENROUTER_API_KEY",
        display_name="OpenRouter",
        litellm_prefix="openrouter",  # 模型映射：claude-3 → openrouter/claude-3
        skip_prefixes=(),
        env_extras=(),
        is_gateway=True,
        is_local=False,
        detect_by_key_prefix="sk-or-",
        detect_by_base_keyword="openrouter",
        default_api_base="https://openrouter.ai/api/v1",
        strip_model_prefix=False,
        model_overrides=(),
        supports_prompt_caching=True,
        default_model="openrouter/anthropic/claude-sonnet-4-5",
    ),
    # AiHubMix：提供 OpenAI-compatible 接口的全局网关。
    # strip_model_prefix=True：它不理解 "anthropic/claude-3"，因此先移除为
    # "claude-3"，再添加前缀得到 "openai/claude-3"。
    ProviderSpec(
        name="aihubmix",
        keywords=("aihubmix",),
        env_key="OPENAI_API_KEY",  # 兼容 OpenAI
        display_name="AiHubMix",
        litellm_prefix="openai",  # 结果为 openai/{model}
        skip_prefixes=(),
        env_extras=(),
        is_gateway=True,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="aihubmix",
        default_api_base="https://aihubmix.com/v1",
        strip_model_prefix=True,  # 模型映射：anthropic/claude-3 → claude-3 → openai/claude-3
        model_overrides=(),
    ),
    # SiliconFlow：OpenAI-compatible 网关，模型名保留组织前缀。
    ProviderSpec(
        name="siliconflow",
        keywords=("siliconflow",),
        env_key="OPENAI_API_KEY",
        display_name="SiliconFlow",
        litellm_prefix="openai",
        skip_prefixes=(),
        env_extras=(),
        is_gateway=True,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="siliconflow",
        default_api_base="https://api.siliconflow.cn/v1",
        strip_model_prefix=False,
        model_overrides=(),
    ),
    # VolcEngine：OpenAI-compatible 网关。
    ProviderSpec(
        name="volcengine",
        keywords=("volcengine", "volces", "ark"),
        env_key="OPENAI_API_KEY",
        display_name="VolcEngine",
        litellm_prefix="volcengine",
        skip_prefixes=(),
        env_extras=(),
        is_gateway=True,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="volces",
        default_api_base="https://ark.cn-beijing.volces.com/api/v3",
        strip_model_prefix=False,
        model_overrides=(),
    ),
    # === 标准 Provider（按模型名关键字匹配）===============
    # Anthropic：LiteLLM 原生识别 "claude-*"，无需前缀。
    ProviderSpec(
        name="anthropic",
        keywords=("anthropic", "claude"),
        env_key="ANTHROPIC_API_KEY",
        display_name="Anthropic",
        litellm_prefix="",
        skip_prefixes=(),
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
        supports_prompt_caching=True,
        default_model="anthropic/claude-sonnet-5",
    ),
    # OpenAI：LiteLLM 原生识别 "gpt-*"，无需前缀。
    ProviderSpec(
        name="openai",
        keywords=("openai", "gpt"),
        env_key="OPENAI_API_KEY",
        display_name="OpenAI",
        litellm_prefix="",
        skip_prefixes=(),
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
        default_model="openai/gpt-5.5",
    ),
    # OpenAI Codex：使用 OAuth，而不是 API key。
    ProviderSpec(
        name="openai_codex",
        keywords=("openai-codex",),
        env_key="",  # 基于 OAuth，无 API key
        display_name="OpenAI Codex",
        litellm_prefix="",  # 不通过 LiteLLM 路由
        skip_prefixes=(),
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="codex",
        default_api_base="https://chatgpt.com/backend-api",
        strip_model_prefix=False,
        model_overrides=(),
        is_oauth=True,  # 基于 OAuth 的认证
        default_model="openai-codex/gpt-5-codex",
    ),
    # Github Copilot：使用 OAuth，而不是 API key。
    ProviderSpec(
        name="github_copilot",
        keywords=("github_copilot", "copilot"),
        env_key="",  # 基于 OAuth，无 API key
        display_name="Github Copilot",
        litellm_prefix="github_copilot",  # 模型映射：github_copilot/model → github_copilot/model
        skip_prefixes=("github_copilot/",),
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
        is_oauth=True,  # 基于 OAuth 的认证
        default_model="github_copilot/gpt-4o",
    ),
    # DeepSeek：LiteLLM 路由需要 "deepseek/" 前缀。
    ProviderSpec(
        name="deepseek",
        keywords=("deepseek",),
        env_key="DEEPSEEK_API_KEY",
        display_name="DeepSeek",
        litellm_prefix="deepseek",  # 模型映射：deepseek-chat → deepseek/deepseek-chat
        skip_prefixes=("deepseek/",),  # 避免重复添加前缀
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
        requires_reasoning_content_replay=True,
        default_model="deepseek/deepseek-v4-flash",
    ),
    # Gemini：LiteLLM 需要 "gemini/" 前缀。
    ProviderSpec(
        name="gemini",
        keywords=("gemini",),
        env_key="GEMINI_API_KEY",
        display_name="Gemini",
        litellm_prefix="gemini",  # 模型映射：gemini-pro → gemini/gemini-pro
        skip_prefixes=("gemini/",),  # 避免重复添加前缀
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
        default_model="gemini/gemini-2.5-flash",
    ),
    # 智谱：LiteLLM 使用 "zai/" 前缀。同时把 key 镜像到 ZHIPUAI_API_KEY，
    # 因为部分 LiteLLM 路径会检查它。skip_prefixes 用于已由网关路由时
    # 避免再次添加 "zai/"。
    ProviderSpec(
        name="zhipu",
        keywords=("zhipu", "glm", "zai"),
        env_key="ZAI_API_KEY",
        display_name="Zhipu AI",
        litellm_prefix="zai",  # 模型映射：glm-4 → zai/glm-4
        skip_prefixes=("zhipu/", "zai/", "openrouter/", "hosted_vllm/"),
        env_extras=(("ZHIPUAI_API_KEY", "{api_key}"),),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
        default_model="zai/glm-4.6",
    ),
    # DashScope：Qwen 模型需要 "dashscope/" 前缀。
    ProviderSpec(
        name="dashscope",
        keywords=("qwen", "dashscope"),
        env_key="DASHSCOPE_API_KEY",
        display_name="DashScope",
        litellm_prefix="dashscope",  # 模型映射：qwen-max → dashscope/qwen-max
        skip_prefixes=("dashscope/", "openrouter/"),
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
        default_model="dashscope/qwen-plus",
    ),
    # Moonshot：Kimi 模型需要 "moonshot/" 前缀。LiteLLM 通过
    # MOONSHOT_API_BASE 环境变量定位端点。Kimi K2.5 API 要求 temperature >= 1.0。
    ProviderSpec(
        name="moonshot",
        keywords=("moonshot", "kimi"),
        env_key="MOONSHOT_API_KEY",
        display_name="Moonshot",
        litellm_prefix="moonshot",  # 模型映射：kimi-k2.5 → moonshot/kimi-k2.5
        skip_prefixes=("moonshot/", "openrouter/"),
        env_extras=(("MOONSHOT_API_BASE", "{api_base}"),),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="https://api.moonshot.ai/v1",  # 国际站；中国区使用 api.moonshot.cn
        strip_model_prefix=False,
        model_overrides=(("kimi-k2.5", {"temperature": 1.0}),),
    ),
    # MiniMax：LiteLLM 路由需要 "minimax/" 前缀，使用
    # api.minimax.io/v1 上的 OpenAI-compatible API。
    ProviderSpec(
        name="minimax",
        keywords=("minimax",),
        env_key="MINIMAX_API_KEY",
        display_name="MiniMax",
        litellm_prefix="minimax",  # 模型映射：MiniMax-M2.1 → minimax/MiniMax-M2.1
        skip_prefixes=("minimax/", "openrouter/"),
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="https://api.minimax.io/v1",
        strip_model_prefix=False,
        model_overrides=(),
    ),
    # === 本地部署（按配置键匹配，不按 api_base）=========
    # vLLM / 任意 OpenAI-compatible 本地服务。配置键为 "vllm"
    # （provider_name="vllm"）时检测到。
    ProviderSpec(
        name="vllm",
        keywords=("vllm",),
        env_key="HOSTED_VLLM_API_KEY",
        display_name="vLLM/Local",
        litellm_prefix="hosted_vllm",  # 模型映射：Llama-3-8B → hosted_vllm/Llama-3-8B
        skip_prefixes=(),
        env_extras=(),
        is_gateway=False,
        is_local=True,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",  # 用户必须在配置中提供
        strip_model_prefix=False,
        model_overrides=(),
    ),
    # === Ollama（本地、OpenAI-compatible）===================================
    ProviderSpec(
        name="ollama",
        keywords=("ollama", "nemotron"),
        env_key="OLLAMA_API_KEY",
        display_name="Ollama",
        litellm_prefix="ollama_chat",  # 模型映射：model → ollama_chat/model
        skip_prefixes=("ollama/", "ollama_chat/"),
        env_extras=(),
        is_gateway=False,
        is_local=True,
        detect_by_key_prefix="",
        detect_by_base_keyword="11434",
        default_api_base="http://localhost:11434",
        strip_model_prefix=False,
        model_overrides=(),
    ),
    # === 辅助 Provider（不是主要 LLM Provider）============================
    # Groq：主要用于 Whisper 语音转写，也可用于 LLM。LiteLLM 路由需要
    # "groq/" 前缀。放在最后，因为它很少在 fallback 中胜出。
    ProviderSpec(
        name="groq",
        keywords=("groq",),
        env_key="GROQ_API_KEY",
        display_name="Groq",
        litellm_prefix="groq",  # 模型映射：llama3-8b-8192 → groq/llama3-8b-8192
        skip_prefixes=("groq/",),  # 避免重复添加前缀
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
        default_model="groq/openai/gpt-oss-120b",
    ),
)


# ---------------------------------------------------------------------------
# 查找辅助函数
# ---------------------------------------------------------------------------


def find_by_model(model: str) -> ProviderSpec | None:
    """按 Model-name Keyword Case-insensitive 匹配 Standard Provider。

    Gateway 与 Local Spec 会跳过，因为它们应由 Provider Name、``api_key``/``api_base`` Detection
    决定；否则
    任意可路由 Model Name 可能误判。按 PROVIDERS Order 返回首个 Keyword Hit，无匹配为 None。
    """
    model_lower = model.lower()
    model_normalized = model_lower.replace("-", "_")
    model_prefix = model_lower.split("/", 1)[0] if "/" in model_lower else ""
    normalized_prefix = model_prefix.replace("-", "_")
    std_specs = [s for s in PROVIDERS if not s.is_gateway and not s.is_local]

    # 显式 Provider 前缀优先，避免 `github-copilot/...codex` 匹配 openai_codex。
    for spec in std_specs:
        if model_prefix and normalized_prefix == spec.name:
            return spec

    for spec in std_specs:
        if any(kw in model_lower or kw.replace("-", "_") in model_normalized for kw in spec.keywords):
            return spec
    return None


def find_gateway(
    provider_name: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> ProviderSpec | None:
    """按明确优先级检测 Gateway 或 Local Provider。

    1）``provider_name`` 若直接映射 Gateway/Local，立即采用；2）匹配 ``api_key`` Prefix，例如
    ``"sk-or-"`` → OpenRouter；3）匹配 API Base Keyword，例如 URL 中 ``"aihubmix"`` → AiHubMix。

    Standard Provider 使用 Custom api_base（例如 Proxy 后的 DeepSeek）不会被误判为 vLLM，旧的
    Generic Fallback 已移除。输入空或无匹配返回 None，Secret 不记录。
    """
    # 1. 按配置键直接匹配。
    if provider_name:
        spec = find_by_name(provider_name)
        if spec and (spec.is_gateway or spec.is_local):
            return spec

    # 2. 按 api_key 前缀 / api_base 关键字自动检测。
    for spec in PROVIDERS:
        if spec.detect_by_key_prefix and api_key and api_key.startswith(spec.detect_by_key_prefix):
            return spec
        if spec.detect_by_base_keyword and api_base and spec.detect_by_base_keyword in api_base:
            return spec

    return None


def find_by_name(name: str) -> ProviderSpec | None:
    """按 Config Field Name 精确查找 ProviderSpec，例如 ``"dashscope"``。

    搜索遵循 Registry Order，但 Name 设计为唯一；不存在返回 None。函数不做 Display Name、Model
    Keyword 或 Alias Matching，Caller 应使用对应专用入口。
    """
    return next((spec for spec in PROVIDERS if spec.name == name), None)
