"""实现基于 LiteLLM 的 Multi-provider LLMProvider。

该 Adapter 用 Provider Registry 解析 Gateway、Model Prefix、Env、Capability 与 Override，把 OpenRouter、
Anthropic、OpenAI、Gemini、MiniMax 等统一为 Base Contract。它同时实现真实 Streaming、Usage/Cache
Token 归一化、Tool Call ID Compatibility、Provider-safe Message Replay 与 Structured Error；不在
代码中维护 Provider if-elif Chain。
"""

import hashlib
import os
import secrets
import string
import warnings
from collections.abc import AsyncIterator
from typing import Any

import json_repair
from loguru import logger

from codeflow.providers.base import LLMProvider, LLMResponse, StreamDelta, ToolCallRequest
from codeflow.providers.litellm_setup import import_litellm
from codeflow.providers.registry import find_by_model, find_gateway

litellm = import_litellm()
acompletion = litellm.acompletion

# LiteLLM 的异步日志 worker（LoggingWorker）会把队列绑定到单一 event loop。
# CodeFlow 每个 Turn 使用新 loop（每次调用 asyncio.run），所以下一个 Turn 会重置
# 队列，所有待处理的 ``Logging.async_*_handler`` 协程都会在未 await 的情况下
# 被丢弃。Python 随后打印 ``coroutine ... was never awaited`` RuntimeWarning，
# 并污染 Ink TUI 渲染。被丢弃的是 LiteLLM 自身的成功/失败日志回调，CodeFlow 不依赖
# 它们。过滤范围只限于 LiteLLM 的 ``Logging`` handler；宽泛的
# ``coroutine '.*'`` 模式也会掩盖 CodeFlow 自身真正的未 await 错误。
warnings.filterwarnings(
    "ignore",
    message=r"coroutine 'Logging\.async_.*' was never awaited",
    category=RuntimeWarning,
)

# 标准 chat-completion 消息字段。
_ALLOWED_MSG_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"})
_ANTHROPIC_EXTRA_KEYS = frozenset({"thinking_blocks"})
_ALNUM = string.ascii_letters + string.digits

# LiteLLM 默认给 OpenRouter 请求设置 X-Title="liteLLM" 和
# HTTP-Referer="https://litellm.ai"，这会让 openrouter.ai/apps 把流量归因给
# LiteLLM 而非 CodeFlow。这里显式覆盖默认值；用户提供的 extra_headers 优先级更高。
_OPENROUTER_ATTRIBUTION: dict[str, str] = {
    "HTTP-Referer": "https://github.com/pei711/codeflow-harness",
    "X-Title": "CodeFlow Agent Harness",
    "X-OpenRouter-Title": "CodeFlow Agent Harness",
    "X-OpenRouter-Categories": "cli-agent,personal-agent",
}


def _first_present(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _usage_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _normalize_usage_payload(usage: Any) -> dict[str, Any]:
    normalized = {
        name: value
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        if (value := _usage_field(usage, name)) is not None
    }
    prompt_details = _usage_field(usage, "prompt_tokens_details")
    completion_details = _usage_field(usage, "completion_tokens_details")
    cache_read = _first_present(
        _usage_field(usage, "prompt_cache_hit_tokens"),
        _usage_field(usage, "cache_read_input_tokens"),
        _usage_field(prompt_details, "cached_tokens"),
        _usage_field(usage, "_cache_read_input_tokens"),
    )
    cache_write = _first_present(
        _usage_field(usage, "cache_creation_input_tokens"),
        _usage_field(prompt_details, "cache_write_tokens"),
        _usage_field(prompt_details, "cache_creation_tokens"),
        _usage_field(usage, "_cache_creation_input_tokens"),
    )
    cache_miss = _usage_field(usage, "prompt_cache_miss_tokens")
    reasoning = _first_present(
        _usage_field(usage, "reasoning_tokens"),
        _usage_field(completion_details, "reasoning_tokens"),
    )
    for key, value in (
        ("cache_read_input_tokens", cache_read),
        ("cache_creation_input_tokens", cache_write),
        ("cache_miss_input_tokens", cache_miss),
        ("reasoning_tokens", reasoning),
    ):
        if value is not None:
            normalized[key] = value
    return normalized


def _short_tool_id() -> str:
    """生成所有 Provider（包括 Mistral）都接受的 9-char Alphanumeric Tool Call ID。

    每个字符用 `secrets.choice` 从 ASCII Letter/Digit 选择，避免 UUID Symbol 或过长 ID 被 Strict API
    拒绝。该 ID 只用于当前 Normalized Response 与后续 Tool Result 配对，不是 Security Token。
    """
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


class LiteLLMProvider(LLMProvider):
    """通过 LiteLLM 为多种 LLM Provider 提供统一 Chat 与 Streaming Interface。

    支持 OpenRouter、Anthropic、OpenAI、Gemini、MiniMax 等。Provider-specific Logic 全由
    ``providers/registry.py`` 的 Spec 驱动，Class 内不需要 if-elif Catalog。Constructor 检测 Gateway/
    Local、设置必要 Env 与 Attribution Header，配置 Extra Body、Transport Retry 与 Drop Params。

    Request 解析 Original/Resolved Model，清理 Message 与 Tool ID，按 Capability 应用 Cache Control；
    Response 统一 Content、Tool Calls、Reasoning、Thinking 与 Usage。Base chat_with_retry 继续拥有
    Same-model/Model-chain Recovery，LiteLLM 自身 Transport Retry 可独立配置。
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
        disable_auto_cache_control: bool = True,
        extra_body: dict[str, Any] | None = None,
        transport_num_retries: int | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        # CallEfficiency 负责在 Runtime 接缝显式重写请求。
        # False 仅为历史实验兼容而保留。
        self.disable_auto_cache_control = disable_auto_cache_control
        # Provider 专属请求体扩展会原样转发给 LiteLLM。常见用途是固定
        # OpenRouter 路由 affinity，以维持 prompt cache 热度，
        #   参数示例：extra_body={"provider": {"order": ["Anthropic"], "allow_fallbacks": False}}
        self.extra_body = extra_body or {}
        self.set_transport_num_retries(transport_num_retries)

        # 检测网关/本地部署。来自配置键的 provider_name 是主要信号；
        # api_key / api_base 仅用于自动检测回退。
        self._gateway = find_gateway(provider_name, api_key, api_base)
        if self._gateway and self._gateway.name == "openrouter":
            self.extra_headers = {**_OPENROUTER_ATTRIBUTION, **self.extra_headers}

        # 配置环境变量。
        if api_key:
            self._setup_env(api_key, api_base, default_model)

        if api_base:
            litellm.api_base = api_base

        # 移除 Provider 不支持的参数，例如 gpt-5 会拒绝部分参数。
        litellm.drop_params = True

    def set_transport_num_retries(
        self,
        num_retries: int | None,
    ) -> None:
        if num_retries is not None and (
            isinstance(num_retries, bool) or not isinstance(num_retries, int) or num_retries < 0
        ):
            raise ValueError("transport_num_retries must be a non-negative integer")
        self.transport_num_retries = num_retries

    def _setup_env(self, api_key: str, api_base: str | None, model: str) -> None:
        """根据已检测 Gateway 或 Model Provider 设置 LiteLLM 所需 Environment Variables。

        找不到 Spec 或 OAuth-only Spec 没有 env_key 时 no-op。Gateway/Local 会覆盖主 Env Key，确保
        当前显式 Config 生效；Standard Provider 使用 setdefault，避免破坏既有 Process Config。
        ``env_extras`` 中 ``{api_key}`` 与 ``{api_base}`` 会替换，Base 缺失时用 spec.default_api_base。
        方法持久修改 Process Environment，不能记录 Secret。
        """
        spec = self._gateway or find_by_model(model)
        if not spec:
            return
        if not spec.env_key:
            # 仅 OAuth/Provider 使用的 spec，例如 openai_codex。
            return

        # 网关/本地部署覆盖现有 env；标准 Provider 不覆盖。
        if self._gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)

        # 解析 env_extras 占位符：
        #   {api_key}  → 用户 API key
        #   {api_base} → 用户 api_base，缺失时回退到 spec.default_api_base
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key)
            resolved = resolved.replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)

    def _resolve_model(self, model: str) -> str:
        """按 Gateway 或 Provider Spec 为 Model Name 应用 Canonical LiteLLM Prefix。

        Gateway Mode 优先：可选 Strip Existing Provider Prefix，再添加 Gateway litellm_prefix；不会再
        应用模型自身 Provider Prefix。Standard Mode 先 Canonicalize Explicit Alias，再在不命中
        skip_prefixes 时添加 Spec Prefix。未知 Model 原样返回，函数不探测 Network。
        """
        if self._gateway:
            # 网关模式：应用网关前缀，跳过 Provider 专属前缀。
            prefix = self._gateway.litellm_prefix
            if self._gateway.strip_model_prefix:
                model = model.split("/")[-1]
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model

        # 标准模式：为已知 Provider 自动添加前缀。
        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            model = self._canonicalize_explicit_prefix(model, spec.name, spec.litellm_prefix)
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"

        return model

    @staticmethod
    def _canonicalize_explicit_prefix(model: str, spec_name: str, canonical_prefix: str) -> str:
        """把 ``github-copilot/...`` 等 Explicit Provider Prefix 规范为 Canonical Prefix。

        没有 Slash 时原样返回；首段转 Lowercase 并把 ``-`` 换 ``_`` 后必须等于 spec_name，才保留
        Remainder 并换成 canonical_prefix。其他显式 Prefix 不动，避免把跨 Provider Model 错误改名。
        """
        if "/" not in model:
            return model
        prefix, remainder = model.split("/", 1)
        if prefix.lower().replace("-", "_") != spec_name:
            return model
        return f"{canonical_prefix}/{remainder}"

    def _supports_cache_control(self, model: str) -> bool:
        """判断当前 Gateway/Provider 是否支持 Content Block 上的 ``cache_control``。

        Gateway 已识别时读取其 supports_prompt_caching；否则按 Model 查 Standard Spec。未知 Model
        返回 False。该 Capability 只表示协议支持，Actual Injection 还受 disable_auto_cache_control
        与 CallEfficiency Request Transform 控制。
        """
        if self._gateway is not None:
            return self._gateway.supports_prompt_caching
        spec = find_by_model(model)
        return spec is not None and spec.supports_prompt_caching

    def supports_explicit_cache_control(self, model: str) -> bool:
        return self._supports_cache_control(model)

    def _apply_cache_control(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """返回注入 ``cache_control`` 的 Message/Tool Copies，不修改 Caller Input。

        System Content 是 String 时转成 Single Text Block 并标 ephemeral；已是 Block List 时只 Copy
        List 并标最后一块。Tool List 存在时 Copy 并标最后一个 Definition。其他 Message Object 可
        复用原引用。Caller 必须先确认 Provider Capability；Empty Block List 不在此函数处理。
        """
        new_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg["content"]
                if isinstance(content, str):
                    new_content = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
                else:
                    new_content = list(content)
                    new_content[-1] = {**new_content[-1], "cache_control": {"type": "ephemeral"}}
                new_messages.append({**msg, "content": new_content})
            else:
                new_messages.append(msg)

        new_tools = tools
        if tools:
            new_tools = list(tools)
            new_tools[-1] = {**new_tools[-1], "cache_control": {"type": "ephemeral"}}

        return new_messages, new_tools

    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """把 Registry 中首个匹配的 Model-specific Parameter Override 写入 Request kwargs。

        Model Lowercase 后按 ProviderSpec.model_overrides 顺序做 Substring Match，例如 Kimi Temperature；
        命中即 kwargs.update 并返回。未知 Provider/无 Pattern no-op。Override 有意覆盖 Caller Default，
        因为它表达 API Hard Requirement。
        """
        model_lower = model.lower()
        spec = find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return

    @staticmethod
    def _extra_msg_keys(original_model: str, resolved_model: str) -> frozenset[str]:
        """返回 Request Sanitization 必须额外保留的 Provider-specific Message Keys。

        Original/Resolved Model 命中 Anthropic Spec、名称含 Claude 或 Prefix 为 anthropic 时保留
        ``thinking_blocks``；其他返回空 Set。双 Model 输入处理 Gateway Prefix 改写前后差异，避免
        Extended Thinking 在 Multi-turn Replay 中丢失。
        """
        spec = find_by_model(original_model) or find_by_model(resolved_model)
        if (
            (spec and spec.name == "anthropic")
            or "claude" in original_model.lower()
            or resolved_model.startswith("anthropic/")
        ):
            return _ANTHROPIC_EXTRA_KEYS
        return frozenset()

    @staticmethod
    def _requires_reasoning_content_replay(original_model: str, resolved_model: str) -> bool:
        spec = find_by_model(original_model) or find_by_model(resolved_model)
        return spec is not None and spec.requires_reasoning_content_replay

    @staticmethod
    def _normalize_tool_call_id(tool_call_id: Any) -> Any:
        """把 ``tool_call_id`` 规范为 Provider-safe 9-char Alphanumeric Form。

        非 String 原样返回；已经 9 位且 isalnum 时保持。其他 String 用 SHA-1 Hex 前 9 位做稳定
        Mapping，同一原 ID 在 Assistant Tool Call 与 Tool Result 间能一致缩短。它解决 Strict
        Provider Length/Charset Contract，不用于 Cryptographic Integrity。
        """
        if not isinstance(tool_call_id, str):
            return tool_call_id
        if len(tool_call_id) == 9 and tool_call_id.isalnum():
            return tool_call_id
        return hashlib.sha1(tool_call_id.encode()).hexdigest()[:9]

    @staticmethod
    def _sanitize_messages(
        messages: list[dict[str, Any]],
        extra_keys: frozenset[str] = frozenset(),
        require_reasoning_content_replay: bool = False,
    ) -> list[dict[str, Any]]:
        """移除 Non-standard Message Keys，并维护 Assistant/Tool Replay 的 Provider Contract。

        Allowed Set 是 Standard Keys 加 ``extra_keys``；Base Sanitizer 会确保 Assistant Content Key。
        对要求 Reasoning Replay 的 Provider，带 Tool Calls 但无 reasoning_content 的 Assistant 补空
        String。所有 Assistant ``tool_calls[].id`` 与 Tool ``tool_call_id`` 通过同一 id_map 同步缩短，
        否则 Strict Provider 会拒绝断裂关联。

        返回 Sanitized Copy，不修改 Session History；Empty Content 在 Caller 先由 Base Helper 清理。
        """
        allowed = _ALLOWED_MSG_KEYS | extra_keys
        sanitized = LLMProvider._sanitize_request_messages(messages, allowed)
        id_map: dict[str, str] = {}

        def map_id(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            return id_map.setdefault(value, LiteLLMProvider._normalize_tool_call_id(value))

        for clean in sanitized:
            if (
                require_reasoning_content_replay
                and clean.get("role") == "assistant"
                and clean.get("tool_calls")
                and clean.get("reasoning_content") is None
            ):
                clean["reasoning_content"] = ""

            # 缩短后保持 assistant tool_calls[].id 与 tool tool_call_id 同步，
            # 否则严格 Provider 会拒绝断裂的关联。
            if isinstance(clean.get("tool_calls"), list):
                normalized_tool_calls = []
                for tc in clean["tool_calls"]:
                    if not isinstance(tc, dict):
                        normalized_tool_calls.append(tc)
                        continue
                    tc_clean = dict(tc)
                    tc_clean["id"] = map_id(tc_clean.get("id"))
                    normalized_tool_calls.append(tc_clean)
                clean["tool_calls"] = normalized_tool_calls

            if clean.get("tool_call_id"):
                clean["tool_call_id"] = map_id(clean["tool_call_id"])
        return sanitized

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """通过 LiteLLM 发送 Single Chat Completion，并返回统一 LLMResponse。

        ``messages`` 是 role/content Dict List，``tools`` 使用 OpenAI Format，Model 可为例如
        ``anthropic/claude-sonnet-4-5``。方法解析 Model Prefix/Capability，按需应用 Cache Control，
        清 Empty/Non-standard Message，``max_tokens`` 最低 1，设置 600 秒 Endpoint Timeout，并加入
        Temperature、Reasoning、Tool Choice、API/Base/Header/Extra Body 与 Transport Retry。

        Model-specific Override 最后修正 Hard Requirement。LiteLLM Success 交 `_parse_response`；Live
        Exception 在退化成 String 前 `classify_error(e)`，返回 finish_reason=error，供 Base Retry/
        Fallback 使用。方法本身不 Sleep 或切 Model。
        """
        original_model = model or self.default_model
        model = self._resolve_model(original_model)
        extra_msg_keys = self._extra_msg_keys(original_model, model)
        require_reasoning_content_replay = self._requires_reasoning_content_replay(original_model, model)

        if self._supports_cache_control(original_model) and not self.disable_auto_cache_control:
            messages, tools = self._apply_cache_control(messages, tools)

        # max_tokens 下限为 1；负数或零会让 LiteLLM 以
        # "max_tokens must be at least 1" 拒绝请求。
        max_tokens = max(1, max_tokens)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._sanitize_messages(
                self._sanitize_empty_content(messages),
                extra_keys=extra_msg_keys,
                require_reasoning_content_replay=require_reasoning_content_replay,
            ),
            "max_tokens": max_tokens,
            "temperature": temperature,
            # 限制单次 LLM 调用时长，避免上游端点半关闭或从不发送 FIN 时 Agent
            # 永久悬挂。LiteLLM 默认 6000 秒，对交互式 Agent Loop 过长；600 秒
            # 足以覆盖缓慢的重推理生成，同时能让卡住的连接快速失败，供
            # chat_with_retry 重试或放弃。
            "timeout": 600,
        }

        # 应用模型专属覆盖，例如 kimi-k2.5 temperature。
        self._apply_model_overrides(model, kwargs)

        # 直接传入 api_key，比只依赖环境变量更可靠。
        if self.api_key:
            kwargs["api_key"] = self.api_key

        # 为自定义端点传入 api_base。
        if self.api_base:
            kwargs["api_base"] = self.api_base

        # 传入额外 header，例如 AiHubMix 的 APP-Code。
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        # 传入 Provider 专属 body 扩展，例如 OpenRouter 路由固定参数。
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs["drop_params"] = True

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if self.transport_num_retries is not None:
            kwargs["num_retries"] = self.transport_num_retries

        try:
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            # 把错误作为 content 返回以便平稳处理，但要在实时异常
            # （status_code + type）退化成字符串前完成分类；重试/fallback 层读取该结论。
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
                error_classification=self.classify_error(e),
            )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """提供与 ``chat()`` Signature 一致的真实 Streaming Counterpart。

        每个 Non-empty Chunk Yield 一个 StreamDelta，Caller 可透明替换 Provider。Existing ``chat()``
        NOT modified，Non-TUI Path（Channels/Cron 等）继续使用旧行为。Request Preparation 与 Chat
        保持 Model、Message、Cache、Credential、Tool 与 Override 规则一致，同时设置 ``stream=True``
        和 include_usage，让 Final Chunk 带 Token Snapshot。

        Raw Chunk 交 `_normalize_stream_chunk`。默认处理 OpenAI Shape；DashScope 等 Provider-specific
        Shape 应在真实 Smoke Evidence 后加入该 Hook。该方法不实施 Base Retry，因为 Partial Stream
        重试会重复已交付内容。
        """
        original_model = model or self.default_model
        model = self._resolve_model(original_model)
        extra_msg_keys = self._extra_msg_keys(original_model, model)
        require_reasoning_content_replay = self._requires_reasoning_content_replay(original_model, model)

        if self._supports_cache_control(original_model) and not self.disable_auto_cache_control:
            messages, tools = self._apply_cache_control(messages, tools)

        max_tokens = max(1, max_tokens)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._sanitize_messages(
                self._sanitize_empty_content(messages),
                extra_keys=extra_msg_keys,
                require_reasoning_content_replay=require_reasoning_content_replay,
            ),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            # OpenAI-compatible Provider 只有在显式请求 usage 时才发出末尾 usage
            # chunk；否则 stream 不含 token 计数，下游成本/上下文追踪会看到零。
            "stream_options": {"include_usage": True},
        }

        self._apply_model_overrides(model, kwargs)

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs["drop_params"] = True
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if self.transport_num_retries is not None:
            kwargs["num_retries"] = self.transport_num_retries

        response = await acompletion(**kwargs)
        async for chunk in response:
            delta = self._normalize_stream_chunk(chunk)
            if delta is not None:
                yield delta

    def _normalize_stream_chunk(self, chunk: Any) -> StreamDelta | None:
        """把 Raw Provider Chunk 归一为 `StreamDelta`，无有效 Payload 时返回 None。

        Default OpenAI Shape 读取 ``chunk.choices[0].delta.content``、``delta.tool_calls`` 与 Trailing
        ``chunk.usage``；Usage-only Final Chunk 也必须返回。Tool Delta 优先用 Pydantic v2 model_dump，
        不支持时手工 Snapshot Index/ID/Function，完整累积属于 Consumer。Reasoning 与 Model 同步保留。

        Qwen DashScope 等 Provider-specific Shape 必须在 Real-provider Smoke 后按 design.md §D4 与
        tasks.md T3.4 加显式 Branch，可按 ``self._gateway`` / ``find_by_model(...).name`` 选择。当前
        Attribute/Index Shape Error 返回 None，不让单个空 Chunk 打断 Stream。
        """
        try:
            usage = getattr(chunk, "usage", None)
            usage_dict: dict[str, Any] | None = None
            if usage is not None:
                usage_dict = _normalize_usage_payload(usage)
            choices = getattr(chunk, "choices", None)
            if not choices:
                if usage_dict is None:
                    return None
                return StreamDelta(
                    content=None,
                    usage=usage_dict,
                    model=getattr(chunk, "model", None) or None,
                )
            delta_obj = getattr(choices[0], "delta", None)
            if delta_obj is None:
                return None
            content = getattr(delta_obj, "content", None)
            tool_calls = getattr(delta_obj, "tool_calls", None)
            reasoning_content = getattr(delta_obj, "reasoning_content", None) or None

            tool_call_delta: dict[str, Any] | None = None
            if tool_calls:
                # 把原始 tool_call delta 作为字典快照列表暴露，供下游重组；
                # 此处有意只做轻量处理，完整工具调用累积属于消费者职责。
                serialized = []
                for tc in tool_calls:
                    try:
                        serialized.append(tc.model_dump())  # 使用 Pydantic v2 接口
                    except AttributeError:
                        serialized.append(
                            {
                                "index": getattr(tc, "index", None),
                                "id": getattr(tc, "id", None),
                                "function": {
                                    "name": getattr(getattr(tc, "function", None), "name", None),
                                    "arguments": getattr(getattr(tc, "function", None), "arguments", None),
                                },
                            }
                        )
                tool_call_delta = {"tool_calls": serialized}

            if content is None and tool_call_delta is None and usage_dict is None and reasoning_content is None:
                return None

            return StreamDelta(
                content=content,
                tool_call_delta=tool_call_delta,
                usage=usage_dict,
                reasoning_content=reasoning_content,
                model=getattr(chunk, "model", None) or None,
            )
        except (AttributeError, IndexError):
            return None

    def _parse_response(self, response: Any) -> LLMResponse:
        """把 LiteLLM Completion Response 解析为 CodeFlow Standard LLMResponse。

        部分 Provider（如 GitHub Copilot）把 Content 与 Tool Calls 拆到 Multiple Choices，方法会合并
        所有 Raw Tool Calls，并选择相应 Finish Reason/首个非空 Content。Arguments String 用
        json_repair 解析；每项换成 9-char ID，同时保留 Call/Function Provider-specific Fields。

        Usage 经 Cache/Reasoning-aware Normalizer，Message Reasoning/Thinking Blocks 原样携带，Actual
        Model 也写入 Response。函数假设至少一个 Choice；Transport/Shape Exception 由 Chat Boundary
        转 Structured Error。
        """
        choice = response.choices[0]
        message = choice.message
        content = message.content
        finish_reason = choice.finish_reason

        # 部分 Provider（如 GitHub Copilot）会把 content 和 tool_calls 拆到多个
        # choice 中；将其合并，避免丢失 tool_calls。
        raw_tool_calls = []
        for ch in response.choices:
            msg = ch.message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                raw_tool_calls.extend(msg.tool_calls)
                if ch.finish_reason in ("tool_calls", "stop"):
                    finish_reason = ch.finish_reason
            if not content and msg.content:
                content = msg.content

        if len(response.choices) > 1:
            logger.debug(
                "LiteLLM response has {} choices, merged {} tool_calls", len(response.choices), len(raw_tool_calls)
            )

        tool_calls = []
        for tc in raw_tool_calls:
            # 必要时从 JSON 字符串解析参数。
            args = tc.function.arguments
            if isinstance(args, str):
                args = json_repair.loads(args)

            provider_specific_fields = getattr(tc, "provider_specific_fields", None) or None
            function_provider_specific_fields = getattr(tc.function, "provider_specific_fields", None) or None

            tool_calls.append(
                ToolCallRequest(
                    id=_short_tool_id(),
                    name=tc.function.name,
                    arguments=args,
                    provider_specific_fields=provider_specific_fields,
                    function_provider_specific_fields=function_provider_specific_fields,
                )
            )

        usage = _normalize_usage_payload(response.usage) if getattr(response, "usage", None) else {}

        reasoning_content = getattr(message, "reasoning_content", None) or None
        thinking_blocks = getattr(message, "thinking_blocks", None) or None

        return LLMResponse(
            content=content,
            model=getattr(response, "model", None) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )

    def get_default_model(self) -> str:
        """返回 Constructor Config 的 Default Model Identifier。

        值尚未经过 Gateway/Provider Prefix Resolution，实际 Call 时 `_resolve_model` 再转换；方法不
        Import Catalog、不探测 Endpoint，也不验证 Credential。
        """
        return self.default_model
