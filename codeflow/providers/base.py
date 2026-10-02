"""定义所有 LLM Provider 共享的 Request、Response、Streaming、Error 与 Retry Contract。

具体 Provider 只负责供应商 API Shape；AgentLoop 统一依赖 `LLMProvider`、`LLMResponse`、
`StreamDelta` 与 `ToolCallRequest`。Base Layer 还规范 Empty Content、Provider-safe Message Keys、
结构化 Error Classification、Same-model Backoff 和 Cross-model Fallback，避免各实现用字符串猜测
失败或产生不一致终态。
"""

import asyncio
import json
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from codeflow.tracing import semconv, trace


@dataclass(frozen=True)
class ErrorClassification:
    """记录 Failed LLM Call 的 Structured Verdict，取代 Caller Substring Guessing。

    ``retryable`` 驱动 Backoff 后重试 Same Model；``should_fallback`` 表示 Different Model/Provider
    可能成功；``should_compress`` 专指 Context-window Overflow，应先 Shrink 再 Retry。``category``
    只用于 Logging/Telemetry 与稳定错误表达，不应单独决定 Recovery。Frozen Object 让一次失败
    在 Retry、Fallback 与 AgentLoop 间保持同一 Verdict。
    """

    category: str
    retryable: bool = False
    should_fallback: bool = False
    should_compress: bool = False


@dataclass
class ToolCallRequest:
    """承载 LLM 生成的一次规范化 Tool Call Request。

    ``id`` 关联 Tool Result/Event，``name`` 用于 Registry Dispatch，``arguments`` 是已解析 Object。
    两层 ``provider_specific_fields`` 分别保留 Tool Call 与 Function 的供应商扩展，使 History
    Round-trip 不必丢信息，同时核心执行只依赖统一字段。该对象不表示 Tool 已执行。
    """

    id: str
    name: str
    arguments: dict[str, Any]
    provider_specific_fields: dict[str, Any] | None = None
    function_provider_specific_fields: dict[str, Any] | None = None

    def to_openai_tool_call(self) -> dict[str, Any]:
        """序列化为 OpenAI-style ``tool_call`` Payload。

        Arguments 使用 `json.dumps(..., ensure_ascii=False)` 变成 Function Arguments String，保留
        CJK；Call Type 固定 ``function``。存在的 Provider-specific Fields 会放回原层级。返回新
        Dict，不修改 Request，也不验证 Argument Schema。
        """
        tool_call = {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }
        if self.provider_specific_fields:
            tool_call["provider_specific_fields"] = self.provider_specific_fields
        if self.function_provider_specific_fields:
            tool_call["function"]["provider_specific_fields"] = self.function_provider_specific_fields
        return tool_call


@dataclass
class LLMResponse:
    """统一表示一次 LLM Provider 调用完成后的完整 Response。

    ``content`` 可为空，Tool Calls 可与正文并存；``finish_reason`` 区分 stop/tool_calls/error，Usage、
    Reasoning 与 Anthropic Thinking Blocks 保留 Provider Evidence。Error Response 可附
    `ErrorClassification`，避免 Retry Layer 丢失 Live Exception 信息；``model``、Call Record、Cache
    Policy 支持 Accounting/Tracing。Response 只描述模型调用，不代表 Turn 或用户 Task 完成。
    """

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None  # Kimi、DeepSeek-R1 等
    thinking_blocks: list[dict] | None = None  # Anthropic 扩展思考内容
    # finish_reason == "error" 时设置。持有实时异常的 Provider 在此附加精确分类；
    # 否则由重试层根据错误字符串补全。
    error_classification: "ErrorClassification | None" = None
    model: str | None = None
    call_record: Any | None = None
    cache_policy: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        """返回 Response 是否包含至少一个规范化 Tool Call。

        该 Property 只检查 List Truthiness，不推断 finish_reason，也不验证 Name/Arguments。AgentLoop
        据此进入 Tool Execution Branch；空 List 即 False。
        """
        return bool(self.tool_calls)


@dataclass
class StreamDelta:
    """表示 Streaming LLM Response 的一个 Normalized Delta。

    Producer ``provider.chat_stream`` 为每个 non-empty chunk Yield 一项；AgentLoop ``on_token_delta``
    Path 与
    TUI SubscriptionEmitter 读取 ``.content`` 作为 Incremental Text。``tool_call_delta`` 携带 In-stream
    Tool Fragment，``usage`` 通常在 Final Chunk 给 Snapshot；Reasoning、Finish Reason、Classification、
    Model、Cache Policy 与 Call Record 允许流结束后重建 Shape-compatible LLMResponse。
    """

    content: str | None
    tool_call_delta: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    reasoning_content: str | None = None  # Kimi、DeepSeek-R1、qwen、o-series 的思考流
    finish_reason: str | None = None
    error_classification: ErrorClassification | None = None
    model: str | None = None
    cache_policy: str | None = None
    call_record: Any | None = None


@dataclass(frozen=True)
class GenerationSettings:
    """保存 LLM Call 的 Default Generation Parameters。

    Provider 持有 Temperature、``max_tokens``、``reasoning_effort``，使所有 Caller 无需逐层 Thread 同一
    Default；具体 Call 仍可向 ``chat()`` / ``chat_with_retry()`` 传 Explicit Keyword Override。
    Frozen Settings 是 Config Snapshot，不根据 Model Capability 自动修正参数。
    """

    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: str | None = None


class LLMProvider(ABC):
    """规定各 LLM Provider 在隐藏供应商 API 差异后必须提供的一致接口。

    Concrete Implementation 负责 Authentication、Endpoint、Payload 与 Response Parsing；Base Class
    提供 Request Sanitization、Streaming Fallback、Error Classification、Retry Ladder、Model Chain
    Fallback 与 Generation Default。AgentLoop 因而只处理统一 LLMResponse，不 import Provider SDK。

    `chat` 与 `get_default_model` 是必实现边界；未实现真实 Streaming 的 Provider 自动把完整 Chat
    Response 包成一个 Terminal Delta。Cancellation 始终向外传播，不能被 Retry 当普通 Error。
    """

    _CHAT_RETRY_DELAYS = (1, 2, 4)
    _SENTINEL = object()

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base
        self.generation: GenerationSettings = GenerationSettings()

    def supports_explicit_cache_control(self, model: str) -> bool:
        return False

    @staticmethod
    def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """替换会导致 Provider 400 的 Empty Text Content，同时保留 Tool-call Structure。

        MCP 等 Tool 可能返回 Nothing，多数 Provider 拒绝 Empty String 或 List 中空 Text Block。
        普通空内容替换为 ``(empty)``；带 Tool Calls 的 Assistant 必须用 ``None``，维持合法协议。
        List 会过滤空 text/input_text/output_text，Dict Content 规范为 Single-item List。返回新 List，
        没有需要清理的 Message 可复用原 Dict。
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")

            if isinstance(content, str) and not content:
                clean = dict(msg)
                clean["content"] = None if (msg.get("role") == "assistant" and msg.get("tool_calls")) else "(empty)"
                result.append(clean)
                continue

            if isinstance(content, list):
                filtered = [
                    item
                    for item in content
                    if not (
                        isinstance(item, dict)
                        and item.get("type") in ("text", "input_text", "output_text")
                        and not item.get("text")
                    )
                ]
                if len(filtered) != len(content):
                    clean = dict(msg)
                    if filtered:
                        clean["content"] = filtered
                    elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                        clean["content"] = None
                    else:
                        clean["content"] = "(empty)"
                    result.append(clean)
                    continue

            if isinstance(content, dict):
                clean = dict(msg)
                clean["content"] = [content]
                result.append(clean)
                continue

            result.append(msg)
        return result

    @staticmethod
    def _sanitize_request_messages(
        messages: list[dict[str, Any]],
        allowed_keys: frozenset[str],
    ) -> list[dict[str, Any]]:
        """只保留 ``allowed_keys`` 中的 Provider-safe Message Field，并规范 Assistant Content。

        每条消息投影为新 Dict；Assistant 若过滤后没有 ``content``，显式补 ``None``，避免 SDK 因
        Missing Key 拒绝 Tool-call Message。函数不清理 Empty String（由 `_sanitize_empty_content`
        负责），也不验证 Role Sequence。
        """
        sanitized = []
        for msg in messages:
            clean = {k: v for k, v in msg.items() if k in allowed_keys}
            if clean.get("role") == "assistant" and "content" not in clean:
                clean["content"] = None
            sanitized.append(clean)
        return sanitized

    @abstractmethod
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
        """发送一次 Chat Completion Request，并返回统一 `LLMResponse`。

        ``messages`` 是含 role/content 的 Provider-ready List；``tools`` 可带 Function Definitions，
        ``model`` 是 Provider-specific Identifier。``max_tokens``、``temperature`` 与可选
        ``reasoning_effort`` 控制生成，``tool_choice`` 支持 ``"auto"``、``"required"`` 或 Specific
        Tool Dict。实现应返回 Content、Tool Calls 或 Structured Error，不把供应商对象泄漏给 Caller。

        该 Abstract Method 只执行 Single Attempt；Retry/Fallback 由 `chat_with_retry` 统一所有。
        """
        pass

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
        """为无 Real Streaming 实现的 Provider 提供 Single Terminal Delta Fallback。

        TUI AgentLoop 始终通过 ``chat_stream`` 驱动 Turn；Custom-bespoke/Azure/Codex 若只有 ``chat``
        本会 AttributeError。本默认调用一次 ``chat()``，把全部 Content、Usage、Reasoning、Finish/Error
        信息放进一个 StreamDelta；ToolCallRequest 也转换为带 Index/Name/JSON Arguments 的 Delta。

        这只保证 Streaming Path 可用，不提供 Token-level Streaming。``LiteLLMProvider`` 覆盖为真实
        Incremental 实现，Caller 不应根据接口存在就假定逐 Token 到达。
        """
        response = await self.chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        tool_call_delta: dict[str, Any] | None = None
        if response.tool_calls:
            tool_call_delta = {
                "tool_calls": [
                    {
                        "index": i,
                        "id": tc.id,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for i, tc in enumerate(response.tool_calls)
                ]
            }
        yield StreamDelta(
            content=response.content,
            tool_call_delta=tool_call_delta,
            usage=response.usage or None,
            reasoning_content=response.reasoning_content,
            finish_reason=response.finish_reason,
            error_classification=response.error_classification,
            model=response.model,
        )

    @staticmethod
    def _extract_status_code(exc: BaseException | None) -> int | None:
        """沿 Exception Cause/Context Chain 查找第一个合法 HTTP Status Code。

        每层依次读取 status_code、http_status、code，只接受 100–599 Integer；Seen Object ID 防止
        Cycle。没有 Exception 或任何有效值时返回 ``None``。函数不 import Provider SDK，也不从
        Error Message 猜数字。
        """
        seen: set[int] = set()
        cur: BaseException | None = exc
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            for attr in ("status_code", "http_status", "code"):
                val = getattr(cur, attr, None)
                if isinstance(val, int) and 100 <= val < 600:
                    return val
            cur = cur.__cause__ or cur.__context__
        return None

    @staticmethod
    def _error_type_names(exc: BaseException | None) -> set[str]:
        """收集 Exception MRO 与 Cause Chain 上全部 Lowercased Class Names。

        这样无需 import Provider SDK，也能识别 RateLimitError、ContextWindowExceededError 等类型。
        Seen ID 防止 Cause Cycle，Result Set 同时包含 Base Class Name，供 Classification 做稳定匹配；
        空 Exception 返回空 Set。
        """
        names: set[str] = set()
        seen: set[int] = set()
        cur: BaseException | None = exc
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            for klass in type(cur).__mro__:
                names.add(klass.__name__.lower())
            cur = cur.__cause__ or cur.__context__
        return names

    @classmethod
    def classify_error(
        cls,
        exc: BaseException | None = None,
        content: str | None = None,
    ) -> ErrorClassification:
        """结合 Exception Type、HTTP Status 与 Message 分类 Failed LLM Call。

        有 Live Exception 时使用 Status/Class Name 获得精确 Verdict；Provider 已吞掉异常时退化为
        ``content`` Substring Matching。Order 是 Contract：Context Overflow 和 Rate Limit 必须先于
        Generic 400/Server Bucket，否则会选错 Recovery。

        Overflow 只 should_compress；429、Server 5xx、Network 可 Retry 且 Fallback；Auth 与普通
        Invalid Request Fatal；Billing/Model unavailable 只 Fallback；无法识别返回 unknown。函数不
        Sleep、不重试，也不把 Message 原文写入 Category。
        """
        status = cls._extract_status_code(exc)
        names = cls._error_type_names(exc)
        msg = (content if content is not None else str(exc) if exc is not None else "").lower()

        def has(*needles: str) -> bool:
            return any(n in msg for n in needles)

        # 上下文窗口溢出：压缩后重试，不做 fallback。更小窗口无济于事，压缩后的
        # 同一模型可以恢复。优先按异常类名检测，否则裸 400 看起来像 invalid_request。
        if "contextwindowexceedederror" in names or has(
            "context length",
            "context window",
            "maximum context",
            "too many tokens",
            "reduce the length",
        ):
            return ErrorClassification("context_overflow", should_compress=True)

        # 频率限制：等待并重试；其他 Provider 可能未被限流。
        if (
            status == 429
            or "ratelimiterror" in names
            or has(
                "rate limit",
                "429",
                "too many requests",
            )
        ):
            return ErrorClassification("rate_limit", retryable=True, should_fallback=True)

        # 暂时性服务/容量错误：重试并 fallback。
        if (
            status in (500, 502, 503, 504)
            or {"internalservererror", "serviceunavailableerror", "badgatewayerror"} & names
            or has(
                "overloaded",
                "server error",
                "service unavailable",
                "temporarily unavailable",
                "500",
                "502",
                "503",
                "504",
            )
        ):
            return ErrorClassification("server", retryable=True, should_fallback=True)

        # 超时/连接错误：重试并 fallback。
        if {"timeout", "apitimeouterror", "apiconnectionerror", "oserror"} & names or has(
            "timeout",
            "timed out",
            "connection",
        ):
            return ErrorClassification("network", retryable=True, should_fallback=True)

        # 认证/权限错误：属于致命配置问题，重试和 fallback 都无法修复。
        if (
            status in (401, 403)
            or {"authenticationerror", "permissiondeniederror"} & names
            or has(
                "unauthorized",
                "invalid api key",
                "permission denied",
            )
        ):
            return ErrorClassification("auth")

        # 账单/配额错误：同一模型无法恢复，其他 Provider 可能可用。
        if status == 402 or has(
            "billing",
            "quota",
            "insufficient",
            "credit",
            "payment",
            "exceeded your current",
        ):
            return ErrorClassification("billing", should_fallback=True)

        # 模型不可用/不存在：重试没有意义，尝试其他模型。
        if (
            status == 404
            or "notfounderror" in names
            or has(
                "model not found",
                "does not exist",
                "no endpoints",
                "not available",
                "unavailable",
            )
        ):
            return ErrorClassification("model_unavailable", should_fallback=True)

        # 普通 bad request（非上下文 400）：致命错误，切换模型也无济于事。
        if status == 400 or "badrequesterror" in names or has("invalid request", "invalid_request"):
            return ErrorClassification("invalid_request")

        return ErrorClassification("unknown")

    @classmethod
    def _is_transient_error(cls, content: str | None) -> bool:
        """提供旧 Caller 所需的 Back-compat Transient Error Boolean。

        它只把 String Content 交给 `classify_error` 并返回 ``retryable`` Verdict；新代码应直接消费
        ErrorClassification，避免再次维护 Substring Rule。
        """
        return cls.classify_error(content=content).retryable

    @classmethod
    def _should_fallback(cls, content: str | None) -> bool:
        """提供旧 Caller 所需的 Back-compat Fallback Boolean。

        String Content 经统一 Classifier 后返回 ``should_fallback``。该 Shim 不包含 Model Chain 或
        Remaining Model 判断，只保持旧接口行为。
        """
        return cls.classify_error(content=content).should_fallback

    @staticmethod
    def _jittered(delay: float) -> float:
        """给 Backoff Delay 应用 ±10% Random Jitter，避免多个 Caller 同步重试。

        ``delay <= 0`` 精确返回 0；正值乘 0.9–1.1 Uniform Factor。函数不 Sleep，也不改变 Retry
        Count；Test 可 Patch Random 以获得确定结果。
        """
        if delay <= 0:
            return 0.0
        return delay * random.uniform(0.9, 1.1)

    async def _chat_attempt_with_retry(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: object,
        temperature: object,
        reasoning_effort: object,
        tool_choice: str | dict[str, Any] | None,
        response_observer: Callable[[LLMResponse, str | None], Awaitable[None]] | None = None,
        attempt_started: Callable[[str | None], None] | None = None,
    ) -> LLMResponse:
        """让 Single Model 走完整 Retry Ladder，并为每次 Failure 建立 Classification。

        总 Attempt 数是 ``len(_CHAT_RETRY_DELAYS)`` 个可 Sleep Retry + 1 个 Final no-sleep Attempt。
        只有 ``retryable`` Error 用 Jittered Backoff 重试；Non-retryable 或最后一次立即返回。Thrown
        Exception（除 Cancel）转为 finish_reason=error 的 LLMResponse，Response Model 缺失时补当前值。

        Provider Attached Classification 优先，其次 Live Exception，最后 String。每次 Response 可交
        Observer，Attempt Start 可通知 Accounting。最终 Error 必带 ``error_classification``，让外层
        Model-chain Fallback 无需 Reclassify。
        """
        total_attempts = len(self._CHAT_RETRY_DELAYS) + 1
        last_response: LLMResponse | None = None
        for attempt in range(1, total_attempts + 1):
            exc: Exception | None = None
            try:
                if attempt_started is not None:
                    attempt_started(model)
                response = await self.chat(
                    messages=messages,
                    tools=tools,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    tool_choice=tool_choice,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                exc = e
                response = LLMResponse(content=f"Error calling LLM: {e}", finish_reason="error")

            if response.model is None:
                response.model = model

            if response.finish_reason != "error":
                if response_observer is not None:
                    await response_observer(response, model)
                return response

            # 优先使用 Provider 附加的分类，因为它持有实时异常；其次分类捕获到的异常，
            # 最后才根据字符串分类。
            classification = response.error_classification or self.classify_error(exc, response.content)
            response.error_classification = classification
            if response_observer is not None:
                await response_observer(response, model)
            last_response = response

            if not classification.retryable or attempt == total_attempts:
                return response

            delay = self._jittered(self._CHAT_RETRY_DELAYS[attempt - 1])
            logger.warning(
                "LLM error [{}] (attempt {}/{}) model={}, retrying in {:.1f}s: {}",
                classification.category,
                attempt,
                total_attempts,
                model,
                delay,
                (response.content or "")[:120],
            )
            await asyncio.sleep(delay)

        return last_response  # type: ignore[return-value]  # 循环总会在最后一次尝试时返回

    @trace.instrument("llm.call", extract=semconv.llm_call)
    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = _SENTINEL,
        temperature: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        fallback_models: list[str] | None = None,
        request_transform: Callable[
            [list[dict[str, Any]], list[dict[str, Any]] | None, str | None],
            tuple[list[dict[str, Any]], list[dict[str, Any]] | None, str | None],
        ]
        | None = None,
        response_observer: Callable[[LLMResponse, str | None], Awaitable[None]] | None = None,
        attempt_started: Callable[[str | None], None] | None = None,
    ) -> LLMResponse:
        """先对 ``chat()`` 的 Transient Failure 重试，再按配置 Model Chain Fallback。

        ``[model, *fallback_models]`` 每个 Model 都走完整 Retry Ladder。只有当前模型耗尽且
        ``error_classification.should_fallback``、同时仍有 Next Model 时才继续；否则 Error 原样向
        Caller 暴露。Fallback List 为空时与旧 Single-model Retry Byte-for-behavior 一致。

        未显式传 max_tokens/temperature/reasoning_effort 时读取 ``self.generation``，Caller 无需逐层
        Thread Default。可选 request_transform 对每个 Model 生成实际 Messages/Tools/Model；Observer
        与 Attempt Callback 传入内层。成功立即返回，全部失败返回最后 Structured Error。
        """
        if max_tokens is self._SENTINEL:
            max_tokens = self.generation.max_tokens
        if temperature is self._SENTINEL:
            temperature = self.generation.temperature
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.generation.reasoning_effort

        model_chain = [model, *(fallback_models or [])]
        response: LLMResponse | None = None
        for idx, current_model in enumerate(model_chain):
            attempt_messages = messages
            attempt_tools = tools
            attempt_model = current_model
            if request_transform is not None:
                attempt_messages, attempt_tools, attempt_model = request_transform(
                    messages,
                    tools,
                    current_model,
                )
            response = await self._chat_attempt_with_retry(
                messages=attempt_messages,
                tools=attempt_tools,
                model=attempt_model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
                response_observer=response_observer,
                attempt_started=attempt_started,
            )
            if response.model is None:
                response.model = attempt_model
            if response.finish_reason != "error":
                return response

            classification = response.error_classification or self.classify_error(content=response.content)
            has_next = idx + 1 < len(model_chain)
            if has_next and classification.should_fallback:
                next_model = model_chain[idx + 1]
                logger.warning(
                    "LLM call failed on model={} [{}], falling back to {}: {}",
                    attempt_model,
                    classification.category,
                    next_model,
                    (response.content or "")[:120],
                )
                continue
            return response

        return response  # type: ignore[return-value]  # 调用链始终非空

    @abstractmethod
    def get_default_model(self) -> str:
        """返回该 Provider 在 Caller 未指定时使用的 Stable Default Model Identifier。

        Concrete Provider 必须实现并使用其 API 可接受名称。该方法不探测 Network、不保证 Model
        当前 Available；Unavailable Error 仍由 Classification/Fallback 处理。
        """
        pass
