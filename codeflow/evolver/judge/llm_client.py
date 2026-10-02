"""为 Judge 提供 pluggable LLM backend 与 mix routing。

Judge 有两类 workload：L1 detection 判断 trajectory 是否被 infrastructure bug 破坏，主要是
pattern recognition，cheap model 足够；L2/L3 patch proposal 要决定 file 与 change，需要更强
reasoning，值得调用 larger model。

:class:`JudgeLLMConfig` 支持 ``single``，由一个足够强的 backend 处理全部、适合 ablation；
``two_step`` 是 default mix，cheap L1 backend 先跑，判 L1 就返回并跳过 expensive call，否则
丢弃 cheap patch guess，改由 strong backend 重写；``pure_<name>`` 是 single 的配置 convenience，
两个 slot 使用同一 backend，例如 pure Qwen/OpenRouter。

:class:`LitellmBackend` 包装 ``codeflow.providers`` 的 ``LLMProvider``，包括 self-hosted Qwen-397B；
:class:`OpenRouterBackend` 直接 HTTPS 调用 ``openrouter.ai/api/v1``，访问 Claude/GPT/Gemini/
Qwen/DeepSeek；:class:`MockBackend` 返回 scripted response，提供 deterministic test。

configuration data-driven，使 production、test、ablation 走同一代码路径。backend call 成功只
得到 raw Judge text；parse 成功才有 schema result，而二者都不是 benchmark 正向 evidence。
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional

from .parser import JudgeParseError, parse_judge_output
from .prompts import build_judge_messages
from .schema import IssueType, JudgeResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 后端协议与实现
# ---------------------------------------------------------------------------


class JudgeLLMBackend(ABC):
    """Judge backend 必须实现的 minimum interface。

    Judge 不需要 streaming、Tool call 或 function-calling，只需返回 assistant text 的 plain chat
    completion。narrow interface 使 Vertex、Together、local vLLM endpoint 等新 backend 可在
    约 30 行内接入。实现拥有 stable ``name`` 供日志和 repr 使用。
    """

    name: str  # 由子类设置

    @abstractmethod
    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> str:
        """执行一次 chat completion，返回 assistant text body。

        implementation 应在 provider/network/shape 失败时抛出异常，不得把空或非 string 内容
        伪装成合法 Judge output。
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class LitellmBackend(JudgeLLMBackend):
    """委托 CodeFlow ``LLMProvider`` 的 Judge backend。

    主要用于已有 LiteLLM provider 路由的 self-hosted Qwen-397B，也支持 provider stack 的其他
    model。wrapped provider 必须提供 ``chat_with_retry(messages, ...)``，返回标准
    ``LLMResponse`` shape 且 ``.content`` 为 string；其他字段丢弃。调用走 base retry，使 Judge
    继承 empty-response retry 与 sync OpenAI fallback。
    """

    def __init__(
        self,
        provider: Any,  # codeflow.providers.base.LLMProvider，延迟确定类型
        *,
        model: Optional[str] = None,
        name: str = "litellm",
    ) -> None:
        self._provider = provider
        self._model = model
        self.name = name

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> str:
        # 经由 base.LLMProvider 的 chat_with_retry 路由，使评判器继承智能体循环已依赖的空响应
        # 重试和同步 OpenAI 回退。直接调用 provider.chat() 会在评判层暴露提供商空内容错误，
        # 早期干跑约占 3%。提供商的 chat[_with_retry] 接受 model=None 并回退到配置默认值；
        # 只有显式指定时才传入覆盖值。
        kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
        if self._model:
            kwargs["model"] = self._model
        response = await self._provider.chat_with_retry(messages, **kwargs)
        content = getattr(response, "content", None)
        if not isinstance(content, str):
            raise RuntimeError(
                f"LitellmBackend({self.name}): provider returned non-string content"
                f" ({type(content).__name__}); cannot parse as judge output."
            )
        return content


class OpenRouterBackend(JudgeLLMBackend):
    """直接调用 OpenRouter（``openrouter.ai``）的 HTTP backend。

    不经 litellm 是为了让 OpenRouter API key 与 CodeFlow main provider 分离，并独立核算 external-LLM
    budget，便于 hard cap 与 ablation 换模；HTTP shape 与 OpenAI compatible。model 使用
    namespaced form，例如 ``anthropic/claude-haiku-4-5``、``openai/gpt-4.1-mini``、
    ``google/gemini-2.5-flash``、``qwen/qwen3-235b``。

    key resolution 顺序是 explicit ``api_key`` -> ``api_key_env`` ->
    ``OPENROUTER_API_KEY``，call time 仍缺失则抛错。``httpx`` lazy import，使只用
    ``MockBackend`` 的 unit test 不依赖它。empty content 与 transient HTTP error 使用
    1/2/4 秒退避加 final attempt，共 4 次。
    """

    DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
    DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
    # 对空内容和暂时性 HTTP 错误重试。OpenRouterBackend 绕过 CodeFlow 提供商栈，无法直接获得
    # chat_with_retry，因此添加最小内部重试以保持一致。
    _RETRY_DELAYS = (1.0, 2.0, 4.0)

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout_seconds: float = 60.0,
        name: str = "openrouter",
    ) -> None:
        self._model = model
        self._explicit_key = api_key
        self._api_key_env = api_key_env or self.DEFAULT_API_KEY_ENV
        self._api_base = (api_base or self.DEFAULT_API_BASE).rstrip("/")
        self._timeout = timeout_seconds
        self.name = name

    def _resolve_api_key(self) -> str:
        if self._explicit_key:
            return self._explicit_key
        key = os.environ.get(self._api_key_env)
        if not key:
            raise RuntimeError(
                f"OpenRouterBackend({self.name}): no API key — pass api_key, or set env {self._api_key_env}"
            )
        return key

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> str:
        # 延迟导入：不访问 OpenRouter 的测试无需 httpx。
        import asyncio  # noqa: PLC0415

        import httpx  # noqa: PLC0415

        api_key = self._resolve_api_key()
        url = f"{self._api_base}/chat/completions"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # OpenRouter 建议标识调用方，以便进行路由分析：
            "HTTP-Referer": "https://github.com/pei711/codeflow-harness",
            "X-Title": "CodeFlow Harness Evolver",
        }

        # 对空内容和暂时性 HTTP 错误重试。LitellmBackend 通过 provider.chat_with_retry 自动
        # 获得重试；OpenRouter 绕过提供商栈，因此在此添加同样“检测空值 -> 退避 -> 重试”形态的
        # 最小原地重试。总尝试次数为 len(_RETRY_DELAYS) 加最后一次，共 4 次。
        last_exc: Exception | None = None
        last_data: dict[str, Any] | None = None
        for attempt, delay in enumerate(self._RETRY_DELAYS, start=1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                content = self._extract_content(data)
                if content and content.strip():
                    return content
                # 空内容——退避后重试。
                last_data = data
            except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
                last_exc = exc
            await asyncio.sleep(delay)

            # 最后一次尝试不再抑制异常。
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        content = self._extract_content(data)
        if content and content.strip():
            return content
        # 全部重试后仍为空，抛出异常让批处理记录错误。
        if last_exc is not None:
            raise RuntimeError(
                f"OpenRouterBackend({self.name}): empty content after "
                f"{len(self._RETRY_DELAYS) + 1} attempts; last exc: {last_exc!r}"
            ) from last_exc
        raise RuntimeError(
            f"OpenRouterBackend({self.name}): empty content after "
            f"{len(self._RETRY_DELAYS) + 1} attempts; last payload: "
            f"{(last_data or data)!r}"
        )

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> Optional[str]:
        """从 OpenAI-shaped response 读取 message content，shape 不符返回 ``None``。"""
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None


class MockBackend(JudgeLLMBackend):
    """test 使用的 in-memory scripted backend。

    ``responses`` 按调用顺序 pop；script 不足时抛出 ``IndexError``，fail loud 而不重复最后一项。
    ``calls`` 记录每次 messages/max_tokens/temperature，供断言次数与 payload。无 network I/O。
    """

    def __init__(self, responses: list[str], *, name: str = "mock") -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.name = name

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if not self._responses:
            raise IndexError(
                f"MockBackend({self.name}): no more scripted responses; received {len(self.calls)} call(s) total"
            )
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# 配置数据类与门面
# ---------------------------------------------------------------------------


Mode = Literal["single", "two_step"]
TrajectoryFormat = Literal["full", "compressed"]


@dataclass
class JudgeLLMConfig:
    """配置 :class:`JudgeLLM` routing、trajectory format 与 sampling 参数。

    backend slot 由外部注入，不直接放入 config，使 dataclass 保持 YAML/JSON-friendly，test 可直接
    注入 ``MockBackend``。L1 slot 默认看 ``compressed``，因为 empty/Docker/repetition signal 在
    约 10K summary 中可见；patch slot 默认看 ``full``，保留具体 Tool call 与 reasoning detail。

    caller 向 ``JudgeLLM.judge`` 提供 full text 与 optional compressed text；配置要求 compressed
    但未提供时，fallback 到 full 并在 debug_log 下告警，保证 correctness 但成本可能上升。
    :func:`build_judge_llm` 是从含 backend spec 的 YAML/JSON dict 装配 facade 的 integration point。
    """

    mode: Mode = "two_step"
    l1_trajectory_format: TrajectoryFormat = "compressed"
    patch_trajectory_format: TrajectoryFormat = "full"
    max_tokens: int = 4000
    temperature: float = 0.0
    # 诊断：记录每次调用的输入长度和所用后端。默认关闭，使测试标准输出保持安静。
    debug_log: bool = False


class JudgeLLM:
    """编排一次 Judge analysis 的 facade。

    single mode 直接构造 messages、调用 patch backend、parse 返回。default two-step 先用 L1
    backend；若判 infrastructure L1 立即返回、节省 expensive call；若判 L2/L3，丢弃 cheap
    model patch guess，调用 strong patch backend 重写。

    丢弃 cheap patch 是信任边界：bad patch 会浪费 downstream evaluation budget；cheap model
    的 L1 detection high-recall，但 patch low-precision。facade 拥有 backend/config reference，
    不缓存 Judge result。
    """

    def __init__(
        self,
        l1_backend: JudgeLLMBackend,
        patch_backend: JudgeLLMBackend,
        config: Optional[JudgeLLMConfig] = None,
    ) -> None:
        self._l1 = l1_backend
        self._patch = patch_backend
        self._config = config or JudgeLLMConfig()

    @property
    def config(self) -> JudgeLLMConfig:
        return self._config

    async def judge(
        self,
        trajectory_id: str,
        task_description: str,
        trajectory_text: str,
        trajectory_text_compressed: Optional[str] = None,
    ) -> JudgeResult:
        """对一段 trajectory 运行完整 Judge pipeline。

        ``trajectory_text`` 是 required full trajectory；``trajectory_text_compressed`` 是 optional
        约 10K-token ``agent debugger`` summary。每个 slot 按 ``*_trajectory_format`` 选择文本，
        missing compressed 时 fallback full。single 只调用 patch backend；two_step 最多两次 call。

        返回 validated ``JudgeResult``；backend/parse error 向上抛出。返回结果仍需 downstream
        evidence 验证，不能直接应用 patch。
        """
        l1_text = self._select_trajectory_text(
            self._config.l1_trajectory_format,
            trajectory_text,
            trajectory_text_compressed,
            slot="l1",
        )
        patch_text = self._select_trajectory_text(
            self._config.patch_trajectory_format,
            trajectory_text,
            trajectory_text_compressed,
            slot="patch",
        )

        patch_messages = build_judge_messages(
            trajectory_id=trajectory_id,
            task_description=task_description,
            trajectory_text=patch_text,
        )
        if self._config.mode == "single":
            return await self._call_and_parse(self._patch, patch_messages, trajectory_id)

        # 两步模式
        l1_messages = build_judge_messages(
            trajectory_id=trajectory_id,
            task_description=task_description,
            trajectory_text=l1_text,
        )
        l1_result = await self._call_and_parse(self._l1, l1_messages, trajectory_id)
        if l1_result.issue_type == IssueType.L1:
            if self._config.debug_log:
                logger.info(
                    "judge: %s → L1 detected by l1_backend, skipping patch_backend",
                    trajectory_id,
                )
            return l1_result
        if self._config.debug_log:
            logger.info(
                "judge: %s → %s detected, calling patch_backend",
                trajectory_id,
                l1_result.issue_type.value,
            )
        return await self._call_and_parse(self._patch, patch_messages, trajectory_id)

    def _select_trajectory_text(
        self,
        want: TrajectoryFormat,
        full: str,
        compressed: Optional[str],
        *,
        slot: str,
    ) -> str:
        """按 slot 配置选择 full/compressed trajectory，并提供 fallback。

        ``want="compressed"`` 但 compressed 为 ``None`` 时返回 full，使 Judge 继续运行；成本会
        上升但不因配置遗漏 crash，debug_log 开启时记录 warning。其他情况返回请求的文本。
        """
        if want == "compressed":
            if compressed is None:
                if self._config.debug_log:
                    logger.warning(
                        "judge[%s]: config wants compressed trajectory but none "
                        "provided; falling back to full (cost may be higher than "
                        "expected)",
                        slot,
                    )
                return full
            return compressed
        return full

    async def _call_and_parse(
        self,
        backend: JudgeLLMBackend,
        messages: list[dict[str, str]],
        trajectory_id: str,
    ) -> JudgeResult:
        raw = await backend.call(
            messages,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )
        try:
            return parse_judge_output(raw, expected_trajectory_id=trajectory_id)
        except JudgeParseError:
            logger.warning(
                "judge.parse failed on backend=%s, traj=%s; raw=%s",
                backend.name,
                trajectory_id,
                raw[:500],
            )
            raise

    def __repr__(self) -> str:
        return f"JudgeLLM(mode={self._config.mode!r}, l1={self._l1!r}, patch={self._patch!r})"


# ---------------------------------------------------------------------------
# 配置驱动构建器（yaml/json 的集成点）
# ---------------------------------------------------------------------------


def build_backend(spec: dict[str, Any]) -> JudgeLLMBackend:
    """从 config dict 构造一个 Judge backend。

    ``type="openrouter"`` 要求 model，可选 api_key/api_key_env/api_base/timeout/name；
    ``type="litellm"`` 要求已实例化 ``provider``，可选 model/name，不能从 pure dict 独立创建，
    通常复用 AgentLoop provider；``type="mock"`` 使用 response string list，仅供 test。
    unknown type 或 litellm 缺 provider 抛出 ``ValueError``。
    """
    backend_type = spec.get("type")
    if backend_type == "openrouter":
        return OpenRouterBackend(
            model=spec["model"],
            api_key=spec.get("api_key"),
            api_key_env=spec.get("api_key_env"),
            api_base=spec.get("api_base"),
            timeout_seconds=spec.get("timeout_seconds", 60.0),
            name=spec.get("name", "openrouter"),
        )
    if backend_type == "litellm":
        if "provider" not in spec:
            raise ValueError(
                "litellm backend spec requires a pre-built 'provider' object; cannot construct from pure dict"
            )
        return LitellmBackend(
            provider=spec["provider"],
            model=spec.get("model"),
            name=spec.get("name", "litellm"),
        )
    if backend_type == "mock":
        return MockBackend(
            responses=list(spec.get("responses", [])),
            name=spec.get("name", "mock"),
        )
    raise ValueError(f"unknown backend type {backend_type!r}; supported: openrouter, litellm, mock")


def build_judge_llm(spec: dict[str, Any]) -> JudgeLLM:
    """从单个 dict 装配 :class:`JudgeLLM`。

    Expected shape::

        {
          "mode": "two_step",          # or "single"
          "max_tokens": 4000,
          "temperature": 0.0,
          "debug_log": false,
          "l1_backend":    { ... build_backend spec ... },
          "patch_backend": { ... build_backend spec ... },
        }

    ``mode="single"`` 可省略 l1_backend，patch_backend 处理全部；即使同时提供 l1 也被忽略。
    ``two_step`` 必须同时有两个 spec。将两者设为相同 spec 就是 pure Qwen/OpenRouter pattern，
    无需 special mode。missing required spec 或 backend build failure 向上抛出。
    """
    mode = spec.get("mode", "two_step")
    config = JudgeLLMConfig(
        mode=mode,
        l1_trajectory_format=spec.get("l1_trajectory_format", "compressed"),
        patch_trajectory_format=spec.get("patch_trajectory_format", "full"),
        max_tokens=spec.get("max_tokens", 4000),
        temperature=spec.get("temperature", 0.0),
        debug_log=spec.get("debug_log", False),
    )

    patch_spec = spec.get("patch_backend")
    if patch_spec is None:
        raise ValueError("build_judge_llm: 'patch_backend' is required")
    patch_backend = build_backend(patch_spec)

    if mode == "two_step":
        l1_spec = spec.get("l1_backend")
        if l1_spec is None:
            raise ValueError("build_judge_llm: mode='two_step' requires 'l1_backend'")
        l1_backend = build_backend(l1_spec)
    else:
        # 单后端模式：在 l1 槽复用 patch_backend，实际不会调用。
        l1_backend = patch_backend

    return JudgeLLM(
        l1_backend=l1_backend,
        patch_backend=patch_backend,
        config=config,
    )


__all__ = [
    "JudgeLLMBackend",
    "LitellmBackend",
    "OpenRouterBackend",
    "MockBackend",
    "JudgeLLMConfig",
    "JudgeLLM",
    "Mode",
    "TrajectoryFormat",
    "build_backend",
    "build_judge_llm",
]
