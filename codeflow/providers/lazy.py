"""延迟到 First Model Call 才构建 Real LLM Provider。

Real Provider 会 import litellm，耗时约 ~2-7s；若在 ``AgentLoop`` Construction Eager 执行，即使
Tools/Skills/Memory 不需要它也会 Stall Startup。``LazyProvider`` 直接从 Config 回答 Call 前只会
读取的 ``get_default_model`` 与 ``generation``，Chat/Stream/Cache Capability 首次需要时才通过
Memoized Thread-safe Factory 建立真实实例。
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Callable
from typing import Any

from codeflow.providers.base import GenerationSettings, LLMProvider, LLMResponse, StreamDelta


class LazyProvider(LLMProvider):
    """在 First Chat-related Call 构建 Real Provider 的 Memoized Proxy。

    `_built` 使用 Double-check + Thread Lock，Prewarm Thread 与 Event-loop First Call Race 时只会创建
    一次实例。Chat、Stream、Retry 与 Cache Capability 都原样 Delegate；Default Model 与 Generation
    无需触发 Import。Factory Error 不缓存为假 Provider，会由实际 Call 明确暴露。
    """

    def __init__(
        self,
        factory: Callable[[], LLMProvider],
        default_model: str,
        generation: GenerationSettings,
    ):
        super().__init__()
        self._factory = factory
        self._default_model = default_model
        self.generation = generation
        self._provider: LLMProvider | None = None
        self._lock = threading.Lock()

    def _built(self) -> LLMProvider:
        if self._provider is None:
            with self._lock:
                if self._provider is None:
                    self._provider = self._factory()
        return self._provider

    def prewarm(self) -> None:
        """在 Daemon Thread 预建 Provider，把 ~2-7s litellm Import 隐藏在 Render/User Think-time 后。

        与 First Real Call Race 是安全的，因为 ``_built`` 有 Lock Guard。Prewarm Error 被吞掉，不在
        Background Thread 打断 Startup；First Call 会再次/正式 Surface Build Error。Daemon Thread
        不阻止 Process Exit，方法立即返回。
        """

        def _run() -> None:
            try:
                self._built()
            except Exception:
                pass

        threading.Thread(target=_run, name="litellm-prewarm", daemon=True).start()

    def get_default_model(self) -> str:
        return self._default_model

    def supports_explicit_cache_control(self, model: str) -> bool:
        return self._built().supports_explicit_cache_control(model)

    async def chat(self, *args: Any, **kwargs: Any) -> LLMResponse:
        return await self._built().chat(*args, **kwargs)

    async def chat_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[StreamDelta]:
        async for delta in self._built().chat_stream(*args, **kwargs):
            yield delta

    async def chat_with_retry(self, *args: Any, **kwargs: Any) -> LLMResponse:
        return await self._built().chat_with_retry(*args, **kwargs)
