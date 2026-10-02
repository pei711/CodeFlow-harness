"""Unit tests for ``LazyProvider`` -- defers the real (litellm-importing) build."""

import asyncio
import threading
import time

import pytest

from codeflow.providers.base import GenerationSettings
from codeflow.providers.lazy import LazyProvider


class _FakeProvider:
    def __init__(self) -> None:
        self.generation = GenerationSettings()

    def get_default_model(self) -> str:
        return "built-model"

    async def chat(self, *args, **kwargs) -> str:
        return "chat"

    async def chat_stream(self, *args, **kwargs):
        yield "delta"

    async def chat_with_retry(self, *args, **kwargs) -> str:
        return "retry"


def _lazy(calls: list) -> LazyProvider:
    def factory() -> _FakeProvider:
        calls.append(1)
        return _FakeProvider()

    return LazyProvider(factory, default_model="cfg-model", generation=GenerationSettings(temperature=0.5))


def test_construction_and_config_reads_do_not_build() -> None:
    calls: list = []
    lp = _lazy(calls)

    assert calls == []
    assert lp.get_default_model() == "cfg-model"
    assert lp.generation.temperature == 0.5
    assert calls == []


def test_first_chat_builds_and_memoizes() -> None:
    calls: list = []
    lp = _lazy(calls)

    assert asyncio.run(lp.chat([])) == "chat"
    assert calls == [1]
    assert asyncio.run(lp.chat([])) == "chat"
    assert calls == [1]


def test_chat_stream_and_retry_delegate() -> None:
    calls: list = []
    lp = _lazy(calls)

    async def _drain():
        return [d async for d in lp.chat_stream([])]

    assert asyncio.run(_drain()) == ["delta"]
    assert asyncio.run(lp.chat_with_retry([])) == "retry"
    assert calls == [1]


def test_built_is_thread_safe() -> None:
    calls: list = []

    def factory() -> _FakeProvider:
        time.sleep(0.05)
        calls.append(1)
        return _FakeProvider()

    lp = LazyProvider(factory, "cfg-model", GenerationSettings())
    threads = [threading.Thread(target=lp._built) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls == [1]


def test_prewarm_builds_in_background() -> None:
    built = threading.Event()

    def factory() -> _FakeProvider:
        built.set()
        return _FakeProvider()

    lp = LazyProvider(factory, "cfg-model", GenerationSettings())
    lp.prewarm()

    assert built.wait(timeout=2.0), "prewarm did not build the provider in the background"


def test_prewarm_swallows_build_error() -> None:
    def factory():
        raise RuntimeError("boom")

    lp = LazyProvider(factory, "cfg-model", GenerationSettings())
    lp.prewarm()
    time.sleep(0.05)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(lp.chat([]))
