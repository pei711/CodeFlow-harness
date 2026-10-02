"""为 :class:`MemoryBackend` Adapters 提供 Shared Contract Test Base Classes。

Plugin Author 在 Adapter Test Suite 中继承 :class:`MemoryBackendContractTests`，Override
:meth:`make_backend` 返回 Fresh Backend，Pytest 就会对它运行全部 Cross-adapter Assertions。这样“理论上
可扩展 → 实际能运行”的 Design Promise 才成为 CI 可强制的 Evidence。

选择 Base Class 而非 Fixture，是因为 Subclassing 让 Test Names 对 Runner 可见，例如
``test_recall_returns_memory_list``，Failure 也明确归属具体 Backend，如
``TestMem0Backend::test_recall_returns_memory_list``；Fixture-driven Approach 会把同一测试藏在 Parameter
Name 下。

Base Class **位于 Package 内而非 ``tests/``**，原因有二：Plugin Authors 安装 ``codeflow-harness`` 后需要从
``codeflow`` Import；Pytest 不会从 Non-test Package 自动 Collect，因此 Abstract Base 不会自行运行，只有
Concrete Subclasses 执行。Contract Pass 证明协议下界，不证明检索质量或 Production Durability。
"""

from __future__ import annotations

import pytest

from codeflow.memory_engine.backend import Memory, MemoryBackend


class MemoryBackendContractTests:
    """所有 Backend 共享的 Cross-adapter Contract Assertions。

    Concrete Subclass Override :meth:`make_backend` 构造 Fresh Backend，并可带 Temp Dirs、Fake HTTP
    Servers、In-memory Stores 等 Scaffolding。每个 Test 都通过 ``backend`` Fixture 获得独立实例，并在
    Start/Stop 生命周期内运行，避免 Cross-test State Leakage。
    """

    async def make_backend(self) -> MemoryBackend:
        """为一条 Test 构造 Fresh Backend。

        Subclasses **MUST Override**。Fixture 在构造后 Await ``start()``，Test 后 Await ``stop()``，所以
        Override *不需要*自行调用 Lifecycle。Base Implementation 抛出 `NotImplementedError`，防止抽象
        Contract 被误用。
        """
        raise NotImplementedError(
            "MemoryBackendContractTests subclass must override make_backend()",
        )

    @pytest.fixture
    async def backend(self):
        b = await self.make_backend()
        await b.start()
        try:
            yield b
        finally:
            await b.stop()

    async def test_satisfies_protocol(self, backend) -> None:
        """返回对象必须被 Runtime-checkable Protocol 识别为 `MemoryBackend`。

        这只检查 Surface Shape，Duck-typed Object 也能通过，不替代后续行为 Assertions。
        """
        assert isinstance(backend, MemoryBackend)

    async def test_recall_returns_memory_list(self, backend) -> None:
        """验证 ``recall`` 返回 ``list[Memory]``，Empty Result 是 OK。

        非空时逐项检查 `text`、`score`、`metadata` Types；不要求具体命中内容或最低数量。
        """
        hits = await backend.recall(
            "anything",
            user_id="contract-test",
            top_k=5,
        )
        assert isinstance(hits, list)
        for h in hits:
            assert isinstance(h, Memory)
            assert isinstance(h.text, str)
            assert isinstance(h.score, float)
            assert isinstance(h.metadata, dict)

    async def test_recall_after_store_does_not_raise(self, backend) -> None:
        """验证 ``store`` 后执行 ``recall`` 的 Basic Round-trip 不抛错。

        **不**断言 Just-stored Content 立即 Surface：许多 Backends Asynchronously Index，部分如 Mem0 需要
        Multi-turn Boundary 才完成 Extraction。Contract 下界只保证两次 Call Neither Raises，不能据此
        声称写后读一致或记忆召回成功。
        """
        await backend.store(
            "contract-session",
            [
                {"role": "user", "content": "I love Python"},
                {"role": "assistant", "content": "Noted."},
            ],
        )
        hits = await backend.recall(
            "programming",
            user_id="contract-test",
            top_k=5,
        )
        assert isinstance(hits, list)

    async def test_feedback_accepts_arbitrary_signals(self, backend) -> None:
        """验证 No-op Feedback 合法，任意 Dict 都应被 Tolerated。

        测试 Unknown、Empty 与 Skill-usage Shape，只要求不 Crash，不要求 Backend 产生学习效果。
        """
        await backend.feedback({"unknown_signal": "should not crash"})
        await backend.feedback({})
        await backend.feedback({"kind": "skill_usage", "ids": ["x", "y"]})

    async def test_top_k_respected_or_bounded(self, backend) -> None:
        """验证 ``top_k`` 是 Result Upper Bound，Backend 可以返回更少。

        Contract 防止 Adapter 忽略 Host Budget 返回无界 Hits，但不要求填满 Top-K。
        """
        hits = await backend.recall(
            "q",
            user_id="contract-test",
            top_k=3,
        )
        assert len(hits) <= 3

    async def test_recall_with_empty_owner_does_not_crash(
        self,
        backend,
    ) -> None:
        """验证 Unknown / Never-stored-for Owner 不会让 Recall Crash。

        合法行为通常是 Empty List；测试只约束返回 List，不要求 Backend 预先存在该 User。
        """
        hits = await backend.recall(
            "q",
            user_id="never-existed",
            top_k=5,
        )
        assert isinstance(hits, list)


class LifecycleContractTests:
    """直接验证 Raw ``start`` / ``stop`` Pair 的 Lifecycle Tests。

    它们 **不使用** ``backend`` Fixture，才能自行控制调用顺序。单独 Base Class 使 Subclass 只有在希望
    Assert Idempotence 时才继承这些 Tests，不把更强生命周期要求强加给所有 Contract Adapters。
    """

    async def make_backend(self) -> MemoryBackend:
        raise NotImplementedError

    async def test_start_stop_idempotent(self) -> None:
        b = await self.make_backend()
        await b.start()
        await b.stop()
        # 后端停止后再执行一轮不应抛出异常。
        await b.start()
        await b.stop()

    async def test_stop_without_start_does_not_raise(self) -> None:
        """防御性验证：``start`` Failed 或 Never Ran 的 Backend 仍能 Cleanly ``stop``。

        这保证 Host 在 Partial-init Failure 后可统一 Shutdown；正常返回不代表有资源曾被创建。
        """
        b = await self.make_backend()
        await b.stop()


__all__ = [
    "LifecycleContractTests",
    "MemoryBackendContractTests",
]
