"""把 Eval Engine Verdicts 写回 MemoryEngine 长期记录。

每个被 Judge 的 Turn 通过 MemoryEngine Facade 写成一行 ``HISTORY.md`` Entry。等 ``MemoryEngine``
未来提供专用 ``write_observations`` Method 时，Adapter 可切换过去；当前使用 ``append_history``，在不
发明新 File Types 的前提下保持 Audit Trail 可见。

Adapter 刻意保持 Thin Shim：除调用 MemoryEngine 外不自行执行 IO。Verdict Mapping 与 Formatting 等
Semantic Decisions 集中在这里，让 Hook 只关心何时触发。当前代码已把依赖诚实地收窄到实际提供
`append_history` 的 `MemoryStore`，但保留上述 Write-back Contract。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

from codeflow.eval_engine.judge.judge import JudgeVerdict
from codeflow.memory_engine.consolidate.consolidator import MemoryStore

logger = logging.getLogger(__name__)


class EvalAdapter:
    """通过 :class:`MemoryStore` 把 Judge Verdicts 写入长期 ``HISTORY.md``。

    Phase B-3 将依赖从已删除的 ``MemoryEngine`` Facade 直接改向 `MemoryStore`。Adapter 唯一需要的
    Method 是 ``append_history``，本来就属于 `MemoryStore`；移除 Facade Indirection 只是让真实依赖
    更诚实。实例保存 Memory Store 与可注入 Clock，便于测试稳定时间格式。
    """

    def __init__(
        self,
        memory: MemoryStore,
        *,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._memory = memory
        self._now_fn = now_fn or datetime.now

    def record_task_completion(
        self,
        verdict: JudgeVerdict,
        user_goal: str,
        session_key: str,
    ) -> None:
        """向 `HISTORY.md` 追加一行 ``[YYYY-MM-DD HH:MM] eval verdict=...`` Entry。

        ``unknown`` Verdicts 刻意 **不记录**，因为它们只是 Signal Noise；``completed`` 与 ``failed``
        用于保留 Recent Outcomes。User Goal 只取第一行并截到 160 Characters，便于从文件尾部 Grep，
        同时写入 Session Key 便于关联。

        `append_history` 失败会记录 Debug 并吞掉，Adapter 不能让 AgentLoop 崩溃。因而方法正常返回不
        保证 Verdict 已持久化，Audit Trail 的完整性需另行检查。
        """
        if verdict is JudgeVerdict.unknown:
            return
        timestamp = self._now_fn().strftime("%Y-%m-%d %H:%M")
        # 保持目标简短，便于 grep 检索 HISTORY.md 尾部。
        truncated_goal = (user_goal or "").strip().splitlines()[0][:160]
        entry = f'[{timestamp}] eval verdict={verdict.value} session={session_key} goal="{truncated_goal}"'
        try:
            self._memory.append_history(entry)
        except Exception as exc:  # noqa: BLE001 — adapter 不得导致 AgentLoop 崩溃
            logger.debug(
                "EvalAdapter.append_history failed (%s): %s",
                type(exc).__name__,
                exc,
            )


__all__ = ["EvalAdapter"]
