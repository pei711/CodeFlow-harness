"""把 ``session.jsonl`` 压缩为约 10K-token diagnostic summary。

judge LLM client B3 在 ``trajectory_format="compressed"`` 时使用该 Package。compressor 完全
rule-based，不发起 LLM call，因此成本低且 deterministic。压缩结果服务 diagnosis context，
不是原始 trajectory 的无损副本，也不能单独证明 task 结果。
"""

from .trajectory import (
    CompressorConfig,
    Event,
    TrajectoryCompressor,
    estimate_tokens,
    load_session_jsonl,
)

__all__ = [
    "CompressorConfig",
    "Event",
    "TrajectoryCompressor",
    "estimate_tokens",
    "load_session_jsonl",
]
