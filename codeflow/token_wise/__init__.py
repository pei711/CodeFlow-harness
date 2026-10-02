"""保留 Historical TokenWise 的 Compatibility Surface。

Public API 包含：

- ``StrategyRegistry``：在 LLM Calls 前后串联 `TokenStrategy` Hooks；
- ``UsageTracker``：Strategy 1，记录每次调用的 Tokens 与 Cost；
- ``CacheOptimizer``：Strategy 2，安排 Anthropic ``cache_control`` 的位置；
- ``estimate_cost_usd``：指向 CallEfficiency 定价实现的 Compatibility Alias。

新的 Runtime Integration 使用 ``codeflow.call_efficiency``。这里继续导出旧名称，是为了让 Frozen
Benchmarks、Historical Schemas 与 Source-compatible Extensions 不必立刻迁移；兼容入口的存在不
表示 TokenWise 仍是新功能应依赖的主实现。
"""

from codeflow.token_wise.cache_optimizer import CacheOptimizer
from codeflow.token_wise.pricing import estimate_cost_usd
from codeflow.token_wise.registry import StrategyRegistry
from codeflow.token_wise.usage_tracker import UsageTracker

__all__ = [
    "CacheOptimizer",
    "StrategyRegistry",
    "UsageTracker",
    "estimate_cost_usd",
]
