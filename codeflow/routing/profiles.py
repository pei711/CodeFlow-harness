"""定义 Routing Profiles，用固定权重表达 Quality 与 Cost 的 Trade-offs。

调用方使用 `best`、`balanced` 或 `eco` 这样的稳定名称选择策略，而不必直接拼装权重。`best` 几乎
只看质量，`balanced` 等量考虑质量和成本，`eco` 更重视节省成本；这些 Profile 只改变候选模型的
排序偏好，不改变模型的真实 Benchmark 数据，也不保证某次调用一定更便宜或效果更好。
"""

from codeflow.routing.types import RoutingProfile, RoutingProfileName

ROUTING_PROFILES: dict[RoutingProfileName, RoutingProfile] = {
    "best": RoutingProfile(quality_weight=0.99, cost_weight=0.01),
    "balanced": RoutingProfile(quality_weight=0.50, cost_weight=0.50),
    "eco": RoutingProfile(quality_weight=0.20, cost_weight=0.80),
}
