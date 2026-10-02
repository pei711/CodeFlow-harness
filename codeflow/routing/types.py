"""EcoClaw-style Model Router 各阶段共享的 Data Types。

这些类型把路由数据流分成四层：`ModelBenchmark` 保存远端评测事实，`RoutingProfile` 表达质量与
成本偏好，`ClassificationResult` 记录 Prompt 分类依据，`SelectionResult` 携带最终候选顺序。
Dataclass 只负责结构化传递数据，不执行网络请求、分数计算或模型调用；阅读路由代码时可用这些
类型判断每一步“已经知道什么”，避免把 Benchmark、分类相似度和最终执行结果混为一谈。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ── 基准数据 ───────────────────────────────────────────────────────────


@dataclass
class ModelTaskScore:
    task_id: str
    score: float  # 0 到 100 的百分比
    max_score: float


@dataclass
class ModelBenchmark:
    model: str  # PinchBench 模型 ID，例如 "anthropic/claude-sonnet-4"
    provider: str
    overall_score: float
    speed: float | None  # 平均执行时间（秒），由 PinchBench 记录
    cost: float  # 完整运行 23 项基准任务的总成本（美元）
    task_scores: list[ModelTaskScore]
    submission_id: str


# ── 路由配置档 ─────────────────────────────────────────────────────────

RoutingProfileName = Literal["best", "balanced", "eco"]


@dataclass(frozen=True)
class RoutingProfile:
    quality_weight: float
    cost_weight: float


# ── 分类 ───────────────────────────────────────────────────────────────

TaskCategory = Literal[
    "sanity",
    "calendar",
    "stock",
    "blog",
    "tool_use",
    "summary",
    "events",
    "email",
    "memory",
    "files",
    "workflow",
    "clawdhub",
    "skill_search",
    "image_gen",
    "humanizer",
    "daily_summary",
    "email_triage",
    "email_search",
    "market_research",
    "spreadsheet",
    "eli5_pdf",
    "comprehension",
    "second_brain",
]


@dataclass
class ClassificationResult:
    category: TaskCategory
    similarity: float


# ── 选择 ───────────────────────────────────────────────────────────────


@dataclass
class ModelScore:
    model: str
    provider: str
    task_score: float
    cost_score: float
    composite_score: float


@dataclass
class SelectionResult:
    primary: ModelScore
    fallbacks: list[ModelScore]
    category: TaskCategory
    profile: RoutingProfileName
