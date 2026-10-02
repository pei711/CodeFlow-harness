"""Model Selector，对应 EcoClaw 的 `selector.ts`。

Selector 接收已经分类的 Task 与 Benchmark Data，用 ``composite(quality, cost)`` 为每个合格模型
打分，再返回 Top Choice 与 2 个 Fallbacks。Quality 来自该任务的专项 Benchmark，Cost 会在当前
候选集合内归一化；Routing Profile 决定两者各占多大权重。

这里输出的是基于历史评测的相对排序，不执行模型，也不保证 Primary 在当前请求上一定优于
Fallback。缺少任务专项数据的模型通常被排除，只有所有模型都缺数据时才退回 Overall Score。
"""

from __future__ import annotations

from loguru import logger

from codeflow.routing.classifier import TASK_CATEGORIES
from codeflow.routing.fetcher import BenchmarkData
from codeflow.routing.profiles import ROUTING_PROFILES
from codeflow.routing.types import (
    ModelBenchmark,
    ModelScore,
    RoutingProfileName,
    SelectionResult,
    TaskCategory,
)

_warned_missing: set[str] = set()


def _get_task_score(model: ModelBenchmark, task_id: str) -> float | None:
    """返回模型在 ``task_id`` 上的 Score；没有数据时返回 `None`。

    `None` 表示该模型从未在此任务上完成 Benchmark，因而应从该任务的 Routing 中排除。不能直接用
    Overall Score 代替，因为那会把其他任务上的能力误表述成未经测试任务的能力。首次发现某个
    Model/Task 组合缺分时会记录 Warning，同一组合后续不重复刷日志。
    """
    for ts in model.task_scores:
        if ts.task_id == task_id:
            return ts.score
    key = f"{model.model}:{task_id}"
    if key not in _warned_missing:
        _warned_missing.add(key)
        logger.warning(
            "No task-specific score for '{}' on '{}' — excluded from routing for this task",
            task_id,
            model.model,
        )
    return None


def _normalize(value: float, vmin: float, vmax: float, invert: bool) -> float:
    if vmax == vmin:
        return 1.0
    norm = (value - vmin) / (vmax - vmin)
    return 1.0 - norm if invert else norm


def _score_model_by_overall(
    model: ModelBenchmark,
    profile_name: RoutingProfileName,
    cost_min: float,
    cost_max: float,
) -> ModelScore:
    """用模型的 Overall Benchmark Score 打分，仅供 Last-resort Fallback 使用。

    质量项取 `overall_score`，成本项在当前候选的 `cost_min` 到 `cost_max` 内反向归一化，再按照指定
    Routing Profile 合成 `ModelScore`。只有所有候选都没有 Task-specific Score 时才应调用本函数；
    它提供可用排序，但不能证明模型对当前类别有专项评测证据。
    """
    profile = ROUTING_PROFILES[profile_name]
    cost_score = _normalize(model.cost, cost_min, cost_max, invert=True)
    composite = profile.quality_weight * (model.overall_score / 100.0) + profile.cost_weight * cost_score
    return ModelScore(
        model=model.model,
        provider=model.provider,
        task_score=model.overall_score,
        cost_score=cost_score,
        composite_score=composite,
    )


def _score_model(
    model: ModelBenchmark,
    task_id: str,
    profile_name: RoutingProfileName,
    cost_min: float,
    cost_max: float,
) -> ModelScore | None:
    """返回模型在当前 Task 上的 `ModelScore`；缺少该任务数据时返回 `None`。

    函数将 Task Score 转成 0 到 1 的质量项，把 Cost 反向归一化为“越便宜越高分”，再依据
    `profile_name` 的 Quality/Cost 权重计算 Composite Score。`None` 是明确的证据缺失信号，调用方
    必须排除该模型，不能把它当成零分模型混入排名。
    """
    task_score = _get_task_score(model, task_id)
    if task_score is None:
        return None
    profile = ROUTING_PROFILES[profile_name]
    cost_score = _normalize(model.cost, cost_min, cost_max, invert=True)
    composite = profile.quality_weight * (task_score / 100.0) + profile.cost_weight * cost_score
    return ModelScore(
        model=model.model,
        provider=model.provider,
        task_score=task_score,
        cost_score=cost_score,
        composite_score=composite,
    )


def select_model(
    benchmark_data: BenchmarkData,
    category: TaskCategory,
    profile_name: RoutingProfileName,
) -> SelectionResult:
    """按给定 Category 与 Profile 返回 Primary + 2 Fallback Models。

    首先只保留 Cost 大于零且 Model ID 含 ``/`` 的合格模型；没有候选时抛出 `ValueError`。随后将
    `category` 映射为 PinchBench Task ID，在候选成本区间内为具有专项得分的模型计算 Composite
    Score 并降序排列。若所有候选都缺专项数据，则记录 Warning 并改用 Overall-score Ranking。

    返回的 `SelectionResult` 保留 Primary、至多两个 Runner-up、Category 与 Profile，供上层构造
    Model Chain 和记录路由依据。它只表示本次排序结果，不表示这些模型当前在线或调用已经成功。
    """
    # 过滤有效模型。
    models = [m for m in benchmark_data.values() if m.cost is not None and m.cost > 0 and "/" in m.model]
    if not models:
        raise ValueError("No eligible models in benchmark data")

    task_id = TASK_CATEGORIES[category]

    costs = [m.cost for m in models]
    cost_min = min(costs)
    cost_max = max(costs)

    scored_raw = [_score_model(m, task_id, profile_name, cost_min, cost_max) for m in models]
    scored = sorted(
        [s for s in scored_raw if s is not None],
        key=lambda s: s.composite_score,
        reverse=True,
    )
    if not scored:
        # 所有模型都缺少任务专项数据，回退到总分排序。
        logger.warning(
            "No models have task-specific data for '{}'; falling back to overall-score ranking",
            task_id,
        )
        scored = sorted(
            [_score_model_by_overall(m, profile_name, cost_min, cost_max) for m in models],
            key=lambda s: s.composite_score,
            reverse=True,
        )

    return SelectionResult(
        primary=scored[0],
        fallbacks=scored[1:3],
        category=category,
        profile=profile_name,
    )
