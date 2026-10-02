"""构建 Trial pool，连接 trial directory 与 analysis modules。

模块把 :func:`codeflow.evolver.analysis.stability_bucket.compute_stability` 的 per-task stability，
和 :func:`codeflow.evolver.analysis.proxy_features.extract_trial_dir` 的 per-trial feature，组合为
:class:`codeflow.evolver.scheduler.cold_start_bandit.ColdStartCoverageBandit` 消费的统一
:class:`Trial` view。

k=3 paired baseline 的目录通常映射为 89 task x 3 attempt，约 267 个 Trial。同一 task 的
attempt 共享 task-level ``stability``，但各自携带 per-trial ``proxy_features``；feature 被投影
为 numeric-only dict，使 K-means sub-strata 可直接计算 Euclidean distance。

categorical ``ExitStatus`` 通过稳定 ordinal table ``_EXIT_STATUS_ORDINAL`` 映射，确保不同
Python run 的 K-means distance deterministic；built-in ``hash()`` 受 ``PYTHONHASHSEED``
影响，不适合 paper reproducibility。该 ordinal distance 本身没有因果或语义次序含义。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from codeflow.evolver.analysis.proxy_features import (
    ExitStatus,
    ProxyFeatures,
    extract_trial_dir,
)
from codeflow.evolver.analysis.stability_bucket import (
    compute_stability,
)
from codeflow.evolver.scheduler.cold_start_bandit import Trial

_EXIT_STATUS_ORDINAL: dict[ExitStatus, int] = {
    ExitStatus.PASSED: 0,
    ExitStatus.FAILED_VERIFIER: 1,
    ExitStatus.AGENT_TIMEOUT: 2,
    ExitStatus.VERIFIER_TIMEOUT: 3,
    ExitStatus.REWARD_FILE_NOT_FOUND: 4,
    ExitStatus.RUNTIME_ERROR: 5,
    ExitStatus.NO_SESSION: 6,
    ExitStatus.OTHER: 7,
}


def proxy_features_to_kmeans_dict(pf: ProxyFeatures) -> dict[str, float]:
    """把 :class:`ProxyFeatures` 投影为 bandit K-means 使用的 numeric dict。

    所有值转为 float，使 ``cold_start_bandit._kmeans_strata`` 能统一 min-max normalize。
    ``has_tool_calls_ever`` 映射为 0/1，``exit_status_ordinal`` 通过
    :data:`_EXIT_STATUS_ORDINAL` 转换 categorical :class:`ExitStatus`。Enum 间 distance 不具
    semantic meaning，但映射稳定，可通过 centroid co-occurrence 表达“行为相似”。
    """
    return {
        "turn_count": float(pf.turn_count),
        "has_tool_calls_ever": 1.0 if pf.has_tool_calls_ever else 0.0,
        "assistant_text_length_avg": float(pf.assistant_text_length_avg),
        "docker_error_count": float(pf.docker_error_count),
        "exit_status_ordinal": float(_EXIT_STATUS_ORDINAL.get(pf.final_exit_status, 99)),
    }


def build_trial_pool(trial_dir: str | Path) -> list[Trial]:
    """从 legacy-runner trial dir 构造 :class:`Trial` list。

    函数通过既有 analysis module 读取 stability/features，按 task 分组并用 trial_id 排序，分配
    deterministic attempt index ``1..k``；随后为每个 Trial 组合 task-level bucket、是否 pass
    与 per-trial K-means dict。两个 reader 理论上覆盖同一目录；缺少 stability 的 edge case
    保守跳过。

    返回按 ``(task_id, attempt)`` 排序，便于 repr；
    ``ColdStartCoverageBandit`` 仍会在自己的 RNG seed 下 shuffle borderline pool。构建结果只
    是抽样输入，不代表 cohort 已选择或 coverage gate 通过。
    """
    trial_dir = Path(trial_dir)
    stability_map = compute_stability(trial_dir)
    features_map = extract_trial_dir(trial_dir)

    # 按任务分组 ProxyFeatures，以便给尝试编号。
    by_task: dict[str, list[ProxyFeatures]] = defaultdict(list)
    for pf in features_map.values():
        by_task[pf.task_id].append(pf)
    for tid in by_task:
        by_task[tid].sort(key=lambda p: p.trial_id)

    trials: list[Trial] = []
    for task_id in sorted(by_task.keys()):
        task_stability = stability_map.get(task_id)
        if task_stability is None:
            # 防御性处理：两个模块遍历同一目录，理论上不应发生；边界情况下静默跳过而非崩溃。
            continue
        for attempt_idx, pf in enumerate(by_task[task_id], start=1):
            passed = pf.final_exit_status == ExitStatus.PASSED
            trials.append(
                Trial(
                    trial_id=pf.trial_id,
                    task_id=task_id,
                    attempt=attempt_idx,
                    passed=passed,
                    stability=task_stability.bucket,
                    proxy_features=proxy_features_to_kmeans_dict(pf),
                )
            )
    return trials


__all__ = [
    "build_trial_pool",
    "proxy_features_to_kmeans_dict",
]
