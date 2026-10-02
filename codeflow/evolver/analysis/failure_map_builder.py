"""把 Judge result 聚合为 ``failure_map.json``。

输入是 :class:`JudgeResult` list，通常来自 cold-start coverage bandit 的
``claude_judge(trial)``；输出是一份结构化 aggregation，是 spec §14 step ③ cold-start
coverage 与 step ④ first evolution round 之间的桥。它承担四项职责：检查 judge 是否覆盖至少
``min_why_classes=7`` 个 WHY pathology axis；把 L1 alert 暴露为
``human_review_needed``，让 Evolver 暂停、engineer 修 infrastructure；把 L2/L3 proposal 按
``(PatchWhere, PatchWhy)`` cell 聚合；计算 WHERE/WHY marginal，供 paper §15 Must-nail #1
diversity plot 使用。

JSON layout（``schema_version = "1.0"``）::

    {
      "schema_version": "1.0",
      "n_total_judged": 25,
      "n_l1": 3,
      "n_l2": 12,
      "n_l3": 10,
      "covered_why_classes": ["budget_awareness", ...],
      "covered_why_count": 7,
      "coverage_satisfied": true,
      "min_why_classes_target": 7,
      "l1_alerts": [
        {
          "trajectory_id": "...",
          "signal_description": "...",
          "reasoning": "...",
          "confidence": 0.9
        }
      ],
      "cells": {
        "hook_new::budget_awareness": {
          "n_candidates": 3,
          "trajectory_ids": ["...", ...],
          "candidates": [
            {
              "trajectory_id": "...",
              "issue_type": "L3",
              "confidence": 0.85,
              "components": [
                {"component_id": "comp_1", "target_file": "...",
                 "summary": "...", "depends_on": []}
              ],
              "reasoning": "..."
            }
          ]
        },
        ...
      },
      "where_distribution": {"hook_new": 5, "skill": 3, ...},
      "why_distribution": {"budget_awareness": 4, ...}
    }

Cell key 使用 ``"<WHERE>::<WHY>"``，用 double-colon 而非 nesting，使文件保持 valid JSON
且可从 shell grep。coverage gate 只证明 judge taxonomy 覆盖范围，不证明 proposal 正确、patch
已应用或 benchmark 效果为正。
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from codeflow.evolver.judge.schema import (
    IssueType,
    JudgeResult,
    PatchWhere,
    PatchWhy,
)

SCHEMA_VERSION = "1.0"
DEFAULT_MIN_WHY_CLASSES = 7  # 规范第 14 节步骤 ③ 的验收门


def build_failure_map(
    judge_results: Iterable[JudgeResult],
    *,
    min_why_classes: int = DEFAULT_MIN_WHY_CLASSES,
) -> dict[str, Any]:
    """把 ``JudgeResult`` iterable 聚合为 failure_map dict。

    ``judge_results`` 通常来自 cold-start bandit sampled trial 的 claude judge call；
    ``min_why_classes`` 是 WHY coverage target，默认 7（spec §14 ③）。L1 只进入 alert；L2/L3
    必须按 schema 携带 proposal，并按 WHERE/WHY 计数与分 cell。``PatchWhy.other`` 使用完整
    ``patch_why_extra`` 子名称，缺失时为 ``other:unknown``。

    返回可直接 ``json.dump`` 的 structured map。``coverage_satisfied`` 仅比较 distinct WHY
    count 与 target，不衡量 confidence、候选质量或真实效果。
    """
    results = list(judge_results)

    cells: dict[str, dict[str, Any]] = defaultdict(lambda: {"n_candidates": 0, "trajectory_ids": [], "candidates": []})
    l1_alerts: list[dict[str, Any]] = []
    where_distribution: dict[str, int] = defaultdict(int)
    why_distribution: dict[str, int] = defaultdict(int)
    covered_why: set[str] = set()

    n_l1 = n_l2 = n_l3 = 0
    for r in results:
        if r.issue_type == IssueType.L1:
            n_l1 += 1
            l1_alerts.append(
                {
                    "trajectory_id": r.trajectory_id,
                    "signal_description": r.signal_description,
                    "reasoning": r.proposed_action.reasoning,
                    "confidence": r.confidence,
                }
            )
            continue

        # L2/L3 必须有 patch_proposal，模式不变量会强制执行。
        if r.issue_type == IssueType.L2:
            n_l2 += 1
        else:
            n_l3 += 1

        action = r.proposed_action
        where = action.patch_where
        why = action.patch_why
        if where is None or why is None:
            # 防御性处理：模式的 __post_init__ 本应捕获此情况。
            continue

        where_key = where.value
        # 按模式约定，patch_why_extra 携带包含 "other:" 前缀的完整子名称；
        # 参见 PatchWhy.other 文档字符串。
        why_key = why.value if why != PatchWhy.other else (action.patch_why_extra or "other:unknown")
        where_distribution[where_key] += 1
        why_distribution[why_key] += 1
        covered_why.add(why_key)

        cell_key = f"{where_key}::{why_key}"
        cells[cell_key]["n_candidates"] += 1
        cells[cell_key]["trajectory_ids"].append(r.trajectory_id)
        cells[cell_key]["candidates"].append(
            {
                "trajectory_id": r.trajectory_id,
                "issue_type": r.issue_type.value,
                "confidence": r.confidence,
                "reasoning": action.reasoning,
                "components": [c.to_dict() for c in action.components],
            }
        )

    coverage_count = len(covered_why)
    return {
        "schema_version": SCHEMA_VERSION,
        "n_total_judged": len(results),
        "n_l1": n_l1,
        "n_l2": n_l2,
        "n_l3": n_l3,
        "covered_why_classes": sorted(covered_why),
        "covered_why_count": coverage_count,
        "min_why_classes_target": min_why_classes,
        "coverage_satisfied": coverage_count >= min_why_classes,
        "l1_alerts": l1_alerts,
        "cells": dict(cells),
        "where_distribution": dict(where_distribution),
        "why_distribution": dict(why_distribution),
    }


def write_failure_map(
    failure_map: dict[str, Any],
    out_path: str | Path,
    *,
    indent: int = 2,
) -> None:
    """把 failure_map atomic write 到 ``out_path``。

    使用 temp file + rename 提供 crash safety，与 ``evolver/tree/store.py`` 模式相同。
    ``indent`` 默认 2，并按 key 排序。函数不验证 map schema；返回只表示 rename 完成。
    """
    out_path = Path(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(failure_map, indent=indent, sort_keys=True))
    tmp.replace(out_path)


def coverage_gap(failure_map: dict[str, Any]) -> list[str]:
    """返回 judge output 尚未覆盖的 canonical WHY Enum value。

    非空表示 cold-start bandit 尚未满足 spec §14 step ③，可用于决定增加 budget 重跑，或明确
    接受 partial coverage 后继续。比较对象只含 first-class ``PatchWhy``，排除 ``other``；
    ``other:*`` 会增加 coverage_count，但不会填补 canonical WHY axis。返回按字典序排序。
    """
    covered = set(failure_map.get("covered_why_classes", []))
    canonical = {w.value for w in PatchWhy if w != PatchWhy.other}
    return sorted(canonical - covered)


def candidates_for_cell(
    failure_map: dict[str, Any],
    where: PatchWhere | str,
    why: PatchWhy | str,
) -> list[dict[str, Any]]:
    """返回指定 ``(WHERE, WHY)`` cell 的 candidate list。

    ``where``/``why`` 可传 Enum 或 underlying string。cell absent/empty 时返回 ``[]``；返回
    新 list，但其中 dict 仍引用原 map 内容。函数不按 confidence 排序或筛选。
    """
    where_key = where.value if isinstance(where, PatchWhere) else where
    why_key = why.value if isinstance(why, PatchWhy) else why
    cell_key = f"{where_key}::{why_key}"
    cell = failure_map.get("cells", {}).get(cell_key)
    if cell is None:
        return []
    return list(cell.get("candidates", []))


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_MIN_WHY_CLASSES",
    "build_failure_map",
    "write_failure_map",
    "coverage_gap",
    "candidates_for_cell",
]
