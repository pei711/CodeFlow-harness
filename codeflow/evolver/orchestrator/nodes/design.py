"""执行 Step ②：选择 WHY，并在预算内语义化设计 candidates。

Two halves, matching the SOP: a *deterministic* WHY selection off the failure
map (pick 1-2 pathologies worth attacking this round), then a *semantic* design
call per candidate that writes an env-gated ``AppliedPatch`` (2-3 per WHY,
across levers). The budget (``max_why_per_round`` x ``candidates_per_why``) is
enforced in code so a chatty driver can't blow up the round's candidate count.

The design call's output is parsed straight through ``AppliedPatch.from_dict``,
so the dataclass's own invariants (non-empty components, unique component ids,
``patch_why=other`` needs ``patch_why_extra``, resolvable ``depends_on``) do the
schema validation; any violation raises and :class:`SemanticNode` feeds it back
for a bounded repair-retry.

Writing a *useful* diff needs the target file's current contents; the caller
supplies them via ``file_context`` (empty here keeps the node structural — it
still produces a schema-valid patch, but a production run wires repo context in).
patch 必须 env-gated 且 default-off，使 activation flag 未设置时 vanilla build byte-identical
（SOP §2 ②）。设计输出只是 proposal，仍需 apply、manifest 与 eval gate。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from codeflow.evolver.orchestrator.config import Budget
from codeflow.evolver.orchestrator.nodes.semantic import CallFn, SemanticNode
from codeflow.evolver.tree.node import AppliedPatch


@dataclass(frozen=True)
class WhyTarget:
    """本 round 选中的 pathology，以及支持它的 WHERE、candidate count 与 trajectory evidence。"""

    why: str
    where_options: list[str]
    n_candidates: int
    trajectory_ids: list[str]


def select_target_whys(failure_map: dict[str, Any], budget: Budget) -> list[WhyTarget]:
    """从 failure map 选择 top ``budget.max_why_per_round`` WHY。

    Ranks by how many L2/L3 candidates the diagnosis attributed to each WHY
    (``why_distribution``), then gathers each WHY's WHERE levers and evidence
    trajectory ids from ``cells`` (keyed ``"<WHERE>::<WHY>"``). Ties break by
    WHY name，使 selection reproducible。返回 target 不证明 pathology 判断正确。
    """
    why_dist: dict[str, int] = failure_map.get("why_distribution", {})
    cells: dict[str, Any] = failure_map.get("cells", {})
    if not why_dist:
        return []

    ranked = sorted(why_dist.items(), key=lambda kv: (-kv[1], kv[0]))
    targets: list[WhyTarget] = []
    for why, _count in ranked[: budget.max_why_per_round]:
        where_options: list[str] = []
        trajectory_ids: list[str] = []
        n_candidates = 0
        for key, cell in cells.items():
            where, _, cell_why = key.partition("::")
            if cell_why != why:
                continue
            where_options.append(where)
            n_candidates += int(cell.get("n_candidates", 0))
            trajectory_ids.extend(cell.get("trajectory_ids", []))
        targets.append(
            WhyTarget(
                why=why,
                where_options=sorted(set(where_options)),
                n_candidates=n_candidates,
                trajectory_ids=trajectory_ids,
            )
        )
    return targets


def build_design_messages(
    target: WhyTarget,
    *,
    attempt_index: int,
    parent_summary: str,
    file_context: str = "",
    archive_summary: str = "",
) -> list[dict[str, str]]:
    """为一个 WHY 的单个 candidate 组装 design prompt。

    ``archive_summary`` (the GSME elite bank, one line per cell) tells the
    driver which mechanisms are already verified, so it neither re-invents a
    banked win，也不会把 pruned approach 当 novel。函数只构造 messages，不调用 model。
    """
    system = (
        "You design a single harness patch that fixes one pathology in an agent "
        "harness. Output ONLY a JSON object, no prose, no code fences, with keys: "
        "patch_where (one of the given WHERE options), patch_why, patch_why_extra "
        "(null unless patch_why is 'other'), overall_reasoning, and components: a "
        "list of {component_id, target_file, diff, rationale, depends_on}. The diff "
        "must be a valid unified diff. The patch MUST be env-gated and default OFF "
        "so the harness is byte-identical to vanilla when the flag is unset. Prefer "
        "one component for a simple fix."
    )
    user = (
        f"Pathology (WHY): {target.why}\n"
        f"Allowed WHERE levers: {', '.join(target.where_options) or 'config'}\n"
        f"Parent harness: {parent_summary}\n"
        f"This is design attempt #{attempt_index}; make it a distinct approach "
        f"from other attempts on this WHY (different lever or mechanism).\n"
        f"Evidence trajectories: {', '.join(target.trajectory_ids[:8])}\n"
        f"{(archive_summary + chr(10)) if archive_summary else ''}"
        f"{('Target file context:' + chr(10) + file_context) if file_context else ''}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_applied_patch(raw: str) -> AppliedPatch:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s[:4].lower() == "json":
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object found in design output")
    return AppliedPatch.from_dict(json.loads(s[start : end + 1]))


def design_candidate(
    call_fn: CallFn,
    target: WhyTarget,
    *,
    attempt_index: int,
    parent_summary: str,
    file_context: str = "",
    archive_summary: str = "",
    max_retries: int = 3,
) -> AppliedPatch:
    """为 ``target`` 设计一个 env-gated、schema-validated ``AppliedPatch``。

    bounded repair 耗尽会抛错；成功只表示 patch object 合规，不表示 diff 可应用。
    """
    messages = build_design_messages(
        target,
        attempt_index=attempt_index,
        parent_summary=parent_summary,
        file_context=file_context,
        archive_summary=archive_summary,
    )
    node: SemanticNode[AppliedPatch] = SemanticNode(
        name=f"design:{target.why}#{attempt_index}",
        call_fn=call_fn,
        parse_fn=_parse_applied_patch,
        parse_error_types=(ValueError, json.JSONDecodeError),
        max_retries=max_retries,
    )
    return node.run(messages)


def design_round(
    call_fn: CallFn,
    failure_map: dict[str, Any],
    budget: Budget,
    *,
    parent_summary: str = "vanilla",
    file_context_for: Any = None,
    archive_summary: str = "",
    max_retries: int = 3,
) -> list[AppliedPatch]:
    """选择 WHY，并为每项最多设计 ``candidates_per_why`` 个 candidate。

    ``file_context_for`` is an optional ``callable(WhyTarget) -> str`` returning
    the target file contents to ground the diff. A candidate whose design never
    parse 的 candidate 被跳过而不 abort round，因此返回数量可低于 budget。
    """
    patches: list[AppliedPatch] = []
    for target in select_target_whys(failure_map, budget):
        file_context = file_context_for(target) if file_context_for else ""
        for attempt in range(1, budget.candidates_per_why + 1):
            try:
                patches.append(
                    design_candidate(
                        call_fn,
                        target,
                        attempt_index=attempt,
                        parent_summary=parent_summary,
                        file_context=file_context,
                        archive_summary=archive_summary,
                        max_retries=max_retries,
                    )
                )
            except Exception:  # noqa: BLE001 — 跳过无法解析的候选项
                continue
    return patches


__all__ = [
    "WhyTarget",
    "select_target_whys",
    "build_design_messages",
    "design_candidate",
    "design_round",
]
