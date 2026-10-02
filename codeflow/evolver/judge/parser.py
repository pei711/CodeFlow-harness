"""把 LLM-judge raw text 解析为 validated ``JudgeResult``。

prompt 要求单个 JSON object（spec §3.2 schema + spec §12.4-§12.5 Enum），但真实 LLM 可能
包 markdown code fence、添加 ``Here is the analysis:`` 前缀，或在 JSON 后追加 summary。parser
容忍这些包装：移除 `````json``/````` fence（任意 language tag），定位 first ``{`` 并找到
string-aware balanced ``}``，校验 ``IssueType``/``PatchWhere``/``PatchWhy``，再由
``JudgeResult.__post_init__`` 强制 L1/L2/L3 与 action kind invariant。

它不容忍 truncated JSON、multiple top-level JSON object、missing required field 或 invalid
Enum string。所有缺陷抛出带短 reason 的 ``JudgeParseError``；caller 通常 log + skip，或把
reason 反馈给 Judge 做 ``retry, fix this``。parse 成功只证明输出 shape 合规，不证明分类正确。
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .schema import (
    ActionKind,
    IssueType,
    JudgeAction,
    JudgeResult,
    PassFailResult,
    PatchWhere,
    PatchWhy,
    ProposedComponent,
)


class JudgeParseError(ValueError):
    """LLM judge output 无法解析为 typed result 时抛出的异常。"""


# ---------------------------------------------------------------------------
# 文本转 JSON 字典
# ---------------------------------------------------------------------------


_CODE_FENCE_RE = re.compile(r"^\s*```(?:json|JSON|jsonc)?\s*\n?|\n?\s*```\s*$", re.MULTILINE)


def _extract_json_object(raw: str) -> str:
    """从 ``raw`` 提取 first balanced ``{...}`` block。

    先移除 Markdown fence，再找 first ``{``，向后逐字符维护 brace depth；scanner 对
    double-quoted string 与 backslash escape aware，所以忽略 string 内 brace。找到 matching
    close 后返回 inclusive substring。无 open brace 或始终不 balance 时抛出
    ``JudgeParseError``。

    这比 ``raw[raw.find('{'):raw.rfind('}')+1]`` 更健壮，因为 LLM 可能在 main JSON 后追加
    ``Note: see {related_file} for context.`` 之类第二段 brace text。
    """
    stripped = _CODE_FENCE_RE.sub("", raw).strip()
    start = stripped.find("{")
    if start < 0:
        raise JudgeParseError("no '{' found in judge output")

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : i + 1]
    raise JudgeParseError("braces never balance — JSON likely truncated")


def _require(d: dict[str, Any], key: str, what: str) -> Any:
    if key not in d:
        raise JudgeParseError(f"missing required field '{key}' in {what}")
    return d[key]


def _coerce_enum(value: Any, enum_cls: type, field_name: str) -> Any:
    """把 string value 转为 Enum，并在失败时列出合法值。

    只接受 Enum member 的 string ``value``；int、member ``.name`` 或其他 shape 都拒绝，并在
    ``JudgeParseError`` 中包含 ``field_name``。
    """
    if not isinstance(value, str):
        raise JudgeParseError(f"field '{field_name}' must be a string, got {type(value).__name__}")
    try:
        return enum_cls(value)
    except ValueError as exc:
        valid = [m.value for m in enum_cls]
        raise JudgeParseError(f"field '{field_name}'={value!r} not one of {valid}") from exc


def _parse_components(action_obj: dict[str, Any]) -> list[ProposedComponent]:
    """解析 patch_proposal action 的 ``components`` array。

    新 schema 要求 non-empty object list，逐项验证 component_id、target_file、summary 与
    string-list depends_on；缺少 component_id 时生成 ``comp_<n>``。backwards-compat 路径允许
    flat ``target_file`` + ``patch_summary``，合成 single component，使已训练 Judge prompt 在
    multi-component rollout 期间继续工作。
    """
    raw_components = action_obj.get("components")

    # 路径 A：新模式，显式组件列表。
    if raw_components is not None:
        if not isinstance(raw_components, list):
            raise JudgeParseError(f"proposed_action.components must be a list, got {type(raw_components).__name__}")
        if not raw_components:
            raise JudgeParseError("proposed_action.components must be non-empty for patch_proposal")
        parsed: list[ProposedComponent] = []
        for i, item in enumerate(raw_components):
            if not isinstance(item, dict):
                raise JudgeParseError(f"proposed_action.components[{i}] must be an object, got {type(item).__name__}")
            component_id = item.get("component_id") or f"comp_{i + 1}"
            if not isinstance(component_id, str) or not component_id.strip():
                raise JudgeParseError(f"proposed_action.components[{i}].component_id must be a non-empty string")
            target_file = item.get("target_file")
            if not isinstance(target_file, str) or not target_file.strip():
                raise JudgeParseError(f"proposed_action.components[{i}].target_file must be a non-empty string")
            summary = item.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                raise JudgeParseError(f"proposed_action.components[{i}].summary must be a non-empty string")
            depends_on_raw = item.get("depends_on") or []
            if not isinstance(depends_on_raw, list) or not all(isinstance(x, str) for x in depends_on_raw):
                raise JudgeParseError(f"proposed_action.components[{i}].depends_on must be a list of strings")
            parsed.append(
                ProposedComponent(
                    component_id=component_id,
                    target_file=target_file,
                    summary=summary,
                    depends_on=list(depends_on_raw),
                )
            )
        return parsed

    # 路径 B：旧扁平模式，合成单个组件。
    target_file = action_obj.get("target_file")
    patch_summary = action_obj.get("patch_summary")
    if target_file is None or patch_summary is None:
        raise JudgeParseError(
            "proposed_action must contain either 'components' (preferred) or "
            "the legacy 'target_file' + 'patch_summary' pair"
        )
    if not isinstance(target_file, str) or not target_file.strip():
        raise JudgeParseError("proposed_action.target_file must be a non-empty string")
    if not isinstance(patch_summary, str) or not patch_summary.strip():
        raise JudgeParseError("proposed_action.patch_summary must be a non-empty string")
    return [
        ProposedComponent(
            component_id="comp_1",
            target_file=target_file,
            summary=patch_summary,
            depends_on=[],
        ),
    ]


def _parse_evidence_range(raw: Any) -> Optional[tuple[int, int]]:
    """验证 inclusive ``[start, end]`` Turn range。

    ``None`` 原样返回；其他值必须是恰好两个 int 的 list/tuple，且 start <= end，否则抛出
    ``JudgeParseError``。
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise JudgeParseError(f"evidence_turn_range must be a list/tuple, got {type(raw).__name__}")
    if len(raw) != 2:
        raise JudgeParseError(f"evidence_turn_range must have exactly 2 elements, got {len(raw)}")
    a, b = raw
    if not (isinstance(a, int) and isinstance(b, int)):
        raise JudgeParseError(f"evidence_turn_range elements must be ints, got {a!r}, {b!r}")
    if a > b:
        raise JudgeParseError(f"evidence_turn_range start ({a}) > end ({b})")
    return (a, b)


# ---------------------------------------------------------------------------
# 顶层解析器
# ---------------------------------------------------------------------------


def parse_judge_output(
    raw_text: str,
    *,
    expected_trajectory_id: Optional[str] = None,
) -> JudgeResult:
    """把一次 Judge response 解析为 validated ``JudgeResult``。

    ``expected_trajectory_id`` 非空时要求 parsed ID exact match，或只多出从 task WRAPPER_PATH
    复制的 suffix，即 parsed ID 以 expected 开头；完全不同 task 抛错，避免 batch pipeline
    silent mis-binding。随后校验 confidence、signal、evidence range、action kind、WHERE/WHY 与
    component，并构造 schema dataclass。

    原始 ``raw_text`` 保存到 ``JudgeResult.raw_response``，供 post-hoc audit 查看 LLM 原话。
    function 不修改或自动修复 Judge output。
    """
    json_text = _extract_json_object(raw_text)
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"JSON decode failed: {exc}") from exc

    if not isinstance(obj, dict):
        raise JudgeParseError(f"top-level must be a JSON object, got {type(obj).__name__}")

    trajectory_id = _require(obj, "trajectory_id", "judge output")
    if not isinstance(trajectory_id, str):
        raise JudgeParseError("trajectory_id must be a string")
    if expected_trajectory_id is not None:
        # 评判器偶尔会回显 trajectory_id，并附上从任务描述复制的包装路径后缀，例如：
        #   预期：'swe-vanilla-500/django__django-11292'
        #   实际：'swe-vanilla-500/django__django-11292-t1-exec'
        # 其中 '-t1-exec' 来自用户消息中的 WRAPPER_PATH。若解析出的 ID 以预期 ID 开头则接受，
        # 避免批处理因装饰性后缀丢失其他方面有效的记录；完全不同任务的严格不匹配仍会抛错。
        if trajectory_id != expected_trajectory_id and not trajectory_id.startswith(expected_trajectory_id):
            raise JudgeParseError(f"trajectory_id mismatch: expected {expected_trajectory_id!r}, got {trajectory_id!r}")

    issue_type = _coerce_enum(_require(obj, "issue_type", "judge output"), IssueType, "issue_type")
    confidence = _require(obj, "confidence", "judge output")
    if not isinstance(confidence, (int, float)):
        raise JudgeParseError(f"confidence must be a number, got {type(confidence).__name__}")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise JudgeParseError(f"confidence must be in [0.0, 1.0], got {confidence}")

    signal_description = _require(obj, "signal_description", "judge output")
    if not isinstance(signal_description, str):
        raise JudgeParseError("signal_description must be a string")

    evidence_turn_range = _parse_evidence_range(obj.get("evidence_turn_range"))

    action_obj = _require(obj, "proposed_action", "judge output")
    if not isinstance(action_obj, dict):
        raise JudgeParseError("proposed_action must be an object")

    kind = _coerce_enum(_require(action_obj, "kind", "proposed_action"), ActionKind, "proposed_action.kind")
    reasoning = _require(action_obj, "reasoning", "proposed_action")
    if not isinstance(reasoning, str):
        raise JudgeParseError("proposed_action.reasoning must be a string")

    # 补丁字段仅在 kind=patch_proposal 时必填。
    patch_where = None
    patch_why = None
    patch_why_extra = None
    components: list[ProposedComponent] = []

    if kind == ActionKind.patch_proposal:
        patch_where = _coerce_enum(
            _require(action_obj, "patch_where", "proposed_action(patch_proposal)"),
            PatchWhere,
            "proposed_action.patch_where",
        )
        patch_why = _coerce_enum(
            _require(action_obj, "patch_why", "proposed_action(patch_proposal)"),
            PatchWhy,
            "proposed_action.patch_why",
        )
        components = _parse_components(action_obj)
        # patch_why=other 时必须提供子名称。
        patch_why_extra_raw = action_obj.get("patch_why_extra")
        if patch_why == PatchWhy.other:
            if not patch_why_extra_raw or not isinstance(patch_why_extra_raw, str):
                raise JudgeParseError("patch_why='other' requires non-empty patch_why_extra string")
            patch_why_extra = patch_why_extra_raw

    try:
        action = JudgeAction(
            kind=kind,
            reasoning=reasoning,
            patch_where=patch_where,
            patch_why=patch_why,
            patch_why_extra=patch_why_extra,
            components=components,
        )
    except ValueError as exc:
        raise JudgeParseError(str(exc)) from exc

    # JudgeResult.__post_init__ 负责强制跨字段不变量。
    return JudgeResult(
        trajectory_id=trajectory_id,
        issue_type=issue_type,
        confidence=confidence,
        signal_description=signal_description,
        proposed_action=action,
        evidence_turn_range=evidence_turn_range,
        raw_response=raw_text,
    )


def parse_pass_fail(raw: str, *, expected_trajectory_id: str = "") -> PassFailResult:
    """解析无 benchmark verifier 的 pass/fail verdict，缺陷全部抛错以触发 retry。

    复用 tolerant JSON extractor，使同一 ``SemanticNode`` repair loop 可用。必须包含
    boolean-ish ``passed``；string 的 true/pass/passed/yes/1 视为真，其他值按 bool 转换。
    缺失 trajectory_id 时使用 expected ID。返回是 LLM scorer verdict，不是 deterministic
    verifier evidence。
    """
    obj = json.loads(_extract_json_object(raw))
    if "passed" not in obj:
        raise JudgeParseError("pass/fail verdict missing required 'passed' field")
    passed = obj["passed"]
    if isinstance(passed, str):
        passed = passed.strip().lower() in ("true", "pass", "passed", "yes", "1")
    return PassFailResult(
        trajectory_id=str(obj.get("trajectory_id", expected_trajectory_id)),
        passed=bool(passed),
        reasoning=str(obj.get("reasoning", "")),
        raw_response=raw,
    )


__all__ = [
    "JudgeParseError",
    "parse_judge_output",
    "parse_pass_fail",
]
