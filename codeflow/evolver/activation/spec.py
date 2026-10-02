"""定义 node 的 machine-checkable activation_spec：“我何时生效”。

design section 3 为每类 mechanism 定义一种 kind：

* ``trajectory_regex`` 属于 code class，统计 ``scope`` record 中至少一行匹配 ``pattern`` 的
  trajectory；Delta-3 的 ``pure cd`` predicate 属于此类。
* ``consecutive_repeat`` 属于 hook repetition-trigger family，统计连续至少 ``threshold`` 个
  相同 ``scope`` content。默认跳过 empty/whitespace；设置 ``ignore_empty: false`` 才计入。
  其他 hook trigger family 应用 ``trajectory_regex`` 表达，或在此新增 kind + evaluator。
* ``short_content_run`` 属于 reasoning-visibility hook，要求连续至少 ``threshold`` 个
  Tool-call iteration 的 visible assistant content 短于 ``max_chars``，默认 80。它复用
  ``codeflow.evolver.activation.predicates.is_short_toolcall_iteration``，与 Runtime
  ``ReasoningVisibilityHook`` 完全相同：只有带 ``tool_calls`` 的 record 才计数；WITHOUT
  tool_calls 或 long-content record 会 reset。content 在去除 ``<think>...</think>`` 并折叠
  whitespace 后测量。近似边界是 hook 读取 live ``response.content``，recorded Session 读取
  assistant record 的 ``content``；Qwen chain-of-thought 位于独立 ``reasoning_content``，
  hook 本来也不读取。
* ``empty_run`` 属于 response-quality hook，要求连续至少 ``threshold`` 个 empty iteration，
  复用 Runtime ``EmptyRunBreakerHook`` 使用的 ``is_empty_response``。empty 当且仅当 content
  blank AND 无 ``tool_calls``；blank 但发出 Tool call 的 record NOT empty。round-1 incident
  C1 中旧 content-only predicate 预测 64.8% reachable，而尊重 tool_calls 的 hook fired 0。
  任一 non-empty iteration reset。
* ``repeated_failure_run`` 属于 robustness hook，统计 same command head 连续失败至少
  ``threshold`` 次。assistant content 设置 pending head token，Tool record 用 exit code 判定
  failure；same head + fail 增长 run，success 或 head change reset。无 ``Exit code:`` 的 Tool
  record 按 success，属于 conservative under-count，绝不 over-count。该 kind 必须遍历
  assistant + Tool raw record，所以忽略 ``scope``。
* ``min_iterations`` 属于 budget hook，统计 assistant iteration 至少 ``threshold`` 的
  trajectory；wrap-up nudge 是 iteration-count crossing，不用 semantic predicate，因此结构上
  drift-free。
* ``skill_routing`` 属于 Skill class，可达性由 routing dry-query 判断，不做 corpus replay；
  chamber 委托其他路径，``evaluate_spec`` 拒绝它。
* ``presence`` 属于 always-on class，由 preflight CLI 的 offline render/config assert 判断，
  ``evaluate_spec`` 同样拒绝。

``evaluate_spec(spec, corpus) -> int`` 返回 spec 在 HOW MANY trajectories 中可达；0 表示该
corpus 上机制不会运行，gate 应 block。正数只证明可达性，不证明 candidate 在真实 trial 已
触发、更不证明效果或任务完成。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from codeflow.evolver.activation.predicates import (
    command_head,
    is_empty_response,
    is_short_toolcall_iteration,
)

_CORPUS_KINDS = {
    "trajectory_regex",
    "consecutive_repeat",
    "short_content_run",
    "empty_run",
    "repeated_failure_run",
    "min_iterations",
}
_KNOWN_KINDS = _CORPUS_KINDS | {"skill_routing", "presence"}

_DEFAULT_MAX_CHARS = 80


def _normalize_record(r: dict) -> dict:
    """把 logged Session record 转为 shared predicate 所需 shape。

    ``content`` 强制为 string，``tool_calls`` 缺失时为空 list，使离线 predicate 看到与 hook
    通过 ``normalize_response()`` 构建的相同 view。函数不验证 Tool schema。
    """
    return {"content": str(r.get("content") or ""), "tool_calls": r.get("tool_calls") or []}


@dataclass
class ActivationSpec:
    """一个 mechanism 的 activation kind 与 kind-specific raw 参数。

    对象由 ``from_dict`` 验证构造；``kind`` 决定 evaluator，``raw`` 保存除 kind 外的原始字段。
    property 只在对应 kind 含字段时使用，否则可能抛出 ``KeyError``。
    """

    kind: str
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ActivationSpec":
        """验证 dict 并构造 ``ActivationSpec``。

        缺少/unknown kind、缺少各 kind 必填字段、invalid regex，或 threshold/max_chars 无法
        转为 int 时抛出 ``ValueError``。``skill_routing`` 要求 ``skill_name``，``presence``
        要求 ``needle``。验证只保证配置 shape，不判断 corpus 可达性。
        """
        kind = d.get("kind")
        if not kind:
            raise ValueError("activation_spec requires a 'kind' field")
        if kind not in _KNOWN_KINDS:
            raise ValueError(f"unknown activation_spec kind: {kind!r}")
        if kind == "trajectory_regex" and "pattern" not in d:
            raise ValueError("trajectory_regex requires 'pattern'")
        if kind == "trajectory_regex":
            try:
                re.compile(d["pattern"])
            except re.error as exc:
                raise ValueError(f"trajectory_regex pattern is not valid regex: {exc}") from exc
        if kind == "consecutive_repeat" and "threshold" not in d:
            raise ValueError("consecutive_repeat requires 'threshold'")
        if kind == "consecutive_repeat":
            try:
                int(d["threshold"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"consecutive_repeat threshold must be an int: {d['threshold']!r}") from exc
        if kind == "short_content_run" and "threshold" not in d:
            raise ValueError("short_content_run requires 'threshold'")
        if kind == "short_content_run":
            try:
                int(d["threshold"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"short_content_run threshold must be an int: {d['threshold']!r}") from exc
            if "max_chars" in d:
                try:
                    int(d["max_chars"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"short_content_run max_chars must be an int: {d['max_chars']!r}") from exc
        if kind == "empty_run" and "threshold" not in d:
            raise ValueError("empty_run requires 'threshold'")
        if kind == "empty_run":
            try:
                int(d["threshold"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"empty_run threshold must be an int: {d['threshold']!r}") from exc
        if kind == "repeated_failure_run" and "threshold" not in d:
            raise ValueError("repeated_failure_run requires 'threshold'")
        if kind == "repeated_failure_run":
            try:
                int(d["threshold"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"repeated_failure_run threshold must be an int: {d['threshold']!r}") from exc
        if kind == "min_iterations" and "threshold" not in d:
            raise ValueError("min_iterations requires 'threshold'")
        if kind == "min_iterations":
            try:
                int(d["threshold"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"min_iterations threshold must be an int: {d['threshold']!r}") from exc
        if kind == "skill_routing" and "skill_name" not in d:
            raise ValueError("skill_routing requires 'skill_name'")
        if kind == "presence" and "needle" not in d:
            raise ValueError("presence requires 'needle'")
        raw = {k: v for k, v in d.items() if k != "kind"}
        return cls(kind=kind, raw=raw)

    @property
    def pattern(self) -> str:
        return self.raw["pattern"]

    @property
    def threshold(self) -> int:
        return int(self.raw["threshold"])

    @property
    def max_chars(self) -> int:
        return int(self.raw.get("max_chars", _DEFAULT_MAX_CHARS))


def _scope_contents(traj: list[dict], scope: str) -> list[str]:
    return [str(r.get("content") or "") for r in traj if scope in ("any", r.get("role"))]


def evaluate_spec(spec: ActivationSpec, corpus: list[list[dict]]) -> int:
    """统计 ``spec`` 在 ``corpus`` 中可达的 trajectory 数量。

    仅接受 ``_CORPUS_KINDS``；``skill_routing``/``presence`` 抛出 ``ValueError``，要求走
    preflight CLI。各 trajectory 最多计一次命中。regex、repeat、short/empty run、same-command
    failure 与 minimum iteration 分别按模块 docstring 的规则评估。

    返回 0 表示当前 corpus 无可达证据；返回正数不是 activation ledger、task success 或
    candidate improvement 的替代证据。
    """
    if spec.kind not in _CORPUS_KINDS:
        raise ValueError(f"{spec.kind} is not corpus-evaluable; use the preflight CLI path")
    scope = spec.raw.get("scope", "assistant")
    hits = 0
    if spec.kind == "trajectory_regex":
        pat = re.compile(spec.pattern, re.M)
        for traj in corpus:
            if any(pat.search(c) for c in _scope_contents(traj, scope)):
                hits += 1
    elif spec.kind == "consecutive_repeat":
        threshold = spec.threshold
        ignore_empty = spec.raw.get("ignore_empty", True)
        for traj in corpus:
            run, prev = 0, object()
            for c in _scope_contents(traj, scope):
                if ignore_empty and not c.strip():
                    # 空响应不携带命令；比较命令的重复触发器永远看不到它们。
                    continue
                run = run + 1 if c == prev else 1
                prev = c
                if run >= threshold:
                    hits += 1
                    break
    elif spec.kind == "short_content_run":
        threshold = spec.threshold
        max_chars = spec.max_chars
        for traj in corpus:
            run = 0
            fired = False
            for r in traj:
                if scope not in ("any", r.get("role")):
                    continue
                rec = _normalize_record(r)
                run = run + 1 if is_short_toolcall_iteration(rec, max_chars) else 0
                if run >= threshold:
                    fired = True
                    break
            if fired:
                hits += 1
    elif spec.kind == "empty_run":
        threshold = spec.threshold
        for traj in corpus:
            run = 0
            for r in traj:
                if scope not in ("any", r.get("role")):
                    continue
                rec = _normalize_record(r)
                run = run + 1 if is_empty_response(rec) else 0
                if run >= threshold:
                    hits += 1
                    break
    elif spec.kind == "repeated_failure_run":
        threshold = spec.threshold
        exit_re = re.compile(r"Exit code: (\d+)")
        for traj in corpus:
            run, prev_head = 0, None
            pending_head = None
            for r in traj:
                role, c = r.get("role"), str(r.get("content") or "")
                if role == "assistant" and c.strip():
                    pending_head = command_head(_normalize_record(r))
                elif role == "tool" and pending_head is not None:
                    m = exit_re.search(c)
                    failed = bool(m and m.group(1) != "0")
                    if failed and pending_head == (prev_head or pending_head):
                        run += 1
                        prev_head = pending_head
                    else:
                        run = 1 if failed else 0
                        prev_head = pending_head if failed else None
                    pending_head = None
                    if run >= threshold:
                        hits += 1
                        break
    elif spec.kind == "min_iterations":
        threshold = spec.threshold
        for traj in corpus:
            count = sum(1 for r in traj if r.get("role") == "assistant")
            if count >= threshold:
                hits += 1
    return hits
