"""定义 bench-neutral WHY/WHERE taxonomy 与 open-ended map-reduce induction。

A benchmark's diagnosis step classifies failing trajectories into a
:class:`TaxonomySpec` — WHY (failure-mode) x WHERE (patch-lever) classes. A
bench that has a hand-derived table (AppWorld's W1-W7) passes it as a frozen
constant; a brand-new bench can *discover* one from its vanilla failures via
:func:`induce_taxonomy`: stage-1 writes one open-ended failure report per
trajectory (parallel, no preset table), stage-2 clusters all reports into
classes and assigns each report to its WHY(s).

Induction failure raises :class:`TaxonomyInductionError` — there is no silent
silent fallback，因为 brand-new benchmark 唯一普遍错误答案就是复用 other benchmark table。
bench wiring 决定 safe default，或 loud stop。taxonomy 只组织 diagnosis，不证明因果分类正确。
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_BENCH_DESC = "an agent harness benchmark"


class TaxonomyInductionError(RuntimeError):
    """open-ended induction 无法产生 usable taxonomy 时抛出的异常。"""


@dataclass(frozen=True)
class TaxonomySpec:
    """一个 benchmark 的 WHY failure-mode x WHERE patch-lever classification。

    ``why_classes`` / ``where_classes`` map a stable key to a one-line
    description. ``other`` (why) and ``none`` (where) are the escape hatches and
    总会存在，source 缺失时 post-init 补入。
    """

    why_classes: dict[str, str]
    where_classes: dict[str, str]

    def __post_init__(self) -> None:
        if "other" not in self.why_classes:
            self.why_classes["other"] = "None of the above — provide a short sub-name."
        if "none" not in self.where_classes:
            self.where_classes["none"] = "No harness lever applies."

    def to_dict(self) -> dict[str, Any]:
        return {"why_classes": dict(self.why_classes), "where_classes": dict(self.where_classes)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaxonomySpec":
        return cls(dict(d["why_classes"]), dict(d["where_classes"]))


def strip_code_fence(raw: str) -> str:
    s = raw.strip()
    if "```" in s:
        s = s.split("```")[1] if s.count("```") >= 2 else s
        s = s.split("\n", 1)[-1] if s.lstrip().startswith(("json", "JSON")) else s
    return s


def _why_prefix_match(why: str, key: str) -> bool:
    """仅当 WHY 带 key leading code 且 boundary 清晰时返回 True。

    ``W1_x`` 匹配 ``W1_...``，但 ``W10_x`` 不得误匹配 ``W1_...``。
    """
    prefix = key.split("_")[0].upper()
    w = str(why).upper()
    if not w.startswith(prefix):
        return False
    rest = w[len(prefix) : len(prefix) + 1]
    return not rest.isalnum()


def coerce_mode(obj: dict, taxonomy: TaxonomySpec) -> dict:
    """把 diagnosed failure mode 归一化到 taxonomy key。

    unknown WHY 尝试 prefix match 后回退 other，unknown WHERE 回退 none；reason/fix_hint 截断。
    """
    why = obj.get("why")
    if why not in taxonomy.why_classes:
        cand = next(
            (k for k in taxonomy.why_classes if why and _why_prefix_match(why, k)),
            None,
        )
        why = cand or "other"
    where = obj.get("where") if obj.get("where") in taxonomy.where_classes else "none"
    return {
        "why": why,
        "where": where,
        "dominant": bool(obj.get("dominant", False)),
        "reasoning": str(obj.get("reasoning", ""))[:400],
        "fix_hint": str(obj.get("fix_hint", ""))[:300],
    }


def empty_failure_map() -> dict:
    return {"why_distribution": {}, "cells": {}, "_n_judged": 0}


def add_failure_mode(fm: dict, trajectory_id: str, mode: dict) -> None:
    why, where = mode["why"], mode["where"]
    # 主导模式加权：失败运行中几乎普遍共现的次要症状若按全权重计入，会在 WHY 选择排序的
    # 分布中淹没因果模式。没有该标志的模式（旧调用方）保留原全权重。
    weight = 1.0 if mode.get("dominant", True) else 0.5
    fm["why_distribution"][why] = fm["why_distribution"].get(why, 0) + weight
    cell = fm["cells"].setdefault(f"{where}::{why}", {"candidates": []})
    cell["candidates"].append(
        {
            "trajectory_id": trajectory_id,
            "reasoning": mode["reasoning"],
            "components": [{"summary": mode["fix_hint"]}] if mode["fix_hint"] else [],
        }
    )


def _parse_modes(raw: str, taxonomy: TaxonomySpec) -> list[dict] | None:
    """把 multi-label diagnosis response 解析为 mode list。

    Accepts a JSON array (preferred) or a single JSON object (back-compat with
    single-label caller）并归一化为 one-element list。确保恰一个 dominant mode。
    """
    s = strip_code_fence(raw)
    obj = None
    i, j = s.find("["), s.rfind("]")
    if i >= 0 and j > i:
        try:
            obj = json.loads(s[i : j + 1])
        except json.JSONDecodeError:
            obj = None
    if obj is None:
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            try:
                obj = json.loads(s[i : j + 1])
            except json.JSONDecodeError:
                obj = None
    if obj is None:
        return None
    items = obj if isinstance(obj, list) else [obj]
    modes = [coerce_mode(x, taxonomy) for x in items if isinstance(x, dict)]
    if not modes:
        return None
    # 必须恰有一个主导模式：保留第一个带标志者；模型未标记时，按“主导优先”的排序规则取索引 0。
    first = next((i for i, m in enumerate(modes) if m["dominant"]), 0)
    for i, m in enumerate(modes):
        m["dominant"] = i == first
    return modes


def classify_failures(
    call_fn: Callable[[list], str],
    trajectories,
    taxonomy: TaxonomySpec,
    *,
    bench_intro: str,
    extra_rules: str = "",
    max_workers: int = 8,
    retries: int = 2,
) -> dict:
    """按 ``taxonomy`` 把 failing trajectories Judge 为 multi-label failure_map。

    The bench-neutral diagnosis core: ``bench_intro`` describes the harness and
    task shape (one or two sentences); ``extra_rules`` optionally appends
    taxonomy-specific guidance (e.g. which WHYs are capability ceilings).
    ``trajectories`` = ``(trajectory_id, task_description, transcript)`` tuples;
    each trajectory can contribute several modes and every hit increments its
    WHY. Output shape matches ``failure_map_builder.build_failure_map`` so
    ``select_target_whys``/design 原样消费。每条 trajectory parse failure 在 bounded retry 后跳过。
    """
    sys = (
        f"{bench_intro} You are given ONE failing trajectory. Classify "
        "ALL failure modes it exhibits (usually 1-3, occasionally more) — a trajectory can fail in "
        "several ways at once. For EACH mode name its WHY class, the best patch location (WHERE), a "
        "one-line reasoning and a concrete fix hint.\n\n"
        "WHY classes:\n" + "\n".join(f"  - {k}: {v}" for k, v in taxonomy.why_classes.items()) + "\n\n"
        "WHERE classes:\n" + "\n".join(f"  - {k}: {v}" for k, v in taxonomy.where_classes.items()) + "\n\n"
        'Rules: mark EXACTLY ONE mode "dominant": true — the failure that directly explains the '
        "benchmark's verdict — and list it first; other modes are secondary symptoms "
        '("dominant": false). '
        + (extra_rules + "\n" if extra_rules else "")
        + "Respond with ONLY a JSON ARRAY, no prose, no code fences; each element is one mode:\n"
        '[{"why":"<one WHY key>","where":"<one WHERE key>","dominant":true|false,'
        '"reasoning":"<=1 line","fix_hint":"<=1 line concrete lever>"}]'
    )

    def _one(t):
        tid, desc, transcript = t
        user = (
            f"TASK: {desc}\n\nFAILING TRAJECTORY (trajectory_id={tid}):\n{transcript}\n\n"
            "Classify ALL failure modes. JSON array only."
        )
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        for _ in range(retries + 1):
            try:
                modes = _parse_modes(call_fn(msgs), taxonomy)
            except Exception:  # noqa: BLE001
                modes = None
            if modes:
                return tid, modes
            msgs = msgs + [{"role": "user", "content": "Return ONLY a JSON array of {why,where,...} with valid keys."}]
        return None

    fm = empty_failure_map()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r in ex.map(_one, list(trajectories)):
            if not r:
                continue
            tid, modes = r
            fm["_n_judged"] += 1
            for mode in modes:
                add_failure_mode(fm, tid, mode)
    return fm


# ---- 开放式分类归纳（map-reduce，仅用于新基准）-----------------------------


def _parse_reports(raw: str) -> list[dict] | None:
    """解析 Stage-1 ``{failure_point, evidence, fixes:[...]}`` report list。"""
    s = strip_code_fence(raw)
    i, j = s.find("["), s.rfind("]")
    if i < 0 or j <= i:
        i, j = s.find("{"), s.rfind("}")
        if i < 0 or j <= i:
            return None
        s = "[" + s[i : j + 1] + "]"
    else:
        s = s[i : j + 1]
    try:
        arr = json.loads(s)
    except json.JSONDecodeError:
        return None
    return [x for x in arr if isinstance(x, dict)] or None


def _induce_reports(call_fn, trajectories, *, bench_desc, max_workers, retries) -> list[dict]:
    """Stage 1 map：并行生成每 trajectory 一个 open-ended failure report。"""
    sys = (
        f"You analyse ONE failing agent trajectory from {bench_desc} with NO preset failure "
        "taxonomy. Describe every distinct way it failed (usually 1-3). For each: the concrete "
        "failure_point (what went wrong, at which step), evidence (quote one line from the "
        "trajectory), and harness_fixes (1-2 concrete ways an agent-harness change — prompt / a "
        "runtime hook / the exec tool / the loop — could prevent it; NOT model retraining). Keep "
        "each field short.\n"
        'Respond ONLY a JSON array: [{"failure_point":"...","evidence":"...","harness_fixes":["...","..."]}]'
    )

    def _one(t):
        tid, desc, transcript = t
        user = (
            f"TASK: {desc}\n\nFAILING TRAJECTORY ({tid}):\n{transcript}\n\nReport its failure modes. JSON array only."
        )
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        for _ in range(retries + 1):
            try:
                reps = _parse_reports(call_fn(msgs))
            except Exception:  # noqa: BLE001
                reps = None
            if reps:
                return {"trajectory_id": tid, "modes": reps}
            msgs = msgs + [{"role": "user", "content": "Return ONLY the JSON array."}]
        return None

    out = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for r in ex.map(_one, list(trajectories)):
            if r:
                out.append(r)
    return out


def _parse_taxonomy(raw: str) -> tuple[TaxonomySpec, list[dict]]:
    """解析 Stage-2 reduce 的 TaxonomySpec 与 multi-label per-report assignments。"""
    s = strip_code_fence(raw)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object in taxonomy reduce output")
    obj = json.loads(s[i : j + 1])
    why = {c["key"]: str(c.get("desc", "")) for c in obj.get("why_classes", []) if c.get("key")}
    where = {c["key"]: str(c.get("desc", "")) for c in obj.get("where_classes", []) if c.get("key")}
    if not why:
        raise ValueError("reduce produced no why_classes")
    assignments = [a for a in obj.get("assignments", []) if isinstance(a, dict)]
    return TaxonomySpec(why, where), assignments


def _pack_reports(reports: list[dict], *, budget: int) -> str:
    """在 ``budget`` character 内打包 WHOLE report 的 JSON-array payload。

    A raw ``json.dumps(reports)[:budget]`` cut mid-record, leaving invalid
    JSON and silently biasing the taxonomy toward early trajectories; packing
    whole records keeps the payload parseable and makes any drop explicit
    trailing marker 明确 omitted count。
    """
    parts: list[str] = []
    size = 2  # 方括号
    dropped = 0
    for r in reports:
        s = json.dumps(r)
        if parts and size + len(s) + 2 > budget:
            dropped += 1
            continue
        parts.append(s)
        size += len(s) + 2
    payload = "[" + ", ".join(parts) + "]"
    if dropped:
        payload += f"\n({dropped} more reports omitted for length)"
    return payload


def induce_taxonomy(
    call_fn: Callable[[list], str],
    trajectories,
    *,
    bench_desc: str = DEFAULT_BENCH_DESC,
    max_workers: int = 8,
    retries: int = 2,
    target_min: int = 5,
    target_max: int = 9,
) -> tuple[TaxonomySpec, dict]:
    """从 vanilla failures 以 open-ended map-reduce 发现 WHY/WHERE taxonomy。

    Stage 1 (map): one compact failure report per trajectory, no preset table.
    Stage 2 (reduce): cluster all reports into ``target_min..target_max`` WHY
    classes + WHERE lever classes, and assign each report to its WHY(s). Returns
    the :class:`TaxonomySpec` plus a seed multi-label failure_map from the
    assignments. Raises :class:`TaxonomyInductionError` when no report parses or
    the reduce never yields a taxonomy — never silently substitutes another
    bench table。返回 taxonomy 与 seed failure map，不代表分类经过人工确认。
    """
    reports = _induce_reports(call_fn, trajectories, bench_desc=bench_desc, max_workers=max_workers, retries=retries)
    if not reports:
        raise TaxonomyInductionError("taxonomy induction produced no parseable per-trajectory reports")

    sys = (
        "You are consolidating many per-trajectory failure reports into a REUSABLE failure "
        f"taxonomy for {bench_desc}. Cluster the reports into "
        f"{target_min}-{target_max} WHY classes (abstract failure MODES, each a stable key like "
        "'W1_...' + a one-line description) and a small set of WHERE classes (harness patch LEVERS "
        "abstracted from the fixes: prompt / runtime hook / exec tool / loop / config / none). Then "
        "assign each report (by trajectory_id) to its WHY key(s) — a report may map to several.\n"
        'Respond ONLY JSON: {"why_classes":[{"key":"W1_...","desc":"..."}],'
        '"where_classes":[{"key":"...","desc":"..."}],'
        '"assignments":[{"trajectory_id":"...","whys":["W1_..."],"wheres":["..."]}]}'
    )
    payload = _pack_reports(reports, budget=60000)
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Reports:\n{payload}\n\nProduce the taxonomy JSON."},
    ]
    taxonomy: TaxonomySpec | None = None
    assignments: list[dict] = []
    last_exc: Exception | None = None
    for _ in range(retries + 1):
        try:
            taxonomy, assignments = _parse_taxonomy(call_fn(msgs))
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msgs = msgs + [{"role": "user", "content": f"Invalid ({exc}). Return ONLY the JSON object."}]
    if taxonomy is None:
        raise TaxonomyInductionError(f"taxonomy reduce failed after {retries + 1} attempts; last error: {last_exc!r}")

    # 种子把阶段 1 报告内容（failure_point / harness_fixes）带入单元，使第 1 轮设计步骤可直接
    # 使用；这些轨迹刚刚已评判，重复评判只会再次消耗驱动器。
    seed = empty_failure_map()
    modes_by_tid = {r["trajectory_id"]: r["modes"] for r in reports}
    for a in assignments:
        tid = a.get("trajectory_id")
        if tid not in modes_by_tid:
            continue
        seed["_n_judged"] += 1
        whys = a.get("whys") or []
        wheres = a.get("wheres") or ["none"]
        reps = modes_by_tid[tid]
        for i, w in enumerate(whys):
            rep = reps[i] if i < len(reps) else reps[0]
            fixes = rep.get("harness_fixes") or []
            add_failure_mode(
                seed,
                tid,
                coerce_mode(
                    {
                        "why": w,
                        "where": wheres[0],
                        "dominant": i == 0,
                        "reasoning": str(rep.get("failure_point", "")),
                        "fix_hint": str(fixes[0]) if fixes else "",
                    },
                    taxonomy,
                ),
            )
    return taxonomy, seed


def ensure_taxonomy(
    call_fn: Callable[[list], str],
    trajectories,
    path: str | Path,
    *,
    mode: str = "hardcoded",
    default: Optional[TaxonomySpec] = None,
    bench_desc: str = DEFAULT_BENCH_DESC,
    max_workers: int = 8,
    seed_path: Optional[str | Path] = None,
) -> TaxonomySpec:
    """为 bench 解析 hardcoded taxonomy，或 induce-and-cache。

    ``mode="hardcoded"`` returns ``default`` (required — the bench's own table).
    ``mode="induce"`` loads ``path`` if it exists, else runs
    :func:`induce_taxonomy` over ``trajectories`` and persists the result to
    ``path`` — induction runs once and every later call reuses the frozen
    taxonomy. Induction failure raises; it never falls back to ``default``.

    ``seed_path`` (optional) persists the induction's seed failure map next to
    the taxonomy, so the caller can feed it to round 1 instead of re-judging the
    very trajectories induction just judged (round-0 is genuinely free). It is
    written only when induction actually runs; a cached taxonomy leaves any
    previously written seed。induction failure 不回退 other bench default。
    """
    if mode == "hardcoded":
        if default is None:
            raise ValueError('mode="hardcoded" requires a default TaxonomySpec')
        return default
    p = Path(path)
    if p.exists():
        return TaxonomySpec.from_dict(json.loads(p.read_text()))
    taxonomy, seed = induce_taxonomy(call_fn, trajectories, bench_desc=bench_desc, max_workers=max_workers)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(taxonomy.to_dict(), indent=2))
    if seed_path is not None:
        sp = Path(seed_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(seed, indent=2))
    return taxonomy


def resolve_taxonomy(
    call_fn: Callable[[list], str],
    trajectory_source: Callable[[int, Any], list],
    vanilla_node: Any,
    *,
    mode: str,
    work_dir: str | Path,
    hardcoded: Optional[TaxonomySpec] = None,
    taxonomy_path: Optional[str | Path] = None,
) -> tuple[TaxonomySpec, Optional[dict]]:
    """解析 round taxonomy，并在 induce mode 返回 round-1 seed failure map。

    The shared front half of both benches' diagnose wiring. In ``"hardcoded"``
    mode returns the caller-supplied ``hardcoded`` table and no seed. In
    ``"induce"`` mode discovers the table once from the vanilla failures (cached
    to ``taxonomy_path`` / ``work_dir/taxonomy.json``) and, because induction
    judges those failures already, returns its seed failure map marked with the
    root as diagnosed — so round 1 reuses it instead of re-judging the same
    trajectories (round-0 free). Requires the vanilla ledger to already exist
    cold start ledger 已存在。seed 标记 root diagnosed，避免重复 Judge 同一 trajectory。
    """
    if mode == "hardcoded":
        if hardcoded is None:
            raise ValueError('resolve_taxonomy mode="hardcoded" requires a table')
        return hardcoded, None
    tax_path = Path(taxonomy_path) if taxonomy_path else (Path(work_dir) / "taxonomy.json")
    seed_path = tax_path.with_name(tax_path.stem + "_seed.json")
    taxonomy = ensure_taxonomy(
        call_fn,
        trajectory_source(1, vanilla_node),
        tax_path,
        mode="induce",
        seed_path=seed_path,
    )
    seed_failure_map: Optional[dict] = None
    if seed_path.exists():
        seed = json.loads(seed_path.read_text())
        if seed.get("_n_judged"):
            seed["_diagnosed_parents"] = [vanilla_node.node_id]
            seed_failure_map = seed
    return taxonomy, seed_failure_map


__all__ = [
    "TaxonomySpec",
    "TaxonomyInductionError",
    "DEFAULT_BENCH_DESC",
    "strip_code_fence",
    "coerce_mode",
    "empty_failure_map",
    "add_failure_mode",
    "classify_failures",
    "induce_taxonomy",
    "ensure_taxonomy",
    "resolve_taxonomy",
]
