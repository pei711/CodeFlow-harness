"""维护每个 ``WHERE x WHY`` cell 一个 gated elite 的 GSME archive，并提出 recombination。

The paper's Gated Semantic MAP-Elites has three parts: a categorical archive
keyed on the pathology a patch addresses (one elite per cell, entered only
after the gates), quality-biased selection (each round extends the current
best harness — the loop's greedy parent selection), and cross-cell
recombination (the best harness is stacked with elites from OTHER pathology
cells). The loop always had the middle part; this module adds the other two:

- :meth:`GsmeArchive.consider` observes every gated outcome. An accepted
  candidate whose full-train confirm beat the FIXED vanilla mean enters or
  replaces its cell's elite. Rejected, failed, and inconclusive outcomes never
  enter the archive.
- :meth:`GsmeArchive.eligible_elites` proposes recombination targets for a
  parent: elites from cells NOT already stacked into the parent's lineage,
  skipping pairings already tried and pairs whose edits touch the same files
  (full-file-bytes stacking would silently clobber one side of a same-file
  overlap, mismeasuring the stack — such pairs are skipped, not merged).

The archive never decides credit: a recombinant goes through the exact same
apply -> screen/confirm -> gate pipeline as a designed candidate. It only
steers which combinations get measured, so label noise in WHY degrades
coverage, never the quality of what is ultimately promoted (same argument as
the paper's descriptor-noise analysis).

state 每 round 持久化到 ``config.archive_path`` JSON，resume 保留 elite、lineage metadata 与
attempted pairing。archive 只选择“值得测”的组合，不授予 credit；recombinant 仍走完整 gate。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from codeflow.evolver.orchestrator.gates.policy import CandidateOutcome
from codeflow.evolver.tree.node import AppliedPatch, HarnessNode, PatchWhy


@dataclass(frozen=True)
class CellElite:
    """一个 ``WHERE x WHY`` cell 的 gated elite record。

    ``files``/``deletions`` 是相对自身 parent 的 repo-relative edit path；配合 git_commit_sha，
    recombiner 可从 commit 回读 bytes 并重现机制，无需在 JSON 保存 content，因此 resume-safe。
    ``credited`` 只表示 2sigma label，不改变 navigator elite 资格。
    """

    cell: str
    node_id: str
    git_commit_sha: str
    score: float
    round_index: int
    why: str
    where: str
    credited: bool = False
    files: tuple[str, ...] = ()
    deletions: tuple[str, ...] = ()
    focused_task_ids: tuple[str, ...] = ()
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "node_id": self.node_id,
            "git_commit_sha": self.git_commit_sha,
            "score": self.score,
            "round_index": self.round_index,
            "why": self.why,
            "where": self.where,
            "credited": self.credited,
            "files": list(self.files),
            "deletions": list(self.deletions),
            "focused_task_ids": list(self.focused_task_ids),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CellElite":
        return cls(
            cell=d["cell"],
            node_id=d["node_id"],
            git_commit_sha=d["git_commit_sha"],
            score=float(d["score"]),
            round_index=int(d["round_index"]),
            why=d["why"],
            where=d.get("where", "edit"),
            credited=bool(d.get("credited", False)),
            files=tuple(d.get("files") or ()),
            deletions=tuple(d.get("deletions") or ()),
            focused_task_ids=tuple(d.get("focused_task_ids") or ()),
            summary=d.get("summary", ""),
        )


@dataclass
class RecombinantCandidate:
    """把 cross-cell elite edit 重现到 current parent 的 candidate。

    对 bench shared contract 采用 duck type：files bytes、deletions、why、focused_task_ids、summary，
    因而可被 files/deletions/focused/outcome hook 原样消费，并像 designed candidate 一样走
    apply -> gate。``elite_node_id`` 用于 pairing audit；``has_beacon`` 继承 identical code bytes。
    """

    files: dict[str, bytes]
    why: str
    cell: str
    elite_node_id: str
    focused_task_ids: list[str] = field(default_factory=list)
    summary: str = ""
    deletions: list[str] = field(default_factory=list)
    # 从精英字节继承：携带信标的机制重新叠加后仍保留门控 b 归因，因为代码字节完全相同。
    has_beacon: bool = False


# 路径到杠杆的启发式映射，用于机械绑定 WHERE（论文附录 A：评测框架编辑表面的四类杠杆）。
# 保持模块级，便于布局特殊的基准动态修改或扩展；未知项默认归为运行时，因为在证明相反前，
# 代码编辑都视为控制流变更。
_KNOWLEDGE_MARKERS = ("skills/", "skill/", "memory")
_CONFIG_SUFFIXES = (".yaml", ".yml", ".json", ".toml", ".ini", ".cfg")
_PROMPT_SUFFIXES = (".md", ".txt", ".prompt")


def _lever_of_path(path: str) -> str:
    p = path.replace("\\", "/").lower()
    if any(m in p for m in _KNOWLEDGE_MARKERS):
        return "knowledge"
    if p.endswith(_CONFIG_SUFFIXES):
        return "config"
    if p.endswith(_PROMPT_SUFFIXES):
        return "prompt"
    return "runtime"


def bind_where(paths) -> str:
    """根据 patch touched files 机械绑定 WHERE lever。

    WHERE 从 artifact 读取、从不 self-declare，使该 axis noise-free，model error 全留在 WHY。
    多 lever patch 为 ``mixed``；无 path 的 metadata-only candidate 为 unknown ``edit``。
    """
    levers = {_lever_of_path(p) for p in paths}
    if not levers:
        return "edit"
    if len(levers) == 1:
        return levers.pop()
    return "mixed"


def cell_of(cand: Any) -> Optional[tuple[str, str]]:
    """返回 candidate 的 ``(WHERE lever, WHY)`` coordinate，无 WHY 时为 ``None``。

    WHERE 由 :func:`bind_where` 从 touched files 计算；driver-declared patch_where 只留在 node
    ledger audit，绝不决定 archive coordinate。
    """
    if isinstance(cand, AppliedPatch):
        why = cand.patch_why_extra if cand.patch_why == PatchWhy.other else cand.patch_why.value
        paths = [c.target_file for c in cand.components]
        return bind_where(paths), str(why)
    why = getattr(cand, "why", None)
    if why:
        changed, deleted = _touched_paths(cand)
        return bind_where(list(changed) + list(deleted)), str(why)
    return None


def _touched_paths(cand: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """返回 candidate edit 的 ``(changed, deleted)`` repo-relative paths。"""
    if isinstance(cand, AppliedPatch):
        return tuple(c.target_file for c in cand.components), ()
    files = getattr(cand, "files", None)
    changed = tuple(sorted(files)) if isinstance(files, dict) else ()
    deleted = tuple(getattr(cand, "deletions", None) or ())
    return changed, deleted


def describe_candidate(cand: Any) -> Optional[dict]:
    """生成 node ledger 使用的 JSON-safe candidate metadata。

    wired bench candidate 不是 :class:`AppliedPatch`，若不转换会丢失 WHERE/WHY/touched file/
    activation。无 coordinate 返回 None；有值时保留 summary、beacon、activation_spec、manifest
    与 recombination provenance。description 不验证这些声明。
    """
    coord = cell_of(cand)
    if coord is None:
        return None
    where, why = coord
    changed, deleted = _touched_paths(cand)
    d: dict[str, Any] = {
        "why": why,
        "where": where,
        "files": list(changed),
        "deletions": list(deleted),
        "summary": str(getattr(cand, "summary", "") or "")[:300],
        "has_beacon": bool(getattr(cand, "has_beacon", False)),
    }
    spec = getattr(cand, "activation_spec", None)
    if isinstance(spec, dict):
        d["activation_spec"] = spec
    manifest = getattr(cand, "manifest", None)
    to_dict = getattr(manifest, "to_dict", None)
    if callable(to_dict):
        d["manifest"] = to_dict()
    elite_id = getattr(cand, "elite_node_id", None)
    if elite_id:
        d["recombination_of"] = elite_id
    return d


class GsmeArchive:
    """持久化 GSME cell elite、lineage metadata 与 attempted pairing。

    ``node_meta`` records, for every PROMOTED node, which cells its lineage has
    stacked and which files that lineage touched — the exclusion sets
    :meth:`eligible_elites` filters recombination proposals with. ``pairings``
    records every (parent, elite) recombination already attempted (whatever
    its outcome), so a patience streak does not re-propose the same pair every
    round. A parent with no metadata (the root, or a pre-archive journal)
    just gets an empty lineage: redundant proposals are then caught by the
    pairing record 捕获，最后仍由 gate 测量/拒绝。实例 load failure 保守回到 empty state。
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._cells: dict[str, CellElite] = {}
        self._node_meta: dict[str, dict[str, list[str]]] = {}
        self._pairings: dict[str, dict[str, str]] = {}
        self._load()

    # ---- 观察 -------------------------------------------------------------

    def consider(
        self,
        *,
        parent_id: str,
        node: HarnessNode,
        cand: Any,
        outcome: CandidateOutcome,
        vanilla_train_mean: float,
        round_index: int,
    ) -> None:
        """观察一个 gated outcome，更新 pairing、promoted lineage 与 cell elite。

        recombination 总是记录 pairing。只有 outcome promoted、含 confirm eval、score beat fixed
        vanilla，且优于现有 cell elite 时才入 archive。rejected/failed/inconclusive 永不进入。
        """
        elite_id = getattr(cand, "elite_node_id", None)
        if elite_id:
            self.record_pairing(parent_id, elite_id, outcome.status.value)

        coord = cell_of(cand)
        if coord is None:
            return
        where, why = coord
        key = f"{where}::{why}"
        changed, deleted = _touched_paths(cand)

        if outcome.promoted:
            pmeta = self._node_meta.get(parent_id, {})
            self._node_meta[node.node_id] = {
                "cells": sorted(set(pmeta.get("cells", [])) | {key}),
                "files": sorted(set(pmeta.get("files", [])) | set(changed) | set(deleted)),
            }

        # 导航条（论文算法 1）：已接受的完整训练确认必须超过原始版本。无效、被拒绝或输给原始
        # 版本的结果绝不进入。
        if not outcome.promoted or not outcome.confirm_evals or outcome.score <= vanilla_train_mean:
            return
        prev = self._cells.get(key)
        if prev is not None and prev.score >= outcome.score:
            return
        self._cells[key] = CellElite(
            cell=key,
            node_id=node.node_id,
            git_commit_sha=node.git_commit_sha,
            score=outcome.score,
            round_index=round_index,
            why=why,
            where=where,
            credited=bool(outcome.paired and outcome.paired.credited_2sigma),
            files=changed,
            deletions=deleted,
            focused_task_ids=tuple(getattr(cand, "focused_task_ids", None) or ()),
            summary=str(getattr(cand, "summary", "") or "")[:300],
        )

    def record_pairing(self, parent_id: str, elite_node_id: str, status: str) -> None:
        self._pairings.setdefault(parent_id, {})[elite_node_id] = status

    # ---- 重组提案 ---------------------------------------------------------

    def eligible_elites(self, parent_id: str, *, limit: int = 1) -> list[CellElite]:
        """返回值得 stack 到 ``parent_id`` 的 elite，按 score descending。

        Excluded: cells already in the parent's lineage, the parent itself,
        pairings already attempted, and elites whose edit overlaps a file the
        lineage already changed (byte-level stacking cannot merge same-file
        edits, only replace — an overlapping "stack" would silently drop one
        mechanism 并测量谎言。``limit<=0`` 返回空 list。
        """
        if limit <= 0:
            return []
        meta = self._node_meta.get(parent_id, {})
        lineage_cells = set(meta.get("cells", []))
        lineage_files = set(meta.get("files", []))
        tried = self._pairings.get(parent_id, {})
        out: list[CellElite] = []
        ranked = sorted(self._cells.items(), key=lambda kv: (-kv[1].score, kv[0]))
        for key, elite in ranked:
            if key in lineage_cells:
                continue
            if elite.node_id == parent_id or elite.node_id in tried:
                continue
            if lineage_files & (set(elite.files) | set(elite.deletions)):
                continue
            out.append(elite)
            if len(out) >= limit:
                break
        return out

    # ---- 报告 -------------------------------------------------------------

    def summary_text(self) -> str:
        """生成每个 cell elite 一行的 design-prompt summary。

        empty archive 返回空 string；文本只提示 already banked mechanism，不是重新验证结果。
        """
        if not self._cells:
            return ""
        lines = ["ARCHIVE (verified elites, one per failure cell):"]
        for key in sorted(self._cells):
            e = self._cells[key]
            cred = " credited" if e.credited else ""
            lines.append(
                f"- {key}: {e.node_id} score={e.score:.3f}{cred} r{e.round_index}"
                + (f" — {e.summary}" if e.summary else "")
            )
        return "\n".join(lines)

    @property
    def cells(self) -> dict[str, CellElite]:
        return dict(self._cells)

    # ---- 持久化 ------------------------------------------------------------

    def save(self) -> None:
        """以 tmp+rename best-effort 保存 archive JSON。

        OSError 被吞掉，因此调用返回不证明 state 已持久化；resume 必须从文件实际回读。
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {
                    "cells": {k: e.to_dict() for k, e in self._cells.items()},
                    "node_meta": self._node_meta,
                    "pairings": self._pairings,
                },
                indent=2,
            )
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(payload)
            tmp.replace(self._path)
        except OSError:
            pass

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            d = json.loads(self._path.read_text())
            self._cells = {k: CellElite.from_dict(v) for k, v in (d.get("cells") or {}).items()}
            self._node_meta = {
                k: {"cells": list(v.get("cells", [])), "files": list(v.get("files", []))}
                for k, v in (d.get("node_meta") or {}).items()
            }
            self._pairings = {k: dict(v) for k, v in (d.get("pairings") or {}).items()}
        except (OSError, ValueError, KeyError):
            self._cells, self._node_meta, self._pairings = {}, {}, {}


__all__ = [
    "CellElite",
    "GsmeArchive",
    "RecombinantCandidate",
    "bind_where",
    "cell_of",
    "describe_candidate",
]
