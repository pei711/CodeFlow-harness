"""为 Evolver-generated patch 提供 path guard（spec §22 + §22.7）。

immutable set 是 evolver-immutable，不是 human-immutable。Human-driven PR review 仍可按正常
流程修改这些路径；只有 auto-evolver candidate 被阻止。``IMMUTABLE_PATTERNS`` 支持 exact
file，例如 ``codeflow/agent/loop/main.py``，以及 trailing slash directory subtree，例如
``codeflow/evolver/``，后者匹配目录自身及全部 descendant。``MUTABLE_OVERRIDES`` 优先，可从较宽
immutable directory 中 carve out mutable subtree。

模块刻意使用 repo-relative string matching，不用 full glob：目标是 fast、predictable、
easy-to-audit gate；``**``、``*.py`` 会引入 case sensitivity 与 slash behavior surprise，而
exact path + directory subtree 已无歧义覆盖 §22.2。

spec §22.1/§22.2 的 reference layers 是：L1 self-reference ``evolver/**``；L2 evaluation
substrate（eval_engine + external grader）；L3 capability contract（Agent loop、Tool framework、
provider、Skill loader、sandbox、config schema 等）；L4 audit/data integrity
``tool_audit_hook``；L5 tests、dependency、CI。guard 通过不表示 patch 内容正确，只表示目标
不触碰 Evolver 的信任根。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# 不可变模式列表（规范第 22.2 节）
# ---------------------------------------------------------------------------


# 所有路径均相对于仓库并使用正斜杠分隔；末尾斜杠表示该目录及其全部内容。
IMMUTABLE_PATTERNS: tuple[str, ...] = (
    # ── L1——自指 ─────────────────────────────────────────────────────────
    "codeflow/evolver/",
    # ── L2——评测基础设施 ─────────────────────────────────────────────────
    "codeflow/eval_engine/engine.py",
    "codeflow/eval_engine/adapter/",
    "codeflow/eval_engine/judge/",
    # AppWorld 演化胶水层（adapter/eval/diagnose/editor/precheck，以及执行 /evaluate 调用、
    # 成功/基础设施分类和结果写入的 grade.py）负责给候选项评分。它位于设计器白名单树内部，
    # 因此防护必须把它排除，对应上游 evaluation/ 入口。
    "benchmarks/appworld/evolve/",
    # 批量评分器编排试验并记录运行器级基础设施；允许候选项编辑它就等于允许改变自己的分母。
    # 可编辑的智能体表面是 agent_cli.py（循环/提示）和 tool.py。
    "benchmarks/appworld/batch.py",
    # ── L3——能力契约 ─────────────────────────────────────────────────────
    "codeflow/agent/loop/main.py",
    "codeflow/agent/context/",
    "codeflow/agent/tools/base.py",
    "codeflow/agent/tools/registry.py",
    "codeflow/agent/personalizer/",
    "codeflow/agent/subagent/",
    # CodeFlow 使用 spine 代替上游智能体 API；运行器是每轮必经的桥梁，属于同一契约层级。
    "codeflow/agent/spine_runner.py",
    "codeflow/spine/",
    "codeflow/providers/",
    "codeflow/session/",
    "codeflow/memory_engine/",
    "codeflow/context_engine/",
    "codeflow/sandbox/",
    "codeflow/security/",
    # 配置：schema/loader（.py）不可变，值文件（.yaml/.json）可变。
    "codeflow/config/__init__.py",
    "codeflow/config/codeflow.py",
    "codeflow/config/loader.py",
    "codeflow/config/paths.py",
    "codeflow/config/schema.py",
    "codeflow/config/update.py",
    "codeflow/config/update_channels.py",
    "codeflow/config/update_providers.py",
    # ── L4——审计/数据完整性 ──────────────────────────────────────────────
    "codeflow/eval_engine/hooks/tool_audit_hook.py",
    # ── L5——测试、依赖和 CI ──────────────────────────────────────────────
    "tests/",
    "pyproject.toml",
    "uv.lock",
    ".github/",
    # ── 核心版本（规范第 22.5 节）——仅人工递增 ───────────────────────────
    "codeflow/__init__.py",
)


# 不可变子树中仍应保持可变的路径例外。
MUTABLE_OVERRIDES: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


class ImmutablePathError(ValueError):
    """candidate target 命中 evolver-immutable kernel 时抛出的异常。

    kernel 定义见 spec §22。合法 patch 被挡时只能重构 candidate，拆出 mutable part 并丢弃
    immutable part；或走 human-driven development，PR 修改 ``MUTABLE_OVERRIDES``（若分类错误）
    或直接改 kernel，并按 spec §22.5 bump ``core_version``。auto-evolver 不得绕过。
    """


class UnsafePathError(ImmutablePathError):
    """patch path 不是安全 repository-relative path 时抛出的异常。

    包括 absolute/drive path、NUL、空组件、``.``/``..``、symlink traversal 或逃逸 repo root。
    """


# ---------------------------------------------------------------------------
# 内部辅助方法
# ---------------------------------------------------------------------------


def _normalise(path: str) -> str:
    """把输入规范化为唯一、无歧义的 repository-relative path。

    反斜杠转正斜杠并去除 leading ``./``；unsafe component 或非 string 抛出
    ``UnsafePathError``。函数不访问 filesystem。
    """
    if not isinstance(path, str):
        raise UnsafePathError(f"Patch path must be a string, got {type(path).__name__}")
    if "\x00" in path:
        raise UnsafePathError("Patch path contains a NUL byte")
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if not p:
        raise UnsafePathError("Patch path must not be empty")
    if p.startswith("/") or p.startswith("//") or re.match(r"^[A-Za-z]:/", p):
        raise UnsafePathError(f"Patch path must be repository-relative: {path!r}")
    parts = p.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise UnsafePathError(f"Patch path contains an unsafe component: {path!r}")
    return p


def _assert_no_symlink(path: str, repo_root: Path) -> None:
    root = repo_root.resolve(strict=True)
    cursor = root
    for part in path.split("/"):
        cursor /= part
        if cursor.is_symlink():
            raise UnsafePathError(f"Patch path traverses a symlink: {path!r}")
    resolved = (root / path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise UnsafePathError(f"Patch path escapes the repository: {path!r}") from exc


def _assert_no_tree_symlink(path: str, repo_root: Path, treeish: str) -> None:
    from codeflow.evolver.tree import git_ops

    try:
        traverses_symlink = git_ops.tree_path_has_symlink(repo_root, treeish, path)
    except git_ops.GitOpError as exc:
        raise UnsafePathError(f"Could not inspect patch path in parent tree {treeish!r}: {path!r}") from exc
    if traverses_symlink:
        raise UnsafePathError(f"Patch path traverses a symlink in {treeish}: {path!r}")


def _match(path: str, pattern: str) -> bool:
    """按单个 guard pattern 匹配 normalized path。

    trailing slash 表示 directory subtree，匹配目录自身及 descendant；其他 pattern exact
    string match。两侧使用 ``casefold``，使判定不依赖 host filesystem case behavior。
    """
    candidate = path.casefold()
    expected = pattern.casefold()
    if expected.endswith("/"):
        prefix = expected.rstrip("/")
        return candidate == prefix or candidate.startswith(expected)
    return candidate == expected


# ---------------------------------------------------------------------------
# 谓词
# ---------------------------------------------------------------------------


def is_immutable(path: str) -> bool:
    """判断 ``path`` 是否属于 evolver-immutable kernel。

    先 normalize Windows path/leading ``./``；unsafe path 保守返回 ``True``。随后先检查
    ``MUTABLE_OVERRIDES``，匹配即 mutable；再检查 ``IMMUTABLE_PATTERNS``，匹配即 immutable；
    否则 default mutable。该函数不做 symlink filesystem 检查。
    """
    try:
        norm = _normalise(path)
    except UnsafePathError:
        return True
    for pat in MUTABLE_OVERRIDES:
        if _match(norm, pat):
            return False
    for pat in IMMUTABLE_PATTERNS:
        if _match(norm, pat):
            return True
    return False


def check_patch_paths(
    target_files: Iterable[str],
    *,
    repo_root: str | Path | None = None,
    treeish: str | None = None,
) -> list[str]:
    """返回 ``target_files`` 中 unsafe 或命中 immutable pattern 的 subset。

    供调用方在不抛错时检查 violation，例如按 spec §21.4.2 把 patch 路由到 TODO markdown。
    提供 ``repo_root`` 时检查 current checkout symlink；同时提供 ``treeish`` 时改查对应 Git
    tree。只有 ``treeish`` 没有 repo_root 会抛 ``ValueError``。offender 保留原输入 spelling。
    """
    offenders: list[str] = []
    root = Path(repo_root) if repo_root is not None else None
    if treeish is not None and root is None:
        raise ValueError("treeish requires repo_root")
    for path in target_files:
        try:
            norm = _normalise(path)
            if root is not None:
                if treeish is None:
                    _assert_no_symlink(norm, root)
                else:
                    _assert_no_tree_symlink(norm, root, treeish)
        except (UnsafePathError, OSError, ValueError):
            offenders.append(path)
            continue
        if is_immutable(norm):
            offenders.append(path)
    return offenders


def assert_patch_allowed(
    target_files: Iterable[str],
    *,
    repo_root: str | Path | None = None,
    treeish: str | None = None,
) -> None:
    """任一 target unsafe/immutable 时抛出 :class:`ImmutablePathError`。

    在 Evolver applier 中作为 first-line gate：

    .. code-block:: python

        from codeflow.evolver.applier import assert_patch_allowed
        assert_patch_allowed([c.target_file for c in patch.components])
        # ... 仅在没有错误时继续应用补丁

    error 最多列出前 5 个 offender，并报告剩余数量。无 offender 时无返回值；通过只授权进入
    后续验证，不授权实际 promote。
    """
    offenders = check_patch_paths(
        target_files,
        repo_root=repo_root,
        treeish=treeish,
    )
    if offenders:
        head = offenders[:5]
        more = f" ... and {len(offenders) - 5} more" if len(offenders) > 5 else ""
        raise ImmutablePathError(
            f"Patch targets {len(offenders)} evolver-immutable path(s): "
            f"{head}{more}. The gates, ledgers, and eval glue may never be "
            "edited by a candidate (see IMMUTABLE_PATTERNS in this module)."
        )


__all__ = [
    "IMMUTABLE_PATTERNS",
    "MUTABLE_OVERRIDES",
    "ImmutablePathError",
    "UnsafePathError",
    "assert_patch_allowed",
    "check_patch_paths",
    "is_immutable",
]
