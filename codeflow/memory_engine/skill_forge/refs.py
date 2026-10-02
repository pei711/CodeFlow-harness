"""Skill Body 中 Local Ref Paths 的 Resolution。

模块把 ``{baseDir}/x`` Placeholder 与指向 Bundled Files 的 Markdown Links 转为 Rooted at Skill Directory
的 Absolute Paths。支持 ``references/``、``scripts/``、``assets/``、``examples/``。Active-skills Render
Path（:class:`LocalSkillCatalog`）与 Router-hits Post-gate Hydrate Path（`SkillsSegmentBuilder`）共同使用，
保证 Two Flows 产生 Identical Bodies。

Resolution 对每个 Ref 做 Existence Check：Target Missing 的 ``{baseDir}/x`` 保持 Literal，而不是交给
Agent 一个 Confident 404。Code Fences 完全跳过，避免 Example Markup 被 Silently Mutated。路径解析成功
只证明文件存在，不证明内容安全或已被读取。
"""

from __future__ import annotations

import re
from pathlib import Path

_BUNDLED_DIRS = ("references", "scripts", "assets", "examples")

_MD_LINK_RE = re.compile(
    r"\[([^\]]+)\]\((?:\.{0,2}/)?"
    rf"((?:{'|'.join(_BUNDLED_DIRS)})/[^)\s]+)\)"
)
_BASE_DIR_REF_RE = re.compile(r"\{baseDir\}/(\S+?)(?=[\s)\'\"`]|$)")
_BARE_BASE_DIR_RE = re.compile(r"\{baseDir\}(?!/)")
_CODE_FENCE_RE = re.compile(r"(```.*?```)", re.S)


def resolve_refs(body: str, skill_dir: Path | str | None) -> tuple[str, bool]:
    """返回 ``(rewritten_body, any_resolved)``。

    ``skill_dir`` 是 ``SKILL.md`` 所在目录，Bundled Files 位于例如
    ``<skill_dir>/references/x.md``。为 `None` 或非真实 Directory 时，函数把 ``{baseDir}/`` Strip 成 Bare
    Relative Paths，并保持 Markdown Links；Agent 至少不会在 Prompt 中看到无意义 Literal Placeholder，
    虽然 ``references/x.md`` 仍无法自动 Resolve。

    至少一个 Substitution Materialized Real On-disk Path 时 ``any_resolved=True``，Caller 可据此输出
    ``Skill directory: ...`` Hint Header。Empty Body 返回 ``("", False)``。存在 Fragment/Query 的 Link
    会只检查基础文件，再保留后缀。
    """
    if not body:
        return "", False

    skill_path = Path(skill_dir) if skill_dir is not None else None
    has_dir = skill_path is not None and skill_path.is_dir()

    if not has_dir:
        if "{baseDir}" in body:
            body = body.replace("{baseDir}/", "").replace("{baseDir}", "")
        return body, False

    base_dir = str(skill_path)
    any_resolved = False

    def _md_sub(mo: re.Match[str]) -> str:
        nonlocal any_resolved
        rel = mo.group(2).rstrip(".,;:")
        cut = min((i for i in (rel.find("#"), rel.find("?")) if i != -1), default=-1)
        frag = rel[cut:] if cut != -1 else ""
        rel_file = rel[:cut] if cut != -1 else rel
        if rel_file and (skill_path / rel_file).exists():
            any_resolved = True
            return f"[{mo.group(1)}]({base_dir}/{rel_file}{frag})"
        return mo.group(0)

    segments = _CODE_FENCE_RE.split(body)
    body = "".join(seg if seg.startswith("```") else _MD_LINK_RE.sub(_md_sub, seg) for seg in segments)

    if "{baseDir}" in body:

        def _bd_sub(mo: re.Match[str]) -> str:
            nonlocal any_resolved
            ref = mo.group(1).rstrip(".,;:")
            if ref and (skill_path / ref).exists():
                any_resolved = True
                return f"{base_dir}/{mo.group(1)}"
            return mo.group(0)

        body = _BASE_DIR_REF_RE.sub(_bd_sub, body)
        # 使用函数而非字符串替换：base_dir 是文件系统路径，Windows 上包含反斜杠，
        # 否则 re.subn 会将其解释为转义序列（\U、\a 等），并抛出 ``bad escape``。
        body, bare_n = _BARE_BASE_DIR_RE.subn(lambda _m: base_dir, body)
        if bare_n:
            any_resolved = True

    return body, any_resolved


__all__ = ["resolve_refs"]
