"""实现 host-side 的 grep 内容搜索与 find 文件查找 Tool。

两者复用 ``_FsTool`` 的 Workspace/allowed_dir 解析，因此与 read_file/write_file/list_dir 具有
完全相同的路径边界；它们 never 使用 SandboxExecutor，避免把大结果集跨 VM edge 搬运。
系统根、伪文件系统和驱动器根会被拒绝，纯 Python walk 还有 wall-clock deadline，防止模型
无意遍历整台宿主机。

``grep`` 优先使用 PATH 中的 ``rg``（ripgrep）获得速度与 .gitignore 感知，缺失时回退纯 Python，
所以 CodeFlow 没有 hard binary dependency。两条路径都限制结果数量与字符数，并在 partial result
中明确警告，避免模型把被截断视图当成完整计数。
"""

import asyncio
import fnmatch
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from codeflow.agent.tools.execution import ToolCapability, ToolEffect
from codeflow.agent.tools.filesystem import _FsTool

# 纯 Python 回退和 find 跳过的噪声目录。ripgrep 通过 .gitignore 自行处理忽略逻辑，
# 因此此集合只限制回退路径。
_IGNORE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".coverage",
    "htmlcov",
}

# 绝不允许树形遍历的伪文件系统和系统根目录。否则模型执行 `grep <pat> /` 或在 `/`
# 上 find 时会遍历整台宿主机，包括 /proc、/sys 或 /mnt 下的慢速网络挂载，使运行无限挂起。
# 搜索必须指定真实子树，而不是系统根目录。
_DENY_TRAVERSAL_ROOTS = {Path(p) for p in ("/", "/proc", "/sys", "/dev", "/run", "/boot")}
# 纯 Python os.walk 回退的墙上时间上限，使允许但巨大的目录树也无法挂起循环。
# ripgrep 已有自己的 _RG_TIMEOUT。
_WALK_DEADLINE_S = 20.0


def _denied_traversal_root(base: Path) -> bool:
    """判断 ``base`` 是否解析为禁止 Tree walk 的系统或文件系统根。

    resolve 失败时返回 ``False``，让后续存在性/I/O 路径给出正常错误。Resolved path 的 Parent
    等于自身时视为任意 filesystem/drive root，可覆盖 POSIX ``/``、Windows drive 与 UNC root；
    另外显式拒绝 /proc、/sys、/dev、/run、/boot。返回值只限制递归目录，普通文件不受影响。
    """
    try:
        resolved = base.resolve()
    except OSError:
        return False
    # 任何文件系统或驱动器根目录都是自身的父目录。这能捕获 POSIX 的 "/" 以及
    # _DENY_TRAVERSAL_ROOTS 会漏掉的 Windows 驱动器和 UNC 根路径，避免遍历整个驱动器。
    if resolved.parent == resolved:
        return True
    return resolved in _DENY_TRAVERSAL_ROOTS


# ---------------------------------------------------------------------------
# 内容搜索
# ---------------------------------------------------------------------------


class GrepTool(_FsTool):
    """按 Regular expression 搜索文件内容，优先 ripgrep 并带纯 Python fallback。

    Tool 支持 content、files_with_matches、count 三种 output_mode、可选 glob、大小写忽略、上下文
    行和 limit。Regex 会先编译验证；目录解析为系统根时拒绝。rg 路径设置 30 秒 timeout、统一
    path separator 和 noise exclude；fallback 跳过 binary、受 20 秒 walk deadline 约束。

    返回最多 30,000 字符，并在 limit/字符截断时标明 PARTIAL result，提示用 count 或更窄模式。
    无命中是正常 ``No matches found.``，不是 failed ToolResult。该 Tool 是 concurrency-safe READ。
    """

    capability = ToolCapability(effect=ToolEffect.READ, concurrency_safe=True)
    _MAX_CHARS = 30_000
    _DEFAULT_LIMIT = 100
    _RG_TIMEOUT = 30

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents by regular expression. Prefer this over running "
            "grep/rg through exec — results are paginated, capped, and .gitignore-aware. "
            "output_mode 'content' returns matching lines with path:line numbers, "
            "'files_with_matches' lists only file paths, 'count' shows match counts per file. "
            "Use glob to restrict to file types (e.g. '*.py')."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression to search for"},
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (default: workspace root)",
                },
                "glob": {
                    "type": "string",
                    "description": "Only search files matching this glob (e.g. '*.py', '*.{ts,tsx}')",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "Output format (default: content)",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive matching (default false)",
                },
                "context": {
                    "type": "integer",
                    "description": "Lines of context before and after each match, content mode only (default 0)",
                    "minimum": 0,
                    "maximum": 20,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max matching lines (content) or files (other modes) to return (default 100)",
                    "minimum": 1,
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        output_mode: str = "content",
        case_insensitive: bool = False,
        context: int = 0,
        limit: int | None = None,
        **kwargs: Any,
    ) -> str:
        cap = limit or self._DEFAULT_LIMIT
        try:
            re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regular expression: {e}"
        try:
            base = self._resolve(path)
        except PermissionError as e:
            return f"Error: {e}"
        if not base.exists():
            return f"Error: path not found: {path}"
        if base.is_dir() and _denied_traversal_root(base):
            return (
                f"Error: refusing to search '{path}' — it resolves to a system root "
                f"({base.resolve()}). Searching the whole filesystem hangs the agent. "
                "Specify a narrower directory (e.g. the workspace or a project subtree)."
            )

        rg = shutil.which("rg")
        try:
            if rg:
                return await self._run_rg(rg, pattern, base, glob, output_mode, case_insensitive, context, cap)
            return self._run_python(pattern, base, glob, output_mode, case_insensitive, context, cap)
        except Exception as e:
            return f"Error running grep: {e}"

    async def _run_rg(
        self,
        rg: str,
        pattern: str,
        base: Path,
        glob: str | None,
        output_mode: str,
        case_insensitive: bool,
        context: int,
        cap: int,
    ) -> str:
        args = [rg, "--color=never"]
        if case_insensitive:
            args.append("-i")
        if glob:
            args += ["-g", glob]
        # rg 只在 .gitignore 声明时跳过噪声目录；添加显式排除，使其无论仓库状态如何都与
        # 纯 Python 回退一致。排除规则放在用户 glob 之后，使它在“最后匹配获胜”顺序中生效。
        for d in _IGNORE_DIRS:
            args += ["-g", f"!{d}"]

        if output_mode == "files_with_matches":
            args.append("-l")
        elif output_mode == "count":
            args.append("-c")
        else:
            args += ["--line-number", "--no-heading", "--with-filename"]
            if context:
                args += ["-C", str(context)]
        # 在所有平台上强制 rg 输出路径使用正斜杠，使 Windows 和 POSIX 的结果一致；
        # 该设置只影响路径字段，不影响匹配内容。
        args += ["--path-separator", "/"]
        args += ["-e", pattern, "--", str(base)]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=self._RG_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Error: grep timed out after {self._RG_TIMEOUT}s"
        except BaseException:
            proc.kill()
            await proc.wait()
            raise

        # 无匹配时 rg 以 1 退出，这是正常的空结果。
        if proc.returncode not in (0, 1):
            return f"Error running rg: {err.decode('utf-8', 'replace').strip()}"

        text = out.decode("utf-8", "replace")
        # 将路径转成搜索根目录的相对路径，使输出紧凑易读。rg 使用正斜杠路径；
        # 移除搜索根前缀时使用相同格式，保证 Windows 上也能正常处理。
        base_str = str(base).replace(os.sep, "/")
        text = text.replace(base_str + "/", "").replace(base_str, base.name or ".")
        lines = [ln for ln in text.splitlines() if ln]
        if not lines:
            return "No matches found."

        unit = "matching lines" if output_mode == "content" else "files"
        return self._format_lines(lines, cap, unit)

    def _run_python(
        self,
        pattern: str,
        base: Path,
        glob: str | None,
        output_mode: str,
        case_insensitive: bool,
        context: int,
        cap: int,
    ) -> str:
        flags = re.IGNORECASE if case_insensitive else 0
        rx = re.compile(pattern, flags)
        files = self._iter_files(base, glob)

        content_lines: list[str] = []
        match_files: list[str] = []
        counts: list[tuple[str, int]] = []

        for fp in files:
            rel = self._relpath(fp, base)
            try:
                raw = fp.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:8192]:  # 跳过二进制文件
                continue
            text_lines = raw.decode("utf-8", "replace").splitlines()

            hits = [i for i, line in enumerate(text_lines) if rx.search(line)]
            if not hits:
                continue

            if output_mode == "files_with_matches":
                match_files.append(rel)
            elif output_mode == "count":
                counts.append((rel, len(hits)))
            else:
                self._collect_content(content_lines, rel, text_lines, hits, context)

        if output_mode == "files_with_matches":
            return self._format_lines(match_files, cap, "files") if match_files else "No matches found."
        if output_mode == "count":
            rendered = [f"{rel}:{n}" for rel, n in counts]
            return self._format_lines(rendered, cap, "files") if rendered else "No matches found."
        return self._format_lines(content_lines, cap, "matching lines") if content_lines else "No matches found."

    @staticmethod
    def _collect_content(
        out: list[str],
        rel: str,
        text_lines: list[str],
        hits: list[int],
        context: int,
    ) -> None:
        emitted: set[int] = set()
        for h in hits:
            lo = max(0, h - context)
            hi = min(len(text_lines), h + context + 1)
            for i in range(lo, hi):
                if i in emitted:
                    continue
                emitted.add(i)
                sep = ":" if i == h or context == 0 else "-"
                out.append(f"{rel}{sep}{i + 1}{sep}{text_lines[i]}")

    def _iter_files(self, base: Path, glob: str | None):
        if base.is_file():
            yield base
            return
        deadline = time.monotonic() + _WALK_DEADLINE_S
        for root, dirs, names in os.walk(base):
            if time.monotonic() > deadline:
                # 遇到意外巨大或慢速的目录树时停止，而不是挂起。
                break
            dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
            for n in sorted(names):
                if glob and not fnmatch.fnmatch(n, glob):
                    continue
                yield Path(root) / n

    @staticmethod
    def _relpath(fp: Path, base: Path) -> str:
        try:
            return fp.relative_to(base if base.is_dir() else base.parent).as_posix()
        except ValueError:
            return fp.as_posix()

    def _format_lines(self, lines: list[str], cap: int, unit: str) -> str:
        total = len(lines)
        shown = lines[:cap]
        result = "\n".join(shown)
        notes = []
        if total > cap:
            notes.append(
                f"showing first {cap} of {total} {unit} — {total - cap} more not shown; "
                "this is a PARTIAL result, do not treat it as the complete set or count "
                "from it. Use output_mode='count' for exact totals, or a narrower pattern/glob."
            )
        if len(result) > self._MAX_CHARS:
            result = result[: self._MAX_CHARS]
            notes.append(
                f"output truncated to {self._MAX_CHARS} chars — narrow the pattern/glob or "
                "use output_mode='count' to get exact totals instead of eyeballing this view"
            )
        if notes:
            result += f"\n\n(⚠️ {'; '.join(notes)})"
        return result


# ---------------------------------------------------------------------------
# 文件查找
# ---------------------------------------------------------------------------


class FindTool(_FsTool):
    """使用 pathlib 按 Glob pattern 查找文件，并按最近修改时间排序。

    Pattern 含路径分隔符时按字面 glob；只有 basename 时自动扩为 ``**/pattern``，形成 fd-style
    任意深度查找。目标必须是允许范围内的真实目录，系统/drive root 被拒绝，noise directories
    始终跳过。结果返回相对搜索根路径，默认最多 1000 条；无匹配返回
    ``No files found matching pattern.``。它是 pure-Python、concurrency-safe READ，不调用 Shell。
    """

    capability = ToolCapability(effect=ToolEffect.READ, concurrency_safe=True)
    _DEFAULT_LIMIT = 1000

    @property
    def name(self) -> str:
        return "find"

    @property
    def description(self) -> str:
        return (
            "Find files by glob pattern (e.g. '*.py', 'src/**/*.ts'). Prefer this over "
            "running find/ls through exec. Returns paths relative to the search root, "
            "most-recently-modified first. Noise directories (.git, node_modules, etc.) "
            "are skipped."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. '*.py' or 'src/**/*.ts'",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: workspace root)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default 1000)",
                    "minimum": 1,
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        limit: int | None = None,
        **kwargs: Any,
    ) -> str:
        cap = limit or self._DEFAULT_LIMIT
        try:
            base = self._resolve(path)
        except PermissionError as e:
            return f"Error: {e}"
        if not base.exists():
            return f"Error: path not found: {path}"
        if not base.is_dir():
            return f"Error: not a directory: {path}"
        if _denied_traversal_root(base):
            return (
                f"Error: refusing to search '{path}' — it resolves to a system root "
                f"({base.resolve()}). Specify a narrower directory."
            )

        # 带路径的模式按字面执行 glob；不带路径的模式递归匹配基名（fd 风格），
        # 因此 'foo.py' 可在任意深度被找到。
        glob_expr = pattern if "/" in pattern else f"**/{pattern}"
        try:
            matches = [
                p for p in base.glob(glob_expr) if not any(part in _IGNORE_DIRS for part in p.relative_to(base).parts)
            ]
        except (ValueError, OSError) as e:
            return f"Error running find: {e}"
        if not matches:
            return "No files found matching pattern."

        matches.sort(key=lambda p: self._mtime(p), reverse=True)
        total = len(matches)
        shown = matches[:cap]
        lines = [f"{p.relative_to(base).as_posix()}/" if p.is_dir() else p.relative_to(base).as_posix() for p in shown]
        result = "\n".join(lines)
        if total > cap:
            result += f"\n\n(showing first {cap} of {total} results)"
        return result

    @staticmethod
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0
