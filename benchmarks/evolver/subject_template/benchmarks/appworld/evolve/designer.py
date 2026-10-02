"""The design step: one model call per candidate, one fenced block back.

The designer is deliberately narrow. It shows the model the current module
source plus the failure evidence for a single WHY and asks for the complete
corrected file inside exactly one fenced block. The parser refuses anything
else, so the only thing a model response can become is the full bytes of one
declared file - there is no shell, no patch application, and no path the model
can name.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Callable

from benchmarks.appworld.evolve.adapter import MODULE_PATH
from benchmarks.appworld.evolve.tasks import PUBLIC_SURFACE, WHY_DEFINITIONS, task_for
from codeflow.evolver.orchestrator.nodes.semantic import SemanticNode
from codeflow.evolver.tree import git_ops

MAX_MODULE_BYTES = 32768

_FENCE_RE = re.compile(r"```[A-Za-z0-9_+.-]*\r?\n(.*?)```", re.DOTALL)


class DesignParseError(ValueError):
    """The design response is not one usable module body."""


def parse_module_block(
    text: str,
    *,
    required_names: tuple[str, ...] = (),
    max_bytes: int = MAX_MODULE_BYTES,
) -> str:
    """Extract the single fenced module body from a design response."""
    blocks = _FENCE_RE.findall(text or "")
    if len(blocks) != 1:
        raise DesignParseError(f"expected exactly one fenced code block, found {len(blocks)}")
    source = blocks[0]
    if not source.strip():
        raise DesignParseError("the fenced code block is empty")
    size = len(source.encode("utf-8"))
    if size > max_bytes:
        raise DesignParseError(f"the fenced code block is {size} bytes, over the {max_bytes} byte cap")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise DesignParseError(f"the fenced code block is not valid Python: {exc}") from exc
    defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = sorted(name for name in required_names if name not in defined)
    if missing:
        raise DesignParseError(f"the module body must still define {missing}")
    return source


def _repair_prompt(error: Exception) -> str:
    return (
        f"That response could not be used: {error}. Answer again with the complete corrected "
        "module and nothing else: exactly one fenced code block, no prose before or after it."
    )


def _render_failures(failures: dict[str, list[dict]]) -> str:
    lines = []
    for task_id in sorted(failures):
        task = task_for(task_id)
        lines.append(f"- task {task_id} ({task.description}):")
        for case in failures[task_id]:
            if "fn" not in case:
                lines.append(f"    {case.get('detail', 'no detail')}")
                continue
            args = ", ".join(repr(arg) for arg in case.get("args", []))
            lines.append(
                f"    {case['fn']}({args}) -> {case.get('actual')!r}, expected {case.get('expect')!r}"
                + (f" [raised: {case['error']}]" if case.get("error") else "")
            )
    return "\n".join(lines)


def build_design_messages(*, source: str, why: str, failures: dict[str, list[dict]]) -> list[dict[str, str]]:
    definition = WHY_DEFINITIONS.get(why, why)
    return [
        {
            "role": "system",
            "content": (
                "You repair a small Python module. You always answer with the complete corrected "
                "module inside exactly one fenced code block and no other text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"File: {MODULE_PATH}\n\n"
                "Current source:\n\n"
                f"```python\n{source}\n```\n\n"
                f"Failure class under repair: {why}\n{definition}\n\n"
                "Observed failures:\n"
                f"{_render_failures(failures)}\n\n"
                "Rules:\n"
                f"- keep these public functions with their current signatures: {', '.join(PUBLIC_SURFACE)}\n"
                "- use the Python standard library only\n"
                "- do not import modules; only `from __future__ import annotations` is allowed\n"
                "- other cases currently pass, so do not change their behaviour\n"
                "- reply with the complete corrected file in exactly one fenced code block"
            ),
        },
    ]


def make_design_fn(
    call_fn: Callable[[list], str],
    *,
    repo_root: str | Path,
    sha_of,
    budget,
    evidence_of,
    max_retries: int = 2,
):
    """Build the loop's ``design_fn`` over the parent's failure evidence."""
    repo_root = Path(repo_root)

    def parse(raw: str) -> str:
        return parse_module_block(raw, required_names=PUBLIC_SURFACE)

    def design_fn(round_index: int, failure_map: dict, parent) -> list:
        from benchmarks.appworld.evolve.candidate import Candidate

        evidence = evidence_of(parent)
        if not evidence:
            return []
        parent_sha = sha_of(parent)
        source = git_ops.read_file_at(repo_root, parent_sha, MODULE_PATH).decode("utf-8")
        node: SemanticNode[str] = SemanticNode(
            name=f"design:r{round_index}",
            call_fn=call_fn,
            parse_fn=parse,
            parse_error_types=(DesignParseError,),
            max_retries=max_retries,
            repair_prompt=_repair_prompt,
        )
        candidates = []
        for why in sorted(evidence)[: budget.max_why_per_round]:
            failures = evidence[why]
            messages = build_design_messages(source=source, why=why, failures=failures)
            for _ in range(budget.candidates_per_why):
                new_source = node.run(messages)
                candidates.append(
                    Candidate(
                        files={MODULE_PATH: new_source.encode("utf-8")},
                        why=why,
                        focused_task_ids=sorted(failures),
                        summary=f"rewrite {MODULE_PATH} to repair {why}",
                    )
                )
        return candidates

    return design_fn


__all__ = [
    "MAX_MODULE_BYTES",
    "DesignParseError",
    "build_design_messages",
    "make_design_fn",
    "parse_module_block",
]
