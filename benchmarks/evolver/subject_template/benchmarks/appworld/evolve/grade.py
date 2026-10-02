"""The grading process: run assertion cases against one module file.

This module is executed as a script in a fresh interpreter
(``python grade.py`` with a JSON job on stdin), so it imports nothing but the
standard library and never touches the subject package. The candidate's module
is loaded from an extracted copy by path, which is why a candidate cannot ship
anything except the one file the manifest declares.

Job shape::

    {"module_path": "...", "out_dir": "...",
     "trials": [{"task_id": "...", "k": 0,
                 "cases": [{"fn": "...", "args": [...], "expect": ...}]}]}

One result file per trial, ``<out_dir>/<task_id>_k<k>.json``, carrying
``success`` and an ``infra_error`` marker so the reader can tell "the code
failed" from "the measurement failed".
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path

MODULE_NAME = "small_real_subject_agent_cli"

_FORBIDDEN_NAMES = {
    "BaseException",
    "GeneratorExit",
    "KeyboardInterrupt",
    "SystemExit",
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exit",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}


def _validate_candidate_source(module_path: str) -> None:
    source = Path(module_path).read_text()
    tree = ast.parse(source, filename=module_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raise ValueError("candidate modules may not import modules")
        if isinstance(node, ast.ImportFrom):
            if node.module != "__future__" or any(alias.name != "annotations" for alias in node.names):
                raise ValueError("candidate modules may only import future annotations")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("candidate modules may not mutate outer namespaces")
        if isinstance(node, ast.Name) and (
            node.id in _FORBIDDEN_NAMES or (node.id.startswith("__") and node.id != "__all__")
        ):
            raise ValueError(f"candidate modules may not use name {node.id!r}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"candidate modules may not access private attribute {node.attr!r}")


def _matches(actual: object, expect: object) -> bool:
    if isinstance(actual, bool) or isinstance(expect, bool):
        return actual is expect
    if isinstance(actual, (int, float)) and isinstance(expect, (int, float)):
        return math.isclose(float(actual), float(expect), rel_tol=0.0, abs_tol=1e-9)
    return actual == expect


def _load_module(module_path: str):
    _validate_candidate_source(module_path)
    spec = importlib.util.spec_from_file_location(MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load a module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_case(module, case: dict) -> dict:
    record = {
        "fn": case["fn"],
        "args": list(case["args"]),
        "expect": case["expect"],
        "actual": None,
        "ok": False,
        "error": None,
    }
    fn = getattr(module, case["fn"], None)
    if not callable(fn):
        record["error"] = f"module has no callable {case['fn']!r}"
        return record
    try:
        actual = fn(*case["args"])
    except Exception:  # noqa: BLE001 - 候选项崩溃表示用例失败，而不是整次运行崩溃
        record["error"] = traceback.format_exc(limit=3).strip().splitlines()[-1]
        return record
    try:
        json.dumps(actual)
        record["actual"] = actual
    except (TypeError, ValueError):
        record["actual"] = repr(actual)
    record["ok"] = _matches(actual, case["expect"])
    return record


def write_result(out_dir: Path, record: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record['task_id']}_k{record['k']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


def run_job(job: dict) -> None:
    out_dir = Path(job["out_dir"])
    module = None
    module_error = None
    try:
        module = _load_module(job["module_path"])
    except Exception:  # noqa: BLE001 - 无法导入的候选项在所有任务上都失败
        module_error = traceback.format_exc(limit=3).strip().splitlines()[-1]
    for trial in job["trials"]:
        if module_error is not None:
            record = {
                "task_id": trial["task_id"],
                "k": trial["k"],
                "success": False,
                "infra_error": None,
                "detail": f"module import failed: {module_error}",
                "cases": [],
            }
        else:
            cases = [_run_case(module, case) for case in trial["cases"]]
            record = {
                "task_id": trial["task_id"],
                "k": trial["k"],
                "success": bool(cases) and all(case["ok"] for case in cases),
                "infra_error": None,
                "detail": "",
                "cases": cases,
            }
        write_result(out_dir, record)


def main() -> int:
    run_job(json.loads(sys.stdin.read()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
