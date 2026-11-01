"""Frozen repository task definition and isolated subprocess runner."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.codeflowbench.canonical import canonical_digest

from .trajectory import score_trace

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    repository: Path
    commit: str
    prompt: str
    agent_command: tuple[str, ...]
    fail_to_pass: tuple[tuple[str, ...], ...]
    pass_to_pass: tuple[tuple[str, ...], ...]
    protected_paths: tuple[str, ...]
    timeout_seconds: int = 900
    token_budget: int | None = None
    expected_initial_failures: int | None = None
    environment: dict[str, str] | None = None


def load_task(path: Path) -> CodingTask:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "codeflow.coding-task.v1":
        raise ValueError("unsupported coding task schema (expected codeflow.coding-task.v1)")
    task_id = _required(raw, "task_id")
    if not _SAFE_ID.fullmatch(task_id):
        raise ValueError(f"task_id must be a portable identifier: {task_id!r}")
    repository = Path(_required(raw, "repository")).expanduser().resolve()
    if not (repository / ".git").exists():
        raise ValueError(f"repository is not a Git checkout: {repository}")
    commit = _required(raw, "commit")
    _git(repository, "cat-file", "-e", f"{commit}^{{commit}}")
    task = CodingTask(
        task_id=task_id,
        repository=repository,
        commit=commit,
        prompt=_required(raw, "prompt"),
        agent_command=_argv(raw.get("agent_command"), "agent_command"),
        fail_to_pass=_commands(raw.get("fail_to_pass", []), "fail_to_pass"),
        pass_to_pass=_commands(raw.get("pass_to_pass", []), "pass_to_pass"),
        protected_paths=_paths(raw.get("protected_paths", [])),
        timeout_seconds=_positive_int(raw.get("timeout_seconds", 900), "timeout_seconds"),
        token_budget=_optional_positive_int(raw.get("token_budget"), "token_budget"),
        expected_initial_failures=_optional_nonnegative_int(
            raw.get("expected_initial_failures"), "expected_initial_failures"
        ),
        environment=_environment(raw.get("environment", {})),
    )
    if not task.fail_to_pass:
        raise ValueError("at least one fail_to_pass command is required")
    if task.expected_initial_failures is None or task.expected_initial_failures < 1:
        raise ValueError("expected_initial_failures must be set to a positive number")
    if task.expected_initial_failures > len(task.fail_to_pass):
        raise ValueError("expected_initial_failures cannot exceed fail_to_pass command count")
    if not task.protected_paths:
        raise ValueError("protected_paths must list the test/verifier files the Agent must not change")
    return task


def run_task(task: CodingTask, output_root: Path, *, repetition: int = 1) -> dict[str, Any]:
    """Run baseline, agent, then external test commands in a fresh local clone.

    The subprocess runs with isolated CodeFlow home and trace directories. This
    is workspace isolation, not an OS sandbox. Task definitions and commands
    must be trusted. Token budgets are reported as post-run limits because a
    generic CLI adapter cannot prevent a provider request before it is issued.
    """
    if repetition < 1:
        raise ValueError("repetition must be >= 1")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{task.task_id}-r{repetition}-{uuid.uuid4().hex[:8]}"
    trial_root = output_root / run_id
    trial_root.mkdir()
    workspace = trial_root / "workspace"
    base_env = os.environ.copy()
    if task.environment:
        base_env.update(task.environment)
    for name, path in {
        "CODEFLOW_HOME": trial_root / "codeflow-home",
        "CODEFLOW_TRACING_DIR": trial_root / "traces",
    }.items():
        base_env[name] = str(path)
    base_env["CODEFLOW_TRACING"] = "1"
    Path(base_env["CODEFLOW_HOME"]).mkdir(parents=True)
    Path(base_env["CODEFLOW_TRACING_DIR"]).mkdir(parents=True)

    started = time.monotonic()
    _git(task.repository, "clone", "--shared", "--no-checkout", str(task.repository), str(workspace))
    _git(workspace, "checkout", "--detach", task.commit)
    resolved_commit = _git(workspace, "rev-parse", "HEAD").strip()
    if _git(workspace, "status", "--porcelain").strip():
        raise RuntimeError("fresh task workspace is unexpectedly dirty")

    initial_failures = _run_commands(task.fail_to_pass, workspace, trial_root, base_env, label="initial-fail-to-pass")
    if task.expected_initial_failures is not None and initial_failures["failed"] != task.expected_initial_failures:
        result = _failed_precondition(
            task, run_id, trial_root, resolved_commit, initial_failures, "initial_failure_count_mismatch"
        )
        _write_result(trial_root, result)
        return result
    initial_regression = _run_commands(task.pass_to_pass, workspace, trial_root, base_env, label="initial-pass-to-pass")
    if initial_regression["failed"]:
        result = _failed_precondition(
            task,
            run_id,
            trial_root,
            resolved_commit,
            initial_failures,
            "baseline_pass_to_pass_failed",
            initial_regression,
        )
        _write_result(trial_root, result)
        return result
    protected_before = _protected_digests(task.protected_paths, workspace)
    if any(digest == "MISSING" for digest in protected_before.values()):
        result = _failed_precondition(
            task,
            run_id,
            trial_root,
            resolved_commit,
            initial_failures,
            "protected_verifier_file_missing",
            initial_regression,
        )
        _write_result(trial_root, result)
        return result

    command = _format_command(task.agent_command, task=task.prompt, workspace=str(workspace), run_dir=str(trial_root))
    agent_result = _run_process(
        command,
        workspace,
        base_env,
        task.timeout_seconds,
        trial_root / "agent.stdout.log",
        trial_root / "agent.stderr.log",
    )

    final_failures = _run_commands(task.fail_to_pass, workspace, trial_root, base_env, label="final-fail-to-pass")
    final_regression = _run_commands(task.pass_to_pass, workspace, trial_root, base_env, label="final-pass-to-pass")
    trace_path = Path(base_env["CODEFLOW_TRACING_DIR"]) / "logs" / "audit-spans.log"
    trajectory = score_trace(trace_path) if trace_path.is_file() else score_trace([])
    final_commit = _git(workspace, "rev-parse", "HEAD").strip()
    diff_stat = _git(workspace, "diff", "--stat").strip()
    changed_paths = _git(workspace, "status", "--porcelain").splitlines()
    protected_after = _protected_digests(task.protected_paths, workspace)
    tampered_verifiers = sorted(
        path for path, digest in protected_before.items() if protected_after.get(path) != digest
    )
    elapsed_ms = round((time.monotonic() - started) * 1_000, 2)
    success = (
        agent_result["return_code"] == 0
        and not agent_result["timed_out"]
        and final_failures["total"] > 0
        and final_failures["failed"] == 0
        and final_regression["failed"] == 0
        and not tampered_verifiers
    )
    used_tokens = trajectory["context"]["total_tokens"]
    expected_tests_passed = final_failures["failed"] == 0 and final_regression["failed"] == 0
    trajectory["dimensions"]["execution_completeness"]["task_verifier"] = {
        "evidence_state": "external_test_commands_ran",
        "passed": expected_tests_passed,
        "fail_to_pass_commands": final_failures["total"],
        "pass_to_pass_commands": final_regression["total"],
    }
    result = {
        "schema": "codeflow.coding-run.v1",
        "run_id": run_id,
        "task_id": task.task_id,
        "task_digest": canonical_digest(_task_identity(task)),
        "repetition": repetition,
        "repository_commit": resolved_commit,
        "final_commit": final_commit,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "uv_lock_sha256": _file_sha256(workspace / "uv.lock"),
        },
        "status": "passed" if success else "invalid_verifier" if tampered_verifiers else "failed",
        "agent": agent_result,
        "verification": {
            "initial_fail_to_pass": initial_failures,
            "final_fail_to_pass": final_failures,
            "initial_pass_to_pass": initial_regression,
            "final_pass_to_pass": final_regression,
            "regression_detected": bool(final_regression["failed"]),
            "tampered_verifier_paths": tampered_verifiers,
            "all_expected_tests_passed": expected_tests_passed,
        },
        "metrics": {
            "end_to_end_latency_ms": elapsed_ms,
            "tokens": used_tokens,
            "token_usage_complete": trajectory["evidence"]["usage_complete"],
            "token_budget": task.token_budget,
            "token_budget_exceeded": (used_tokens > task.token_budget)
            if used_tokens is not None and task.token_budget
            else None,
            "changed_paths": changed_paths,
            "diff_stat": diff_stat,
        },
        "trajectory": trajectory,
        "artifact_dir": str(trial_root),
    }
    _write_result(trial_root, result)
    return result


def _run_commands(
    commands: tuple[tuple[str, ...], ...], workspace: Path, run_dir: Path, env: dict[str, str], *, label: str
) -> dict[str, Any]:
    results = []
    for index, argv in enumerate(commands):
        cmd = _format_command(argv, task="", workspace=str(workspace), run_dir=str(run_dir))
        result = _run_process(
            cmd, workspace, env, 180, run_dir / f"{label}-{index}.stdout.log", run_dir / f"{label}-{index}.stderr.log"
        )
        results.append({"argv": cmd, **result})
    return {
        "total": len(results),
        "passed": sum(r["return_code"] == 0 and not r["timed_out"] for r in results),
        "failed": sum(r["return_code"] != 0 or r["timed_out"] for r in results),
        "commands": results,
    }


def _run_process(
    argv: list[str], cwd: Path, env: dict[str, str], timeout: int, stdout_path: Path, stderr_path: Path
) -> dict[str, Any]:
    start = time.monotonic()
    timed_out = False
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=timeout, check=False
            )
        return_code = completed.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        return_code = None
    return {
        "argv": argv,
        "return_code": return_code,
        "timed_out": timed_out,
        "latency_ms": round((time.monotonic() - start) * 1_000, 2),
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
    }


def _format_command(argv: tuple[str, ...], **values: str) -> list[str]:
    formatted = []
    for item in argv:
        for key, value in values.items():
            item = item.replace("{" + key + "}", value)
        formatted.append(item)
    return formatted


def _protected_digests(paths: tuple[str, ...], workspace: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for relative in paths:
        path = (workspace / relative).resolve()
        if not path.is_relative_to(workspace.resolve()):
            raise ValueError(f"protected path escapes task workspace: {relative}")
        if not path.is_file():
            digests[relative] = "MISSING"
        else:
            digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _failed_precondition(
    task: CodingTask,
    run_id: str,
    root: Path,
    commit: str,
    f2p: dict[str, Any],
    reason: str,
    p2p: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "schema": "codeflow.coding-run.v1",
        "run_id": run_id,
        "task_id": task.task_id,
        "task_digest": canonical_digest(_task_identity(task)),
        "repository_commit": commit,
        "status": "invalid_baseline",
        "findings": [reason],
        "verification": {"initial_fail_to_pass": f2p, "initial_pass_to_pass": p2p},
        "artifact_dir": str(root),
    }
    return result


def _write_result(root: Path, result: dict[str, Any]) -> None:
    path = root / "result.json"
    temp = root / "result.json.tmp"
    temp.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _task_identity(task: CodingTask) -> dict[str, Any]:
    value = asdict(task)
    value["repository"] = "<source-repository>"
    value.pop("agent_command", None)
    env_text = json.dumps(task.environment or {}, sort_keys=True, separators=(",", ":"))
    value.pop("environment", None)
    value["environment_sha256"] = hashlib.sha256(env_text.encode("utf-8")).hexdigest()
    return value


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, encoding="utf-8", check=False
    )
    if completed.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout


def _required(raw: dict[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _argv(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{name} must be a non-empty argv array")
    return tuple(value)


def _commands(value: Any, name: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array of argv arrays")
    return tuple(_argv(row, f"{name}[{index}]") for index, row in enumerate(value))


def _paths(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(path, str) and path.strip() for path in value):
        raise ValueError("protected_paths must be a non-empty array of repository-relative paths")
    paths = tuple(value)
    if len(paths) != len(set(paths)) or any(Path(path).is_absolute() or ".." in Path(path).parts for path in paths):
        raise ValueError("protected_paths must be unique relative paths without parent traversal")
    return paths


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_positive_int(value: Any, name: str) -> int | None:
    return None if value is None else _positive_int(value, name)


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _environment(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError("environment must be a string-to-string object")
    return value or None
