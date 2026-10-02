#!/usr/bin/env python3
"""V-C0 and V-S0: deterministic Channel evidence Gates.

Runs the Channel contract bundle and the Channel security and isolation bundle
as separate pytest invocations and writes one machine-readable report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]

# V-C0：全部确定性渠道契约文件。新增确定性渠道测试文件时必须在此登记，否则门会静默失去覆盖。
_CONTRACT_TESTS = (
    "tests/test_auth_allowlist.py",
    "tests/test_channels_base.py",
    "tests/test_channels_contract.py",
    "tests/test_channels_errors.py",
    "tests/test_channels_feishu.py",
    "tests/test_channels_intake.py",
    "tests/test_channels_manager.py",
    "tests/test_channels_media.py",
    "tests/test_channels_outlet.py",
    "tests/test_channels_qq.py",
    "tests/test_channels_registry.py",
    "tests/test_channels_required_marker.py",
    "tests/test_channels_wecom.py",
    "tests/test_cli_channel_commands.py",
    "tests/test_cli_gateway_spine.py",
    "tests/test_config_update_channels.py",
    "tests/test_cron_delivery.py",
    "tests/test_spine_delivery.py",
    "tests/test_verify_channels.py",
)

# V-S0：确定性渠道安全与隔离测试包。三个声明分别由具名测试固定，而非整文件扫描，使重命名或
# 删除会让门失败，而不是悄悄缩小范围。
_SDK_LAZINESS_TESTS = (
    "tests/test_channels_feishu.py::test_feishu_spec_import_is_cheap",
    "tests/test_channels_qq.py::test_qq_spec_import_is_cheap",
    "tests/test_channels_registry.py::test_discover_specs_is_cheap",
    "tests/test_channels_wecom.py::test_wecom_spec_import_is_cheap",
)
_ALLOWLIST_TESTS = (
    "tests/test_auth_allowlist.py",
    "tests/test_channels_feishu.py::test_on_message_disallowed_sender_skips_react_and_download",
    "tests/test_channels_qq.py::test_on_message_disallowed_sender_rejected_before_side_effects",
    "tests/test_channels_wecom.py::test_process_disallowed_sender_skips_download_and_publish",
)
_ISOLATION_TESTS = (
    "tests/test_channels_manager.py::test_empty_allow_from_disables_the_channel_loudly",
    "tests/test_channels_manager.py::test_empty_allow_from_leaves_other_channels_running",
    "tests/test_channels_manager.py::test_factory_crash_disables_only_that_channel",
    "tests/test_channels_manager.py::test_gateway_still_constructs_when_every_channel_fails",
    "tests/test_channels_manager.py::test_init_disables_channel_on_missing_dependency",
    "tests/test_channels_qq.py::test_qq_spec_factory_raises_import_error_without_sdk",
    "tests/test_channels_wecom.py::test_wecom_spec_factory_raises_import_error_without_sdk",
)
_SECURITY_TESTS = _SDK_LAZINESS_TESTS + _ALLOWLIST_TESTS + _ISOLATION_TESTS

_CHECKS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("contract", "V-C0", _CONTRACT_TESTS),
    ("security", "V-S0", _SECURITY_TESTS),
)

_COUNT_PATTERNS = {
    "passed": re.compile(r"(\d+) passed"),
    "failed": re.compile(r"(\d+) failed"),
    "errors": re.compile(r"(\d+) errors?"),
    "skipped": re.compile(r"(\d+) skipped"),
    "xfailed": re.compile(r"(\d+) xfailed"),
    "xpassed": re.compile(r"(\d+) xpassed"),
    "deselected": re.compile(r"(\d+) deselected"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(output: str) -> dict[str, int]:
    return {
        name: int(match.group(1)) if (match := pattern.search(output)) else 0
        for name, pattern in _COUNT_PATTERNS.items()
    }


def _classify(returncode: int, output: str) -> str:
    """Both bundles are deterministic, so anything short of a clean pass fails.

    A skip or an expected failure is not evidence that a Channel contract holds;
    only a run where every selected test actually passed can pass the Gate.
    """
    counts = _counts(output)
    if returncode == 0:
        clean = counts["passed"] > 0 and not any(
            counts[name] for name in ("failed", "errors", "skipped", "xfailed", "xpassed")
        )
        return "passed" if clean else "failed"
    if "infrastructure_failure" in output.lower():
        return "infrastructure_failure"
    return "failed"


def _run_check(name: str, tests: tuple[str, ...], output_root: Path) -> dict[str, Any]:
    log_path = output_root / f"{name}.log"
    command = [sys.executable, "-m", "pytest", "-q", "--strict-markers", *tests]
    try:
        completed = subprocess.run(
            command,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        output = stdout + stderr
        log_path.write_text(output, encoding="utf-8")
        return {
            "command": command,
            "counts": _counts(output),
            "exit_code": None,
            "log": log_path.name,
            "log_sha256": _sha256(log_path),
            "selection": list(tests),
            "status": "infrastructure_failure",
        }
    output = completed.stdout + completed.stderr
    log_path.write_text(output, encoding="utf-8")
    return {
        "command": command,
        "counts": _counts(output),
        "exit_code": completed.returncode,
        "log": log_path.name,
        "log_sha256": _sha256(log_path),
        "selection": list(tests),
        "status": _classify(completed.returncode, output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    checks: dict[str, dict[str, Any]] = {}
    for name, _gate, tests in _CHECKS:
        checks[name] = _run_check(name, tests, output_root)

    status = "passed" if all(check["status"] == "passed" for check in checks.values()) else "failed"
    # v2：v1 只有一个 "deterministic" 检查；v2 将其重命名为 "contract" 并增加 "security"，
    # 按模式版本规则属于破坏性结构变更。
    report = {
        "checks": checks,
        "gate": "V-C0",
        "schema": "codeflow.channels.evidence.v2",
        "security_gate": "V-S0",
        "status": status,
    }
    report_path = output_root / "channels-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for name, gate, _tests in _CHECKS:
        print(f"{gate} {checks[name]['status']}: {report_path}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
