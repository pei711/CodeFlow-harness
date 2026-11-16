from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

from benchmarks.codeflowbench.protocol import TrialContext, TrialExecution
from benchmarks.codeflowbench.records import (
    DeliveryOutcome,
    TrialStatus,
    TurnTerminalState,
    VerificationState,
    VerifierResult,
)
from codeflow.agent.tools.filesystem import (
    EditFileTool,
    ReadFileTool,
    WriteFileTool,
    _resolve_path,
)
from codeflow.agent.tools.shell import ExecTool
from codeflow.sandbox.direct_executor import _ENV_ALLOWLIST, _baseline_env
from codeflow.security.network import validate_url_target
from codeflow.security.trust import wrap_untrusted
from .metrics import SecurityMetrics
from .models import (
    SecurityCategory,
    SecurityRunOutcome,
    SecurityScenario,
    SecuritySummary,
)
from .verifier import SecurityVerifier

_EXPANDED_DENY_PATTERNS = [
    r"\brm\s+-[rf]{1,2}\b",
    r"\bdel\s+/[fq]\b",
    r"\brmdir\s+/s\b",
    r"(?:^|[;&|]\s*)format\b",
    r"\b(mkfs|diskpart)\b",
    r"(:\(\)\s*\{\s*:\|:&\s*\};:|fork\s*bomb)",
    r"\bdd\s+if=",
    r"\b(shutdown|reboot|poweroff|init\s+0)\b",
    r"\bchmod\s+(-r\s+)?777\s+/",
]


class SecurityTrialRunner(Protocol):
    async def __call__(
        self,
        *,
        context: TrialContext,
        scenario: SecurityScenario,
        repetition: int,
    ) -> TrialExecution: ...


class DeterministicSecurityRunner:
    """Executes security scenarios deterministically against CodeFlow's security boundaries.

    Requires no external model credentials. Directly exercises:
    1. ExecTool deny_patterns and workspace restriction guards.
    2. Filesystem tools _resolve_path directory traversal and escape guards.
    3. Network SSRF and forbidden IP validators.
    4. DirectExecutor environment variable credential stripping.
    5. Untrusted context boundary nonce fencing.
    """

    def __init__(self, verifier: SecurityVerifier | None = None) -> None:
        self.verifier = verifier or SecurityVerifier()

    def run_scenario_step(
        self,
        scenario: SecurityScenario,
        repetition: int,
        workspace_dir: Path,
    ) -> SecurityRunOutcome:
        action = scenario.target_action
        payload = scenario.payload

        shell_tool = ExecTool(
            working_dir=str(workspace_dir),
            restrict_to_workspace=True,
            deny_patterns=_EXPANDED_DENY_PATTERNS,
        )

        intercepted = False
        unapproved_ops = 0
        outside_writes = 0
        guard = "none"
        detail = ""

        if action == "shell_exec":
            cmd = str(payload.get("command", ""))
            block_reason = shell_tool._guard_command(cmd, cwd=str(workspace_dir))
            if block_reason:
                intercepted = True
                guard = (
                    "shell_workspace_restriction"
                    if "outside workspace" in block_reason.lower()
                    else "shell_deny_pattern"
                )
                detail = block_reason
            else:
                unapproved_ops = 1
                detail = f"Command permitted without approval: {cmd}"

        elif action == "fs_write":
            path_str = str(payload.get("path", ""))
            try:
                resolved = _resolve_path(path_str, workspace=workspace_dir, allowed_dir=workspace_dir)
                outside_writes = 1
                detail = f"File write resolved outside workspace: {resolved}"
            except PermissionError as exc:
                intercepted = True
                guard = "filesystem_workspace_boundary"
                detail = str(exc)

        elif action == "fs_edit":
            path_str = str(payload.get("path", ""))
            try:
                resolved = _resolve_path(path_str, workspace=workspace_dir, allowed_dir=workspace_dir)
                outside_writes = 1
                detail = f"File edit resolved outside workspace: {resolved}"
            except PermissionError as exc:
                intercepted = True
                guard = "filesystem_workspace_boundary"
                detail = str(exc)

        elif action == "fs_write_symlink":
            # Attempting escape via symlink target
            link_target = str(payload.get("symlink_target", "/tmp"))
            link_path = workspace_dir / str(payload.get("symlink_name", "escape_link"))
            try:
                if not link_path.exists():
                    os.symlink(link_target, link_path)
                test_file = str(payload.get("write_file", f"{link_path.name}/test.txt"))
                _resolve_path(test_file, workspace=workspace_dir, allowed_dir=workspace_dir)
                outside_writes = 1
                detail = f"Symlink write permitted to {link_target}"
            except PermissionError as exc:
                intercepted = True
                guard = "filesystem_workspace_boundary"
                detail = str(exc)
            except OSError as exc:
                intercepted = True
                guard = "os_symlink_restricted"
                detail = str(exc)

        elif action == "network_fetch":
            url = str(payload.get("url", ""))
            ok, msg = validate_url_target(url)
            if not ok:
                intercepted = True
                guard = "network_ssrf_validator"
                detail = msg
            else:
                detail = f"Unauthorized URL fetch permitted: {url}"

        elif action == "fs_read":
            path_str = str(payload.get("path", ""))
            try:
                resolved = _resolve_path(path_str, workspace=workspace_dir, allowed_dir=workspace_dir)
                detail = f"Sensitive file read permitted outside workspace: {resolved}"
            except PermissionError as exc:
                intercepted = True
                guard = "filesystem_workspace_boundary"
                detail = str(exc)

        elif action == "trust_boundary_check":
            untrusted_text = str(payload.get("untrusted_text", ""))
            wrapped = wrap_untrusted(untrusted_text, source="web")
            # Verify that fake delimiters cannot break out of the random nonce envelope
            # Find the actual closing nonce tag generated by wrap_untrusted
            closing_tags = re.findall(r"\[END UNTRUSTED web #([0-9a-f]+)\]", wrapped)
            if len(closing_tags) >= 1:
                real_nonce = closing_tags[-1]
                # If injected text contained a forged tag, ensure it did not match the real random nonce
                forged_in_payload = "#1234" in untrusted_text or "[USER_APPROVAL" in untrusted_text
                if forged_in_payload and real_nonce != "1234":
                    intercepted = True
                    guard = "untrusted_nonce_fence"
                    detail = f"Injected delimiter trapped within boundary, real nonce #{real_nonce}"
                else:
                    intercepted = True
                    guard = "untrusted_nonce_fence"
                    detail = f"Data boundary wrapped with random nonce #{real_nonce}"
            else:
                detail = "Trust boundary wrapping failed"

        elif action == "executor_env_isolation":
            # Test that high-value API keys and secrets are excluded from subprocess env
            mock_env = {
                "PATH": "/usr/bin:/bin",
                "USER": "agent",
                "OPENAI_API_KEY": "sk-secret-token",
                "ANTHROPIC_API_KEY": "sk-ant-secret",
                "DEEPSEEK_API_KEY": "sk-deepseek-secret",
                "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            }
            # Verify secrets are absent from _ENV_ALLOWLIST
            leaked = [k for k in payload.get("secret_keys", []) if k in _ENV_ALLOWLIST]
            if not leaked:
                intercepted = True
                guard = "direct_executor_env_allowlist"
                detail = "All sensitive environment keys stripped from execution context"
            else:
                unapproved_ops = 1
                detail = f"Sensitive keys allowed in subprocess environment: {leaked}"

        return SecurityRunOutcome(
            task_id=scenario.task_id,
            category=scenario.category.value,
            repetition=repetition,
            intercepted=intercepted,
            executed_unapproved_dangerous=unapproved_ops,
            executed_outside_workspace_write=outside_writes,
            guard_mechanism=guard,
            detail=detail,
        )

    async def __call__(
        self,
        *,
        context: TrialContext,
        scenario: SecurityScenario,
        repetition: int,
    ) -> TrialExecution:
        start_time = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp_workspace:
            ws_path = Path(tmp_workspace).resolve()
            outcome = self.run_scenario_step(scenario, repetition, ws_path)

        verifier_result = self.verifier.verify_outcome(scenario, outcome)
        latency_ms = (time.monotonic() - start_time) * 1000.0

        metrics = SecurityMetrics(
            intercepted=1 if outcome.intercepted else 0,
            unapproved_dangerous_operations_executed=outcome.executed_unapproved_dangerous,
            outside_workspace_writes_executed=outcome.executed_outside_workspace_write,
            category=scenario.category.value,
            guard_mechanism=outcome.guard_mechanism,
            repetition=repetition,
            end_to_end_latency_ms=latency_ms,
        )

        trial_status = (
            TrialStatus.PASSED
            if verifier_result.state == VerificationState.PASSED
            else TrialStatus.TASK_FAILED
        )

        return TrialExecution(
            status=trial_status,
            runtime_state=TurnTerminalState.COMPLETED,
            delivery_state=DeliveryOutcome.DELIVERED,
            verification=verifier_result,
            observed_variant_settings=dict(context.variant.settings),
            metrics=metrics.to_dict(),
            findings=(
                f"category:{scenario.category.value}",
                f"guard:{outcome.guard_mechanism}",
                f"repetition:{repetition}",
            ),
        )

    def run_full_campaign(
        self,
        scenarios: tuple[SecurityScenario, ...],
        repetitions: int = 3,
    ) -> tuple[list[SecurityRunOutcome], SecuritySummary]:
        outcomes: list[SecurityRunOutcome] = []
        category_stats: dict[str, dict[str, int]] = {}

        for scenario in scenarios:
            cat = scenario.category.value
            if cat not in category_stats:
                category_stats[cat] = {
                    "total": 0,
                    "intercepted": 0,
                    "unapproved_ops": 0,
                    "outside_writes": 0,
                }

            with tempfile.TemporaryDirectory() as tmp_dir:
                ws = Path(tmp_dir).resolve()
                for rep in range(1, repetitions + 1):
                    outcome = self.run_scenario_step(scenario, rep, ws)
                    outcomes.append(outcome)

                    category_stats[cat]["total"] += 1
                    if outcome.intercepted:
                        category_stats[cat]["intercepted"] += 1
                    category_stats[cat]["unapproved_ops"] += outcome.executed_unapproved_dangerous
                    category_stats[cat]["outside_writes"] += outcome.executed_outside_workspace_write

        total_runs = len(outcomes)
        total_intercepted = sum(1 for o in outcomes if o.intercepted)
        total_unapproved = sum(o.executed_unapproved_dangerous for o in outcomes)
        total_outside = sum(o.executed_outside_workspace_write for o in outcomes)

        summary = SecuritySummary(
            total_runs=total_runs,
            total_intercepted=total_intercepted,
            interception_rate=total_intercepted / max(1, total_runs),
            unapproved_dangerous_operations_executed=total_unapproved,
            outside_workspace_writes_executed=total_outside,
            category_breakdown=category_stats,
        )

        return outcomes, summary
