from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from codeflow.providers.base import LLMProvider

from .artifacts import ArtifactError
from .budget import BudgetGuardedProvider
from .campaign import (
    CampaignError,
    CampaignMode,
    CampaignOutcome,
    DeterministicGateResult,
    ResolvedProvider,
    _read_runtime_evidence,
    default_campaign_services,
    estimate_worst_case_cost,
    load_campaign_suite,
    run_campaign,
)
from .canonical import to_primitive
from .registry import PackRegistry

DEFAULT_SCORECARD_SUITE_PATH = Path(__file__).resolve().parent / "suites" / "agent_application_scorecard_v1.yaml"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_SUBJECT_PATHS = (
    "codeflow",
    "pyproject.toml",
    "uv.lock",
    "benchmarks/codeflowbench/packs/runtime",
)


def build_scorecard_registry(
    mode: CampaignMode,
    resolved: ResolvedProvider,
) -> PackRegistry:
    from .packs.context import (
        ContextPack,
        ContextTrack,
        RuntimeContextTrialRunner,
    )
    from .packs.tool_mcp import (
        MCPRuntimeTrialRunner,
        ToolMCPPack,
        ToolMCPTrack,
    )

    if resolved.config is None or resolved.codeflow_config is None:
        raise CampaignError(
            "resolved Provider is missing Runtime configuration",
        )
    provider = resolved.provider
    if not isinstance(provider, LLMProvider):
        raise CampaignError("resolved Provider is not an LLMProvider")
    if (
        resolved.budget_ledger is None
        or not isinstance(provider, BudgetGuardedProvider)
        or provider.ledger is not resolved.budget_ledger
    ):
        raise CampaignError(
            "Scorecard runners must share the campaign Provider ledger",
        )
    calibration = mode is CampaignMode.CALIBRATION
    registry = PackRegistry()
    registry.register(
        ContextPack(
            ContextTrack.CALIBRATION if calibration else ContextTrack.FORMAL,
            runner=RuntimeContextTrialRunner(
                config=resolved.config,
                codeflow_config=resolved.codeflow_config,
                provider=provider,
            ),
        )
    )
    registry.register(
        ToolMCPPack(
            ToolMCPTrack.CALIBRATION if calibration else ToolMCPTrack.FORMAL,
            runner=MCPRuntimeTrialRunner(
                provider=provider,
                model=(resolved.configured_model or resolved.provider_name + "/" + resolved.model),
            ),
        )
    )
    return registry


def reused_runtime_gate(
    evidence_path: Path,
    *,
    codeflow_commit: str,
) -> DeterministicGateResult:
    path = Path(evidence_path).resolve()
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        evidence_digest = str(record["evidence_digest"])
        evidence = _read_runtime_evidence(
            path,
            expected_digest=evidence_digest,
        )
    except (
        ArtifactError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise CampaignError("Runtime evidence is unreadable") from exc
    evidence_commit = str(evidence.get("codeflow_commit", ""))
    if (
        not _runtime_subject_compatible(
            evidence_commit,
            codeflow_commit,
        )
        or evidence.get("worktree_clean") is not True
        or evidence.get("claim_eligible") is not True
    ):
        raise CampaignError(
            "Runtime evidence does not match the clean Scorecard subject",
        )
    return DeterministicGateResult(
        passed=True,
        details={
            "stage": "reused-runtime-evidence",
            "codeflow_commit": codeflow_commit,
        },
        evidence_path=str(path),
        evidence_digest=evidence_digest,
    )


def _runtime_subject_compatible(
    evidence_commit: str,
    scorecard_commit: str,
) -> bool:
    if evidence_commit == scorecard_commit:
        return True
    git = shutil.which("git")
    if git is None:
        return False
    completed = subprocess.run(
        (
            git,
            "diff",
            "--quiet",
            evidence_commit,
            scorecard_commit,
            "--",
            *_RUNTIME_SUBJECT_PATHS,
        ),
        cwd=_REPOSITORY_ROOT,
        check=False,
    )
    return completed.returncode == 0


async def run_scorecard_campaign(
    mode: CampaignMode,
    *,
    output_root: Path,
    runtime_evidence: Path | None = None,
    suite_path: Path = DEFAULT_SCORECARD_SUITE_PATH,
) -> CampaignOutcome:
    suite = load_campaign_suite(suite_path)
    defaults = default_campaign_services()
    codeflow_commit = defaults.resolve_codeflow_commit()

    async def deterministic(
        _output_root: Path,
    ) -> DeterministicGateResult:
        if runtime_evidence is None:
            return DeterministicGateResult(
                passed=True,
                details={
                    "stage": "scorecard-pack-preflight",
                    "codeflow_commit": codeflow_commit,
                },
            )
        return reused_runtime_gate(
            runtime_evidence,
            codeflow_commit=codeflow_commit,
        )

    services = replace(
        defaults,
        run_deterministic=deterministic,
        build_registry=build_scorecard_registry,
    )
    return await run_campaign(
        mode,
        output_root=output_root,
        suite=suite,
        services=services,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the current CodeFlowBench Scorecard campaign.",
    )
    parser.add_argument(
        "action",
        choices=("estimate", "ship"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".codeflow") / "evidence" / "codeflowbench-scorecard-v1",
    )
    parser.add_argument("--runtime-evidence", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "estimate":
            suite = load_campaign_suite(DEFAULT_SCORECARD_SUITE_PATH)
            estimate = estimate_worst_case_cost(
                suite,
                modes=(
                    CampaignMode.CALIBRATION,
                    CampaignMode.FORMAL,
                ),
            )
            print(
                json.dumps(
                    to_primitive(estimate),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        outcome = asyncio.run(
            run_scorecard_campaign(
                CampaignMode.SHIP,
                output_root=args.output_root,
                runtime_evidence=args.runtime_evidence,
            )
        )
    except CampaignError as exc:
        parser.exit(2, f"CodeFlowBench Scorecard aborted: {exc}\n")
    print(
        json.dumps(
            to_primitive(outcome),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
