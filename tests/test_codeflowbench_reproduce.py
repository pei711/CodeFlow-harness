from __future__ import annotations

import json
import subprocess
from io import StringIO
from pathlib import Path

from rich.console import Console

from benchmarks.codeflowbench.canonical import canonical_digest
from benchmarks.codeflowbench.reproduce import (
    ReproductionConfig,
    ReproductionReport,
    StageResult,
    render_report,
    run_reproduction,
)
from benchmarks.codeflowbench.scorecard import DimensionScore, ScorecardResult
from codeflow.utils.portable_lock import file_lock


def test_reproduction_report_renders_score_dimensions_and_artifact_path(tmp_path: Path) -> None:
    report = ReproductionReport(
        codeflow_commit="a" * 40,
        output_root=tmp_path,
        stages=(
            StageResult("runtime", "reused", tmp_path / "runtime.json"),
            StageResult("tokenwise", "completed", tmp_path / "tokenwise.json"),
            StageResult("context_tool", "completed", tmp_path / "formal.json"),
        ),
        score=ScorecardResult(
            schema="codeflow.codeflowbench.multidimensional-score.v1",
            diagnostic_score=89.62,
            certified_score=None,
            dimensions={
                "capability": DimensionScore(44.62, 50.0),
                "reliability": DimensionScore(20.0, 20.0),
                "efficiency": DimensionScore(15.0, 20.0),
                "process": DimensionScore(10.0, 10.0),
            },
            certification_gates={"safety_evidence_complete": False},
            evidence={"runtime_current": True},
        ),
    )
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=240)

    render_report(report, console=console)

    output = stream.getvalue()
    assert "89.62" in output
    assert "diagnostic" in output
    assert "not certified" in output
    assert "Capability" in output
    assert "44.62 / 50.00" in output
    assert "Runtime" in output
    assert "reused" in output
    assert "Memory safety checks passed" in output
    assert "no" in output
    assert str(tmp_path) in output


def test_reproduction_refuses_paid_stages_before_running_any_command(tmp_path: Path) -> None:
    commands: list[tuple[str, ...]] = []

    def execute(command: tuple[str, ...]) -> str:
        commands.append(command)
        return ""

    report = run_reproduction(
        ReproductionConfig(
            output_root=tmp_path,
            execute_paid_campaign=False,
        ),
        execute_command=execute,
    )

    assert report.status == "failed"
    assert report.stages == (
        StageResult(
            "preflight",
            "failed",
            error="--execute-paid-campaign is required before paid stages",
        ),
    )
    assert commands == []
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "REPORT.md").is_file()


def test_reproduction_reuses_current_evidence_without_running_commands(tmp_path: Path) -> None:
    codeflow_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    inputs = tmp_path / "source"
    inputs.mkdir()
    formal_summary = inputs / "formal-summary.json"
    _write_formal_summary(formal_summary, codeflow_commit=codeflow_commit)
    runtime_evidence = inputs / "runtime-evidence.json"
    runtime_payload = {
        "schema": "codeflow.codeflowbench.runtime-scheduler-experiments.v1",
        "source_commit": codeflow_commit,
        "claim_eligible": True,
    }
    runtime_evidence.write_text(
        json.dumps({**runtime_payload, "evidence_digest": canonical_digest(runtime_payload)}),
        encoding="utf-8",
    )
    tokenwise_report = inputs / "tokenwise-report.json"
    tokenwise_payload = {
        "schema": "codeflow.codeflowbench.tokenwise-cost.report.v1",
        "campaign": {"codeflow_commit": codeflow_commit},
        "claim": {"claim_eligible": True},
    }
    tokenwise_report.write_text(
        json.dumps({**tokenwise_payload, "report_digest": canonical_digest(tokenwise_payload)}),
        encoding="utf-8",
    )
    commands: list[tuple[str, ...]] = []

    report = run_reproduction(
        ReproductionConfig(
            output_root=tmp_path / "result",
            formal_summary=formal_summary,
            runtime_evidence=runtime_evidence,
            tokenwise_report=tokenwise_report,
        ),
        execute_command=lambda command: commands.append(command) or "",
        validate_environment=lambda _config: None,
    )

    assert report.status == "completed", report.error
    assert report.score is not None
    assert report.score.diagnostic_score == 83.3333
    assert report.score.certified_score is None
    assert commands == []
    assert [stage.status for stage in report.stages] == [
        "completed",
        "reused",
        "reused",
        "reused",
        "completed",
    ]
    assert (tmp_path / "result" / "score.json").is_file()
    manifest = json.loads((tmp_path / "result" / "manifest.json").read_text(encoding="utf-8"))
    manifest_digest = manifest.pop("manifest_digest")
    assert canonical_digest(manifest) == manifest_digest
    machine_report = json.loads((tmp_path / "result" / "report.json").read_text(encoding="utf-8"))
    assert all(stage["evidence_sha256"] for stage in machine_report["stages"] if stage["evidence_path"] is not None)
    markdown = (tmp_path / "result" / "REPORT.md").read_text(encoding="utf-8")
    assert "## Dimensions" in markdown
    assert "## Certification checks" in markdown
    assert "All planned trials finished" in markdown

    _write_formal_summary(formal_summary, codeflow_commit="b" * 40)
    rejected = run_reproduction(
        ReproductionConfig(
            output_root=tmp_path / "rejected",
            formal_summary=formal_summary,
            runtime_evidence=runtime_evidence,
            tokenwise_report=tokenwise_report,
        ),
        execute_command=lambda command: commands.append(command) or "",
        validate_environment=lambda _config: None,
    )

    assert rejected.status == "failed"
    assert "current Scorecard subject" in (rejected.error or "")
    assert commands == []


def test_reproduction_runs_all_missing_stages_and_composes_score(tmp_path: Path) -> None:
    codeflow_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commands: list[tuple[str, ...]] = []

    def execute(command: tuple[str, ...]) -> str:
        commands.append(command)
        output_root = Path(command[command.index("--output-root") + 1])
        joined = " ".join(command)
        if "scheduler_experiments" in joined:
            payload = {
                "schema": "codeflow.codeflowbench.runtime-scheduler-experiments.v1",
                "source_commit": codeflow_commit,
                "claim_eligible": True,
            }
            evidence = {**payload, "evidence_digest": canonical_digest(payload)}
            path = output_root / codeflow_commit / "plan" / "runs" / "scheduler-experiments.test.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(evidence), encoding="utf-8")
        elif "tokenwise_cost_campaign" in joined:
            output_root.mkdir(parents=True, exist_ok=True)
            if command[command.index("--mode") + 1] == "preflight":
                (output_root / "preflight.json").write_text(
                    json.dumps({"passed": True}),
                    encoding="utf-8",
                )
            else:
                payload = {
                    "schema": "codeflow.codeflowbench.tokenwise-cost.report.v1",
                    "campaign": {"codeflow_commit": codeflow_commit},
                    "claim": {"claim_eligible": True},
                }
                (output_root / "report.json").write_text(
                    json.dumps({**payload, "report_digest": canonical_digest(payload)}),
                    encoding="utf-8",
                )
        else:
            _write_campaign_artifacts(
                output_root,
                codeflow_commit=codeflow_commit,
                summary=_formal_summary(),
            )
        return "completed"

    report = run_reproduction(
        ReproductionConfig(
            output_root=tmp_path / "result",
            execute_paid_campaign=True,
        ),
        execute_command=execute,
        validate_environment=lambda _config: None,
    )

    assert report.status == "completed", report.error
    assert report.score is not None
    assert report.score.diagnostic_score == 83.3333
    assert len(commands) == 4
    assert [stage.name for stage in report.stages] == [
        "preflight",
        "runtime",
        "tokenwise",
        "context_tool",
        "score",
    ]
    assert all(stage.status == "completed" for stage in report.stages)

    commands.clear()
    resumed = run_reproduction(
        ReproductionConfig(
            output_root=tmp_path / "result",
            execute_paid_campaign=True,
        ),
        execute_command=execute,
        validate_environment=lambda _config: None,
    )

    assert resumed.status == "completed", resumed.error
    assert commands == []
    assert [stage.status for stage in resumed.stages] == [
        "completed",
        "resumed",
        "resumed",
        "resumed",
        "completed",
    ]


def test_makefile_exposes_single_reproduction_command() -> None:
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")

    assert "codeflowbench-reproduce:" in makefile
    assert "python -m benchmarks.codeflowbench.reproduce" in makefile


def test_reproduction_refuses_a_second_writer(tmp_path: Path) -> None:
    commands: list[tuple[str, ...]] = []

    with file_lock(tmp_path / ".reproduction.lock", blocking=False):
        report = run_reproduction(
            ReproductionConfig(
                output_root=tmp_path,
                execute_paid_campaign=True,
            ),
            execute_command=lambda command: commands.append(command) or "",
            validate_environment=lambda _config: None,
        )

    assert report.status == "failed"
    assert "active writer" in (report.error or "")
    assert commands == []


def _formal_summary() -> dict[str, object]:
    return {
        "ship_complete": True,
        "measurement_valid": True,
        "metrics": {
            "context.expected_pair_count": 24,
            "context.treatment_pass_count": 24,
            "context.capability_evidence_complete": True,
            "context.treatment_capability_score_rate": 1.0,
            "context.positive_claim_eligible": True,
            "tool_mcp.expected_pair_count": 24,
            "tool_mcp.treatment_pass_count": 24,
            "tool_mcp.positive_claim_eligible": True,
            "tool_mcp.all_initial_visible_sets_differ": True,
            "tool_mcp.all_mcp_arms_connected": True,
            "tool_mcp.invalid_target_call_rate_noninferior": True,
            "tool_mcp.exact_target_repeat_rate_noninferior": True,
        },
    }


def _write_campaign_artifacts(
    output_root: Path,
    *,
    codeflow_commit: str,
    summary: dict[str, object],
) -> None:
    experiment_root = output_root / "scorecard-formal"
    experiment_root.mkdir(parents=True)
    summary_path = experiment_root / "summary.json"
    _write_formal_summary(
        summary_path,
        codeflow_commit=codeflow_commit,
        summary=summary,
    )
    outcome_root = output_root / "campaigns" / "suite" / codeflow_commit / "ship"
    outcome_root.mkdir(parents=True)
    outcome = {
        "mode": "ship",
        "experiments": [{"experiment_id": experiment_root.name, "root": str(experiment_root)}],
        "reports": [{"experiment_id": experiment_root.name}],
    }
    (outcome_root / "campaign-outcome.json").write_text(
        json.dumps(outcome),
        encoding="utf-8",
    )


def _write_formal_summary(
    path: Path,
    *,
    codeflow_commit: str,
    summary: dict[str, object] | None = None,
) -> None:
    spec = {
        "schema": "codeflow.codeflowbench.experiment.v1",
        "evidence_schema": "codeflow.codeflowbench.evidence.v1",
        "identity": {
            "campaign_mode": "formal",
            "campaign_suite": "agent-application-scorecard-v1",
            "codeflow_commit": codeflow_commit,
        },
        "pack_ids": ["context", "tool-mcp"],
        "repetitions": 1,
    }
    pack_definitions: list[object] = []
    experiment_id = canonical_digest(
        {"spec": spec, "pack_definitions": pack_definitions},
    )
    manifest = {
        "schema": spec["schema"],
        "evidence_schema": spec["evidence_schema"],
        "experiment_id": experiment_id,
        "plan_digest": experiment_id,
        "spec": spec,
        "pack_definitions": pack_definitions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name("manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    payload = {**(summary or _formal_summary()), "experiment_id": experiment_id}
    path.write_text(
        json.dumps({**payload, "report_digest": canonical_digest(payload)}),
        encoding="utf-8",
    )
