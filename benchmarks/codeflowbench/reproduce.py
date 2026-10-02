from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from codeflow.utils.portable_lock import LockTimeoutError, file_lock

from .canonical import canonical_digest
from .scorecard import (
    ScorecardResult,
    _formal_inputs,
    _read_json,
    _runtime_inputs,
    _tokenwise_inputs,
    compute_scorecard,
)


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    evidence_path: Path | None = None
    error: str | None = None


@dataclass(frozen=True)
class ReproductionConfig:
    output_root: Path
    execute_paid_campaign: bool = False
    formal_summary: Path | None = None
    runtime_evidence: Path | None = None
    tokenwise_report: Path | None = None
    scoring_spec_preregistered: bool = True


@dataclass(frozen=True)
class ReproductionReport:
    codeflow_commit: str
    output_root: Path
    stages: tuple[StageResult, ...]
    score: ScorecardResult | None
    status: str = "completed"
    error: str | None = None


_STAGE_LABELS = {
    "preflight": "Preflight",
    "runtime": "Runtime",
    "tokenwise": "TokenWise",
    "context_tool": "Context + Tool/MCP",
    "score": "Score",
}
_DIMENSION_LABELS = {
    "capability": "Capability",
    "reliability": "Reliability",
    "efficiency": "Efficiency",
    "process": "Process",
}
_GATE_LABELS = {
    "scoring_spec_preregistered": "Scoring rules frozen before the run",
    "ship_complete": "All planned trials finished",
    "measurement_valid": "The comparisons are valid",
    "evidence_complete": "Every dimension has current evidence",
    "safety_evidence_complete": "Memory safety checks passed",
}


CommandExecutor = Callable[[tuple[str, ...]], str]
EnvironmentValidator = Callable[[ReproductionConfig], None]


class ReproductionError(RuntimeError):
    pass


def run_reproduction(
    config: ReproductionConfig,
    *,
    execute_command: CommandExecutor,
    validate_environment: EnvironmentValidator | None = None,
) -> ReproductionReport:
    output_root = config.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    codeflow_commit = _current_codeflow_commit()
    try:
        with file_lock(output_root / ".reproduction.lock", blocking=False):
            return _run_reproduction_locked(
                config,
                output_root=output_root,
                codeflow_commit=codeflow_commit,
                execute_command=execute_command,
                validate_environment=validate_environment,
            )
    except LockTimeoutError:
        return ReproductionReport(
            codeflow_commit=codeflow_commit,
            output_root=output_root,
            stages=(
                StageResult(
                    "preflight",
                    "failed",
                    error="reproduction output already has an active writer",
                ),
            ),
            score=None,
            status="failed",
            error="reproduction output already has an active writer",
        )


def _run_reproduction_locked(
    config: ReproductionConfig,
    *,
    output_root: Path,
    codeflow_commit: str,
    execute_command: CommandExecutor,
    validate_environment: EnvironmentValidator | None,
) -> ReproductionReport:
    paid_stages_required = any(
        value is None
        for value in (
            config.formal_summary,
            config.tokenwise_report,
        )
    )
    if paid_stages_required and not config.execute_paid_campaign:
        error = "--execute-paid-campaign is required before paid stages"
        report = ReproductionReport(
            codeflow_commit=codeflow_commit,
            output_root=output_root,
            stages=(StageResult("preflight", "failed", error=error),),
            score=None,
            status="failed",
            error=error,
        )
        _write_report(report)
        return report
    try:
        inputs_root = output_root / "inputs"
        formal_summary = _copy_formal_input(
            config.formal_summary,
            inputs_root,
        )
        runtime_evidence = _copy_input(
            config.runtime_evidence,
            inputs_root / "runtime-evidence.json",
        )
        tokenwise_report = _copy_input(
            config.tokenwise_report,
            inputs_root / "tokenwise-report.json",
        )
        _validate_reused_inputs(
            codeflow_commit=codeflow_commit,
            formal_summary=formal_summary,
            runtime_evidence=runtime_evidence,
            tokenwise_report=tokenwise_report,
        )
        (validate_environment or _validate_environment)(config)
        _freeze_manifest(
            output_root / "manifest.json",
            codeflow_commit=codeflow_commit,
            scoring_spec_preregistered=config.scoring_spec_preregistered,
            inputs={
                "context_tool": formal_summary,
                "context_tool_manifest": (
                    formal_summary.with_name("manifest.json") if formal_summary is not None else None
                ),
                "runtime": runtime_evidence,
                "tokenwise": tokenwise_report,
            },
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return _failed_report(
            codeflow_commit,
            output_root,
            f"preflight failed: {exc}",
        )
    stages: list[StageResult] = [StageResult("preflight", "completed")]
    active_stage = "runtime"
    try:
        runtime_evidence, runtime_status = _runtime_stage(
            runtime_evidence,
            root=output_root / "runtime",
            codeflow_commit=codeflow_commit,
            execute_command=execute_command,
            logs_root=output_root / "logs",
        )
        stages.append(StageResult("runtime", runtime_status, runtime_evidence))
        active_stage = "tokenwise"
        tokenwise_report, tokenwise_status = _tokenwise_stage(
            tokenwise_report,
            root=output_root / "tokenwise",
            codeflow_commit=codeflow_commit,
            execute_command=execute_command,
            logs_root=output_root / "logs",
        )
        stages.append(StageResult("tokenwise", tokenwise_status, tokenwise_report))
        active_stage = "context_tool"
        formal_summary, formal_status = _scorecard_stage(
            formal_summary,
            root=output_root / "context-tool",
            runtime_evidence=runtime_evidence,
            codeflow_commit=codeflow_commit,
            execute_command=execute_command,
            logs_root=output_root / "logs",
        )
        stages.append(StageResult("context_tool", formal_status, formal_summary))
        runtime_record = _read_json(runtime_evidence)
        runtime_current, runtime_claim = _runtime_inputs(runtime_record, codeflow_commit=codeflow_commit)
        tokenwise_current, tokenwise_claim = _tokenwise_inputs(
            _read_json(tokenwise_report),
            codeflow_commit=codeflow_commit,
        )
        score = compute_scorecard(
            _read_json(formal_summary),
            runtime_record,
            runtime_current_evidence=runtime_current,
            tokenwise_current_evidence=tokenwise_current,
            tokenwise_current_claim_eligible=tokenwise_claim,
            turn_efficiency_current_evidence=runtime_current,
            turn_efficiency_current_claim_eligible=runtime_claim,
            scoring_spec_preregistered=config.scoring_spec_preregistered,
        )
    except (OSError, ReproductionError, TypeError, ValueError) as exc:
        stages.append(StageResult(active_stage, "failed", error=str(exc)))
        report = ReproductionReport(
            codeflow_commit=codeflow_commit,
            output_root=output_root,
            stages=tuple(stages),
            score=None,
            status="failed",
            error=f"{active_stage} failed: {exc}",
        )
        _write_report(report)
        return report
    score_path = output_root / "score.json"
    score_path.write_text(
        json.dumps(_score_payload(score), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = ReproductionReport(
        codeflow_commit=codeflow_commit,
        output_root=output_root,
        stages=tuple([*stages, StageResult("score", "completed", score_path)]),
        score=score,
    )
    _write_report(report)
    return report


def _validate_reused_inputs(
    *,
    codeflow_commit: str,
    formal_summary: Path | None,
    runtime_evidence: Path | None,
    tokenwise_report: Path | None,
) -> None:
    if formal_summary is not None:
        _validate_formal_summary(formal_summary, codeflow_commit=codeflow_commit)
    if runtime_evidence is not None:
        _runtime_inputs(_read_json(runtime_evidence), codeflow_commit=codeflow_commit)
    if tokenwise_report is not None:
        _tokenwise_inputs(_read_json(tokenwise_report), codeflow_commit=codeflow_commit)


def _validate_environment(config: ReproductionConfig) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    git = shutil.which("git")
    if git is None:
        raise ValueError("git is required to validate the CodeFlowBench checkout")
    completed = subprocess.run(
        (git, "status", "--short"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise ValueError("CodeFlowBench reproduction requires a clean worktree")
    codeflow_commit = _current_codeflow_commit()
    retained_tokenwise = config.output_root.resolve() / "tokenwise" / "report.json"
    if config.tokenwise_report is None and retained_tokenwise.is_file():
        _tokenwise_inputs(_read_json(retained_tokenwise), codeflow_commit=codeflow_commit)
    tokenwise_pending = config.tokenwise_report is None and not retained_tokenwise.is_file()
    retained_scorecard = _campaign_artifacts(
        config.output_root.resolve() / "context-tool",
        codeflow_commit=codeflow_commit,
    )
    if config.formal_summary is None and isinstance(retained_scorecard, Path):
        _validate_formal_summary(retained_scorecard, codeflow_commit=codeflow_commit)
    scorecard_pending = config.formal_summary is None and retained_scorecard is None
    if tokenwise_pending or scorecard_pending:
        from .packs.tokenwise_cost.runner import load_deepseek_key

        load_deepseek_key()
    if scorecard_pending:
        from .campaign import (
            _validate_resolved_provider,
            default_campaign_services,
            load_campaign_suite,
        )

        resolved = default_campaign_services().resolve_provider()
        if scorecard_pending:
            from .scorecard_campaign import DEFAULT_SCORECARD_SUITE_PATH

            _validate_resolved_provider(
                load_campaign_suite(DEFAULT_SCORECARD_SUITE_PATH),
                resolved,
            )


def _runtime_stage(
    reused: Path | None,
    *,
    root: Path,
    codeflow_commit: str,
    execute_command: CommandExecutor,
    logs_root: Path,
) -> tuple[Path, str]:
    if reused is not None:
        return reused, "reused"
    existing = _find_runtime_evidence(root, codeflow_commit=codeflow_commit)
    if existing is not None:
        return existing, "resumed"
    _run_command(
        "runtime",
        (
            sys.executable,
            "-m",
            "benchmarks.codeflowbench.packs.runtime.scheduler_experiments",
            "--output-root",
            str(root),
        ),
        execute_command=execute_command,
        logs_root=logs_root,
    )
    evidence = _find_runtime_evidence(root, codeflow_commit=codeflow_commit)
    if evidence is None:
        raise ReproductionError("Runtime stage produced no current evidence")
    return evidence, "completed"


def _tokenwise_stage(
    reused: Path | None,
    *,
    root: Path,
    codeflow_commit: str,
    execute_command: CommandExecutor,
    logs_root: Path,
) -> tuple[Path, str]:
    if reused is not None:
        return reused, "reused"
    report = root / "report.json"
    if report.is_file():
        _tokenwise_inputs(_read_json(report), codeflow_commit=codeflow_commit)
        return report, "resumed"
    for mode in ("preflight", "formal"):
        _run_command(
            f"tokenwise-{mode}",
            (
                sys.executable,
                "-m",
                "benchmarks.codeflowbench.tokenwise_cost_campaign",
                "--mode",
                mode,
                "--output-root",
                str(root),
                "--execute-paid-campaign",
            ),
            execute_command=execute_command,
            logs_root=logs_root,
        )
    if not report.is_file():
        raise ReproductionError("TokenWise stage produced no report")
    _tokenwise_inputs(_read_json(report), codeflow_commit=codeflow_commit)
    return report, "completed"


def _scorecard_stage(
    reused: Path | None,
    *,
    root: Path,
    runtime_evidence: Path,
    codeflow_commit: str,
    execute_command: CommandExecutor,
    logs_root: Path,
) -> tuple[Path, str]:
    if reused is not None:
        _validate_formal_summary(reused, codeflow_commit=codeflow_commit)
        return reused, "reused"
    existing = _campaign_artifacts(root, codeflow_commit=codeflow_commit)
    if existing is not None:
        _validate_formal_summary(existing, codeflow_commit=codeflow_commit)
        return existing, "resumed"
    _run_command(
        "context-tool",
        (
            sys.executable,
            "-m",
            "benchmarks.codeflowbench.scorecard_campaign",
            "ship",
            "--output-root",
            str(root),
            "--runtime-evidence",
            str(runtime_evidence),
        ),
        execute_command=execute_command,
        logs_root=logs_root,
    )
    summary = _campaign_artifacts(root, codeflow_commit=codeflow_commit)
    if summary is None:
        raise ReproductionError("Context and Tool/MCP stage produced no formal summary")
    _validate_formal_summary(summary, codeflow_commit=codeflow_commit)
    return summary, "completed"


def _run_command(
    stage: str,
    command: tuple[str, ...],
    *,
    execute_command: CommandExecutor,
    logs_root: Path,
) -> None:
    output = execute_command(command)
    logs_root.mkdir(parents=True, exist_ok=True)
    (logs_root / f"{stage}.log").write_text(output, encoding="utf-8")


def _find_runtime_evidence(root: Path, *, codeflow_commit: str) -> Path | None:
    candidates = sorted(root.glob("**/scheduler-experiments.*.json"))
    valid = []
    for path in candidates:
        try:
            _runtime_inputs(_read_json(path), codeflow_commit=codeflow_commit)
        except (OSError, TypeError, ValueError):
            continue
        valid.append(path.resolve())
    if len(valid) > 1:
        raise ReproductionError("Runtime stage has multiple current evidence files")
    return valid[0] if valid else None


def _campaign_artifacts(
    root: Path,
    *,
    codeflow_commit: str,
) -> Path | None:
    outcomes = sorted(root.glob(f"campaigns/*/{codeflow_commit}/ship/campaign-outcome.json"))
    if not outcomes:
        return None
    if len(outcomes) > 1:
        raise ReproductionError("campaign stage has multiple outcome artifacts")
    outcome_path = outcomes[0].resolve()
    outcome = _read_json(outcome_path)
    experiments = outcome.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ReproductionError("campaign outcome has no experiments")
    formal = experiments[-1]
    if not isinstance(formal, dict) or not isinstance(formal.get("root"), str):
        raise ReproductionError("campaign outcome has no formal experiment root")
    experiment_root = Path(formal["root"])
    if not experiment_root.is_absolute():
        experiment_root = Path(__file__).resolve().parents[2] / experiment_root
    summary = (experiment_root / "summary.json").resolve()
    if not summary.is_file():
        raise ReproductionError("formal campaign summary is missing")
    return summary


def _copy_input(source: Path | None, destination: Path) -> Path | None:
    if source is None:
        return None
    source = source.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"input does not name a file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and source != destination:
        if _sha256_file(source) != _sha256_file(destination):
            raise ValueError(f"retained input differs from {source.name}")
        return destination
    if source != destination:
        shutil.copyfile(source, destination)
    return destination


def _copy_formal_input(source: Path | None, inputs_root: Path) -> Path | None:
    if source is None:
        return None
    source = source.expanduser().resolve()
    manifest = source.with_name("manifest.json")
    if not manifest.is_file():
        raise ValueError("formal summary must be accompanied by its manifest.json")
    copied_summary = _copy_input(source, inputs_root / "formal-summary.json")
    _copy_input(manifest, inputs_root / "manifest.json")
    return copied_summary


def _validate_formal_summary(path: Path, *, codeflow_commit: str) -> None:
    _formal_inputs(path, codeflow_commit=codeflow_commit)


def _freeze_manifest(
    path: Path,
    *,
    codeflow_commit: str,
    scoring_spec_preregistered: bool,
    inputs: dict[str, Path | None],
) -> None:
    payload: dict[str, Any] = {
        "schema": "codeflow.codeflowbench.reproduction-manifest.v1",
        "codeflow_commit": codeflow_commit,
        "scoring_spec_preregistered": scoring_spec_preregistered,
        "tracks": {
            name: (
                {
                    "source": "reused",
                    "file_name": source.name,
                    "sha256": _sha256_file(source),
                }
                if source is not None
                else {"source": "run"}
            )
            for name, source in sorted(inputs.items())
        },
    }
    record = {**payload, "manifest_digest": canonical_digest(payload)}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != record:
            raise ValueError("output root belongs to a different reproduction manifest")
        return
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failed_report(
    codeflow_commit: str,
    output_root: Path,
    error: str,
) -> ReproductionReport:
    report = ReproductionReport(
        codeflow_commit=codeflow_commit,
        output_root=output_root,
        stages=(StageResult("preflight", "failed", error=error),),
        score=None,
        status="failed",
        error=error,
    )
    _write_report(report)
    return report


def _current_codeflow_commit() -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to bind reproduction evidence")
    completed = subprocess.run(
        (git, "rev-parse", "--verify", "HEAD"),
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_report(report: ReproductionReport) -> None:
    payload: dict[str, Any] = {
        "schema": "codeflow.codeflowbench.reproduction-report.v1",
        "status": report.status,
        "codeflow_commit": report.codeflow_commit,
        "error": report.error,
        "stages": [
            {
                "name": stage.name,
                "status": stage.status,
                "evidence_path": (str(stage.evidence_path) if stage.evidence_path else None),
                "evidence_sha256": (
                    _sha256_file(stage.evidence_path)
                    if stage.evidence_path is not None and stage.evidence_path.is_file()
                    else None
                ),
                "error": stage.error,
            }
            for stage in report.stages
        ],
        "score": _score_payload(report.score),
    }
    payload["report_digest"] = canonical_digest(payload)
    report.output_root.mkdir(parents=True, exist_ok=True)
    (report.output_root / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = [
        "# CodeFlowBench reproduction report",
        "",
        f"- Status: {report.status}",
        f"- CodeFlow commit: `{report.codeflow_commit}`",
        f"- Diagnostic score: {_diagnostic_label(report.score)}",
        "",
        "| Stage | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    rows.extend(
        f"| {_STAGE_LABELS.get(stage.name, stage.name)} | {stage.status} | "
        f"{_markdown_cell(stage.evidence_path or stage.error or '-')} |"
        for stage in report.stages
    )
    if report.score is not None:
        rows.extend(
            [
                "",
                "## Dimensions",
                "",
                "| Dimension | Score |",
                "| --- | ---: |",
                *(
                    f"| {_DIMENSION_LABELS.get(name, name)} | {value.earned:.2f} / {value.maximum:.2f} |"
                    for name, value in report.score.dimensions.items()
                ),
                "",
                "## Certification checks",
                "",
                "| Check | Passed |",
                "| --- | --- |",
                *(
                    f"| {_GATE_LABELS.get(name, name)} | {'yes' if passed else 'no'} |"
                    for name, passed in report.score.certification_gates.items()
                ),
            ]
        )
    (report.output_root / "REPORT.md").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def _score_payload(score: ScorecardResult | None) -> dict[str, Any] | None:
    if score is None:
        return None
    return {
        "schema": score.schema,
        "diagnostic_score": score.diagnostic_score,
        "certified_score": score.certified_score,
        "dimensions": {
            name: {"earned": value.earned, "maximum": value.maximum} for name, value in score.dimensions.items()
        },
        "certification_gates": score.certification_gates,
        "evidence": score.evidence,
    }


def _diagnostic_label(score: ScorecardResult | None) -> str:
    return "unavailable" if score is None else f"{score.diagnostic_score:.2f} / 100"


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(
    report: ReproductionReport,
    *,
    console: Console | None = None,
) -> None:
    target = console or Console()
    score = report.score
    if score is not None:
        certification = "certified" if score.certified_score is not None else "not certified"
        target.print(
            f"CodeFlowBench score: {score.diagnostic_score:.2f} / 100 (diagnostic, {certification})",
        )
        dimensions = Table(title="Dimensions")
        dimensions.add_column("Dimension")
        dimensions.add_column("Score", justify="right")
        for name, value in score.dimensions.items():
            dimensions.add_row(
                _DIMENSION_LABELS.get(name, name.replace("_", " ").title()),
                f"{value.earned:.2f} / {value.maximum:.2f}",
            )
        target.print(dimensions)
        checks = Table(title="Certification checks")
        checks.add_column("Check")
        checks.add_column("Passed")
        for name, passed in score.certification_gates.items():
            checks.add_row(
                _GATE_LABELS.get(name, name.replace("_", " ").title()),
                "yes" if passed else "no",
            )
        target.print(checks)

    stages = Table(title="Evidence")
    stages.add_column("Stage")
    stages.add_column("Status")
    stages.add_column("Artifact")
    for stage in report.stages:
        stages.add_row(
            _STAGE_LABELS.get(stage.name, stage.name),
            stage.status,
            str(stage.evidence_path or stage.error or "-"),
        )
    target.print(stages)
    target.print(f"Report: {report.output_root}")


def _execute_command(command: tuple[str, ...]) -> str:
    print(f"\n$ {shlex.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[str] = []
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
    return_code = process.wait()
    output = "".join(lines)
    if return_code != 0:
        tail = "".join(lines[-20:]).strip()
        raise ReproductionError(
            f"command exited with status {return_code}: {tail or shlex.join(command)}",
        )
    return output


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or reuse every CodeFlowBench Scorecard track and render one report.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--execute-paid-campaign",
        action="store_true",
        default=os.environ.get("CODEFLOW_BENCH_EXECUTE_PAID") == "1",
    )
    parser.add_argument("--formal-summary", type=Path)
    parser.add_argument("--runtime-evidence", type=Path)
    parser.add_argument("--tokenwise-report", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    codeflow_commit = _current_codeflow_commit()
    output_root = args.output_root or _environment_path("CODEFLOW_BENCH_REPRODUCTION_ROOT")
    if output_root is None:
        output_root = (
            Path(__file__).resolve().parents[2] / ".codeflow" / "evidence" / "codeflowbench-reproduction" / codeflow_commit
        )
    report = run_reproduction(
        ReproductionConfig(
            output_root=output_root,
            execute_paid_campaign=args.execute_paid_campaign,
            formal_summary=(args.formal_summary or _environment_path("CODEFLOW_SCORECARD_FORMAL_SUMMARY")),
            runtime_evidence=(args.runtime_evidence or _environment_path("CODEFLOW_SCORECARD_RUNTIME_EVIDENCE")),
            tokenwise_report=(args.tokenwise_report or _environment_path("CODEFLOW_SCORECARD_TOKENWISE_REPORT")),
        ),
        execute_command=_execute_command,
    )
    render_report(report)
    return 0 if report.status == "completed" else 2


__all__ = [
    "ReproductionReport",
    "ReproductionConfig",
    "StageResult",
    "render_report",
    "run_reproduction",
]


if __name__ == "__main__":
    raise SystemExit(main())
