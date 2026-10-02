from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore, artifact_dict
from .canonical import canonical_digest
from .claims import ClaimRuleResult, evaluate_claim_rules
from .reducer import PairSummary, reduce_experiment
from .schema import ClaimRule, ExperimentRef

_DEFAULT_REPORT_TITLE = "CodeFlowBench Ship-1 Report"


@dataclass(frozen=True)
class FullReport:
    experiment_id: str
    report_digest: str
    ship_complete: bool
    measurement_valid: bool
    positive_claim_eligible: bool
    planned_trials: int
    terminal_trials: int
    planned_retrieval_cases: int
    terminal_retrieval_cases: int
    metrics: dict[str, Any]
    pair_summaries: tuple[PairSummary, ...]
    claim_results: tuple[ClaimRuleResult, ...]
    selected_status_counts: dict[str, int]
    first_attempt_status_counts: dict[str, int]
    all_attempt_status_counts: dict[str, int]
    retrieval_status_counts: dict[str, int]
    retrieval_first_attempt_status_counts: dict[str, int]
    retrieval_all_attempt_status_counts: dict[str, int]
    findings: tuple[str, ...]
    report_title: str = field(
        default=_DEFAULT_REPORT_TITLE,
        repr=False,
    )

    @property
    def status_counts(self) -> dict[str, int]:
        return self.selected_status_counts


def rebuild_full_report(ref: ExperimentRef) -> FullReport:
    store = ArtifactStore(ref)
    manifest = store.read_json(store.manifest_path)
    reduction = reduce_experiment(ref)
    rules = tuple(
        ClaimRule(
            rule_id=str(rule["rule_id"]),
            metric=str(rule["metric"]),
            operator=str(rule["operator"]),
            threshold=rule["threshold"],
            prerequisites=tuple(rule.get("prerequisites", ())),
        )
        for rule in manifest.get("spec", {}).get("claim_rules", ())
    )
    claims = evaluate_claim_rules(
        rules,
        metrics=reduction.metrics,
        ship_complete=reduction.ship_complete,
        measurement_valid=reduction.measurement_valid,
    )
    payload = {
        "experiment_id": ref.experiment_id,
        "ship_complete": reduction.ship_complete,
        "measurement_valid": reduction.measurement_valid,
        "positive_claim_eligible": claims.positive_claim_eligible,
        "planned_trials": reduction.planned_trials,
        "terminal_trials": reduction.terminal_trials,
        "planned_retrieval_cases": reduction.planned_retrieval_cases,
        "terminal_retrieval_cases": reduction.terminal_retrieval_cases,
        "metrics": reduction.metrics,
        "pair_summaries": reduction.pair_summaries,
        "claim_results": claims.rules,
        "selected_status_counts": reduction.selected_status_counts,
        "first_attempt_status_counts": reduction.first_attempt_status_counts,
        "all_attempt_status_counts": reduction.all_attempt_status_counts,
        "retrieval_status_counts": reduction.retrieval_status_counts,
        "retrieval_first_attempt_status_counts": (reduction.retrieval_first_attempt_status_counts),
        "retrieval_all_attempt_status_counts": (reduction.retrieval_all_attempt_status_counts),
        "findings": reduction.findings,
    }
    report = FullReport(
        report_digest=canonical_digest(payload),
        report_title=_report_title(manifest),
        **payload,
    )
    summary = artifact_dict(report)
    store.write_summary(ref.root / "summary.json", summary)
    _write_cv_metrics(store, report)
    _write_markdown(ref.root / "REPORT.md", report)
    return report


def _write_cv_metrics(store: ArtifactStore, report: FullReport) -> None:
    eligible = _eligible_claim_metrics(report)
    payload = {
        "experiment_id": report.experiment_id,
        "report_digest": report.report_digest,
        "ship_complete": report.ship_complete,
        "measurement_valid": report.measurement_valid,
        "positive_claim_eligible": report.positive_claim_eligible,
        "eligible_metrics": eligible,
    }
    store.write_summary(store.root / "cv-metrics.json", payload)


def _eligible_claim_metrics(report: FullReport) -> dict[str, Any]:
    if not report.ship_complete or not report.measurement_valid:
        return {}
    groups: dict[str, list[ClaimRuleResult]] = {}
    for result in report.claim_results:
        groups.setdefault(_claim_group(result.metric), []).append(result)
    eligible_groups = {
        group for group, results in groups.items() if results and all(result.passed for result in results)
    }
    eligible = {
        result.metric: report.metrics.get(result.metric)
        for result in report.claim_results
        if _claim_group(result.metric) in eligible_groups and _claim_metric_exportable(result.metric, report.metrics)
    }
    return eligible


def _claim_metric_exportable(metric: str, metrics: dict[str, Any]) -> bool:
    required_flag = {
        "memory_retrieval": "memory_retrieval.real_semantic_claim_eligible",
        "skill_fusion": "skill_fusion.real_semantic_claim_eligible",
    }.get(metric.partition(".")[0])
    return required_flag is None or metrics.get(required_flag) is True


def _claim_group(metric: str) -> str:
    prefix = metric.partition(".")[0]
    return {
        "memory_e2e": "memory",
        "memory_retrieval": "memory",
        "skill_e2e": "skill",
        "skill_fusion": "skill",
    }.get(prefix, prefix)


def _write_markdown(path: Path, report: FullReport) -> None:
    lines = [
        f"# {report.report_title}",
        "",
        f"- Experiment: `{report.experiment_id}`",
        f"- Report digest: `{report.report_digest}`",
        f"- Ship complete: `{str(report.ship_complete).lower()}`",
        f"- Measurement valid: `{str(report.measurement_valid).lower()}`",
        (f"- Positive claim eligible: `{str(report.positive_claim_eligible).lower()}`"),
    ]
    lines.extend(
        [
            "",
            "## Denominators",
            "",
            f"- Trials: {report.terminal_trials}/{report.planned_trials}",
            (f"- Retrieval cases: {report.terminal_retrieval_cases}/{report.planned_retrieval_cases}"),
            "",
            "## Operational status",
            "",
            f"- Trial selected: `{report.selected_status_counts}`",
            f"- Trial first attempts: `{report.first_attempt_status_counts}`",
            f"- Trial all attempts: `{report.all_attempt_status_counts}`",
            f"- Retrieval selected: `{report.retrieval_status_counts}`",
            (f"- Retrieval first attempts: `{report.retrieval_first_attempt_status_counts}`"),
            (f"- Retrieval all attempts: `{report.retrieval_all_attempt_status_counts}`"),
            "",
            "## Pair summaries",
            "",
        ]
    )
    for summary in report.pair_summaries:
        lines.append(
            f"- {summary.pack_id}/{summary.treatment_axis}: "
            f"{summary.valid_pairs}/{summary.planned_pairs} valid pairs, "
            f"coverage gate={str(summary.coverage_valid).lower()}"
        )
    lines.extend(["", "## Claim rules", ""])
    for result in report.claim_results:
        lines.append(f"- {result.rule_id}: {result.reason}; observed={result.observed}, threshold={result.threshold}")
    lines.extend(["", "## Findings", ""])
    if report.findings:
        lines.extend(f"- {finding}" for finding in report.findings)
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report_title(manifest: dict[str, Any]) -> str:
    titles = {
        title
        for definition in manifest.get("pack_definitions", ())
        if isinstance(definition, dict)
        and isinstance(identity := definition.get("identity"), dict)
        and isinstance(title := identity.get("report_title"), str)
        and title
    }
    if len(titles) > 1:
        raise ValueError(
            "Pack definitions declare conflicting report titles",
        )
    return next(iter(titles), _DEFAULT_REPORT_TITLE)
