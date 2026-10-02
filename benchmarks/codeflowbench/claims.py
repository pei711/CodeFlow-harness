from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import ClaimRule


@dataclass(frozen=True)
class ClaimRuleResult:
    rule_id: str
    metric: str
    passed: bool
    observed: int | float | None
    threshold: int | float
    reason: str


@dataclass(frozen=True)
class ClaimEvaluation:
    ship_complete: bool
    measurement_valid: bool
    positive_claim_eligible: bool
    rules: tuple[ClaimRuleResult, ...]


def evaluate_claim_rules(
    rules: tuple[ClaimRule, ...],
    *,
    metrics: dict[str, Any],
    ship_complete: bool,
    measurement_valid: bool,
) -> ClaimEvaluation:
    results = tuple(_evaluate_rule(rule, metrics) for rule in rules)
    eligible = ship_complete and measurement_valid and bool(results) and all(result.passed for result in results)
    return ClaimEvaluation(
        ship_complete=ship_complete,
        measurement_valid=measurement_valid,
        positive_claim_eligible=eligible,
        rules=results,
    )


def _evaluate_rule(
    rule: ClaimRule,
    metrics: dict[str, Any],
) -> ClaimRuleResult:
    if any(metrics.get(prerequisite) is not True for prerequisite in rule.prerequisites):
        return ClaimRuleResult(
            rule_id=rule.rule_id,
            metric=rule.metric,
            passed=False,
            observed=_number_or_none(metrics.get(rule.metric)),
            threshold=rule.threshold,
            reason="prerequisite_not_met",
        )
    observed = _number_or_none(metrics.get(rule.metric))
    if observed is None:
        return ClaimRuleResult(
            rule_id=rule.rule_id,
            metric=rule.metric,
            passed=False,
            observed=None,
            threshold=rule.threshold,
            reason="metric_missing",
        )
    passed = {
        "eq": observed == rule.threshold,
        "ge": observed >= rule.threshold,
        "gt": observed > rule.threshold,
        "le": observed <= rule.threshold,
        "lt": observed < rule.threshold,
    }[rule.operator]
    return ClaimRuleResult(
        rule_id=rule.rule_id,
        metric=rule.metric,
        passed=passed,
        observed=observed,
        threshold=rule.threshold,
        reason="passed" if passed else "threshold_not_met",
    )


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value
