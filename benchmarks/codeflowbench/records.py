from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from .schema import JsonValue


class RuntimeTrackState(StrEnum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"


class DeliveryOutcome(StrEnum):
    DELIVERED = "delivered"
    DROPPED = "dropped"
    NO_OUTLET = "no_outlet"


class VerificationState(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


class TrialStatus(StrEnum):
    PASSED = "passed"
    TASK_FAILED = "task_failed"
    TASK_TIMEOUT = "task_timeout"
    PROVIDER_FAILURE = "provider_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    CANCELLED = "cancelled"
    INCONCLUSIVE = "inconclusive"


class RetrievalStatus(StrEnum):
    MEASURABLE = "measurable"
    PROVIDER_FAILURE = "provider_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    CANCELLED = "cancelled"
    INCONCLUSIVE = "inconclusive"


class TurnTerminalState(StrEnum):
    COMPLETED = "completed"
    COMPLETED_WITH_TOOL_FAILURE = "completed_with_tool_failure"
    PROVIDER_FAILED = "provider_failed"
    ERROR = "error"
    CANCELLED = "cancelled"


MEASURABLE_TRIAL_STATUSES = frozenset({TrialStatus.PASSED, TrialStatus.TASK_FAILED, TrialStatus.TASK_TIMEOUT})
MEASURABLE_RETRIEVAL_STATUSES = frozenset({RetrievalStatus.MEASURABLE})


@dataclass(frozen=True)
class TrialKey:
    experiment_id: str
    pack_id: str
    task_id: str
    variant_id: str
    repetition: int


@dataclass(frozen=True)
class ComparisonBlockKey:
    experiment_id: str
    pack_id: str
    task_id: str
    repetition: int


@dataclass(frozen=True)
class AttemptKey:
    block: ComparisonBlockKey
    variant_id: str
    block_attempt: int


@dataclass(frozen=True)
class PairKey:
    experiment_id: str
    pack_id: str
    treatment_axis: str
    task_id: str
    repetition: int
    control_variant_id: str
    treatment_variant_id: str


@dataclass(frozen=True)
class RetrievalCaseKey:
    experiment_id: str
    retrieval_suite_id: str
    query_id: str
    configuration_id: str


@dataclass(frozen=True)
class RetrievalQueryBlockKey:
    experiment_id: str
    retrieval_suite_id: str
    query_id: str


@dataclass(frozen=True)
class RetrievalAttemptKey:
    block: RetrievalQueryBlockKey
    configuration_id: str
    query_block_attempt: int


@dataclass(frozen=True)
class VerifierResult:
    state: VerificationState
    findings: tuple[str, ...] = ()
    metrics: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True)
class AttemptRecord:
    key: AttemptKey
    plan_digest: str
    status: TrialStatus
    runtime_state: TurnTerminalState | None
    delivery_state: DeliveryOutcome | None
    verification: VerifierResult
    declared_variant_settings: dict[str, JsonValue]
    observed_variant_settings: dict[str, JsonValue]
    metrics: dict[str, JsonValue] = field(default_factory=dict)
    findings: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrialRecord:
    key: TrialKey
    plan_digest: str
    pair_memberships: tuple[PairKey, ...]
    status: TrialStatus
    runtime_state: TurnTerminalState | None
    delivery_state: DeliveryOutcome | None
    verification: VerifierResult
    selected_block_attempt: int
    attempt_refs: tuple[str, ...]
    declared_variant_settings: dict[str, JsonValue]
    observed_variant_settings: dict[str, JsonValue]
    metrics: dict[str, JsonValue] = field(default_factory=dict)
    findings: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairResult:
    key: PairKey
    plan_digest: str
    selected_block_attempt: int | None
    valid: bool
    actual_variant_diff: dict[str, JsonValue]
    findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComparisonBlockResult:
    key: ComparisonBlockKey
    plan_digest: str
    resolved: bool
    exhausted: bool
    selected_block_attempt: int
    variant_attempt_refs: tuple[str, ...]
    findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalAttemptRecord:
    key: RetrievalAttemptKey
    plan_digest: str
    status: RetrievalStatus
    label: str
    expected_item_ids: tuple[str, ...]
    ranked_results: tuple[dict[str, JsonValue], ...] = ()
    injected_results: tuple[dict[str, JsonValue], ...] = ()
    usage: dict[str, JsonValue] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalCaseRecord:
    key: RetrievalCaseKey
    plan_digest: str
    query_block: RetrievalQueryBlockKey
    status: RetrievalStatus
    label: str
    expected_item_ids: tuple[str, ...]
    selected_query_block_attempt: int
    attempt_refs: tuple[str, ...]
    ranked_results: tuple[dict[str, JsonValue], ...] = ()
    injected_results: tuple[dict[str, JsonValue], ...] = ()
    usage: dict[str, JsonValue] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalQueryBlockResult:
    key: RetrievalQueryBlockKey
    plan_digest: str
    resolved: bool
    exhausted: bool
    selected_query_block_attempt: int
    configuration_attempt_refs: tuple[str, ...]
    findings: tuple[str, ...] = ()
