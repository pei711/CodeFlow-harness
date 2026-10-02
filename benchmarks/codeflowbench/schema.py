from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

from .canonical import to_primitive

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonMapping: TypeAlias = dict[str, JsonValue]

EXPERIMENT_SCHEMA = "codeflow.codeflowbench.experiment.v1"
EVIDENCE_SCHEMA = "codeflow.codeflowbench.evidence.v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RETRIEVAL_LABELS = frozenset(
    {
        "positive",
        "hard_negative",
        "positive_repository_fact",
        "positive_execution_experience",
        "stale_or_superseded",
        "cross_repository",
    }
)
_MINIMUM_VALID_PAIRS_PER_TASK = "minimum_valid_pairs_per_task"


def _identifier(value: str, field_name: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field_name} must be a portable identifier: {value!r}")


def _mapping(value: dict[str, JsonValue] | None) -> MappingProxyType[str, JsonValue]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class ProviderTrialBudget:
    pack_id: str
    max_provider_calls_per_trial: int
    max_input_tokens_per_call: int
    max_output_tokens_per_call: int

    def __post_init__(self) -> None:
        _identifier(self.pack_id, "pack_id")
        if self.max_provider_calls_per_trial < 1:
            raise ValueError(
                "max_provider_calls_per_trial must be at least one",
            )
        if self.max_input_tokens_per_call < 1:
            raise ValueError(
                "max_input_tokens_per_call must be at least one",
            )
        if self.max_output_tokens_per_call < 1:
            raise ValueError(
                "max_output_tokens_per_call must be at least one",
            )


@dataclass(frozen=True)
class ExecutionPolicy:
    timeout_seconds: float = 180.0
    retry_policy: str = "symmetric"
    provider_call_max_attempts: int = 2
    max_comparison_block_attempts: int = 2
    max_comparison_block_retries_total: int | None = None
    max_retrieval_query_block_attempts: int = 2
    max_provider_calls_per_trial: int | None = None
    provider_trial_budgets: tuple[ProviderTrialBudget, ...] = ()

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.retry_policy != "symmetric":
            raise ValueError("Ship-1 requires symmetric retry policy")
        for name in (
            "provider_call_max_attempts",
            "max_comparison_block_attempts",
            "max_retrieval_query_block_attempts",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one")
        if self.max_provider_calls_per_trial is not None and self.max_provider_calls_per_trial < 1:
            raise ValueError(
                "max_provider_calls_per_trial must be at least one",
            )
        if self.max_comparison_block_retries_total is not None and self.max_comparison_block_retries_total < 0:
            raise ValueError(
                "max_comparison_block_retries_total must not be negative",
            )
        _ensure_unique(
            (budget.pack_id for budget in self.provider_trial_budgets),
            "Provider Trial budget pack_id",
        )
        if self.max_provider_calls_per_trial is not None and any(
            budget.max_provider_calls_per_trial > self.max_provider_calls_per_trial
            for budget in self.provider_trial_budgets
        ):
            raise ValueError(
                "a Provider Trial budget exceeds the execution call ceiling",
            )

    def provider_trial_budget_for(
        self,
        pack_id: str,
    ) -> ProviderTrialBudget | None:
        return next(
            (budget for budget in self.provider_trial_budgets if budget.pack_id == pack_id),
            None,
        )


@dataclass(frozen=True)
class ClaimRule:
    rule_id: str
    metric: str
    operator: str
    threshold: int | float
    prerequisites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.rule_id, "rule_id")
        if self.operator not in {"eq", "ge", "gt", "le", "lt"}:
            raise ValueError(f"unsupported claim operator: {self.operator}")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    payload: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.task_id, "task_id")
        object.__setattr__(self, "payload", _mapping(self.payload))


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    settings: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _identifier(self.variant_id, "variant_id")
        object.__setattr__(self, "settings", _mapping(self.settings))


@dataclass(frozen=True)
class PairSpec:
    treatment_axis: str
    control_variant_id: str
    treatment_variant_id: str

    def __post_init__(self) -> None:
        _identifier(self.treatment_axis, "treatment_axis")
        _identifier(self.control_variant_id, "control_variant_id")
        _identifier(self.treatment_variant_id, "treatment_variant_id")
        if self.control_variant_id == self.treatment_variant_id:
            raise ValueError("a Pair requires distinct variants")


@dataclass(frozen=True)
class RetrievalQuerySpec:
    query_id: str
    label: str
    expected_item_ids: tuple[str, ...] = ()
    payload: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.query_id, "query_id")
        if self.label not in _RETRIEVAL_LABELS:
            raise ValueError(f"unsupported retrieval label: {self.label}")
        object.__setattr__(self, "payload", _mapping(self.payload))


@dataclass(frozen=True)
class RetrievalConfigurationSpec:
    configuration_id: str
    settings: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _identifier(self.configuration_id, "configuration_id")
        object.__setattr__(self, "settings", _mapping(self.settings))


@dataclass(frozen=True)
class RetrievalSuiteSpec:
    retrieval_suite_id: str
    queries: tuple[RetrievalQuerySpec, ...]
    configurations: tuple[RetrievalConfigurationSpec, ...]
    corpus_digest: str
    query_labels_digest: str

    def __post_init__(self) -> None:
        _identifier(self.retrieval_suite_id, "retrieval_suite_id")
        if not self.queries:
            raise ValueError("a retrieval suite requires queries")
        if not self.configurations:
            raise ValueError("a retrieval suite requires configurations")


@dataclass(frozen=True)
class PackDefinition:
    pack_id: str
    tasks: tuple[TaskSpec, ...]
    variants: tuple[VariantSpec, ...]
    pairs: tuple[PairSpec, ...]
    retrieval_suites: tuple[RetrievalSuiteSpec, ...] = ()
    identity: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.pack_id, "pack_id")
        if not self.tasks and not self.retrieval_suites:
            raise ValueError("a Pack requires tasks or retrieval suites")
        if self.tasks and not self.variants:
            raise ValueError("a task Pack requires variants")
        identity = dict(self.identity)
        minimum_valid_pairs = identity.get(_MINIMUM_VALID_PAIRS_PER_TASK)
        if minimum_valid_pairs is not None and (
            not isinstance(minimum_valid_pairs, int) or isinstance(minimum_valid_pairs, bool) or minimum_valid_pairs < 1
        ):
            raise ValueError(
                f"{_MINIMUM_VALID_PAIRS_PER_TASK} must be a positive integer",
            )
        object.__setattr__(self, "identity", _mapping(identity))
        _ensure_unique((task.task_id for task in self.tasks), "task_id")
        _ensure_unique((variant.variant_id for variant in self.variants), "variant_id")
        _ensure_unique(
            (suite.retrieval_suite_id for suite in self.retrieval_suites),
            "retrieval_suite_id",
        )


@dataclass(frozen=True)
class ExperimentSpec:
    suite: str
    repetitions: int
    pack_ids: tuple[str, ...]
    output_root: Path
    identity: dict[str, JsonValue]
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    claim_rules: tuple[ClaimRule, ...] = ()
    schema: str = EXPERIMENT_SCHEMA
    evidence_schema: str = EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        _identifier(self.suite, "suite")
        if self.repetitions < 1:
            raise ValueError("repetitions must be at least one")
        if not self.pack_ids:
            raise ValueError("pack_ids must not be empty")
        _ensure_unique(self.pack_ids, "pack_id")
        for pack_id in self.pack_ids:
            _identifier(pack_id, "pack_id")
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "identity", _mapping(self.identity))
        if self.schema != EXPERIMENT_SCHEMA:
            raise ValueError(f"unsupported experiment schema: {self.schema}")

    def canonical_payload(self) -> JsonMapping:
        return {
            "schema": self.schema,
            "evidence_schema": self.evidence_schema,
            "suite": self.suite,
            "repetitions": self.repetitions,
            "pack_ids": list(self.pack_ids),
            "identity": to_primitive(self.identity),
            "execution": to_primitive(self.execution),
            "claim_rules": to_primitive(self.claim_rules),
        }


@dataclass(frozen=True)
class ExperimentRef:
    experiment_id: str
    root: Path

    def __post_init__(self) -> None:
        _identifier(self.experiment_id, "experiment_id")
        object.__setattr__(self, "root", Path(self.root))


def _ensure_unique(values: Any, field_name: str) -> None:
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"duplicate {field_name}")
