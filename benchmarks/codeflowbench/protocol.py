from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from .records import (
    DeliveryOutcome,
    RetrievalCaseKey,
    RetrievalStatus,
    TrialKey,
    TrialStatus,
    TurnTerminalState,
    VerifierResult,
)
from .schema import (
    ExperimentSpec,
    JsonValue,
    PackDefinition,
    RetrievalConfigurationSpec,
    RetrievalQuerySpec,
    TaskSpec,
    VariantSpec,
)


@dataclass(frozen=True)
class TrialContext:
    experiment_id: str
    plan_digest: str
    key: TrialKey
    block_attempt: int
    experiment: ExperimentSpec
    task: TaskSpec
    variant: VariantSpec


@dataclass(frozen=True)
class RetrievalContext:
    experiment_id: str
    plan_digest: str
    key: RetrievalCaseKey
    query_block_attempt: int
    experiment: ExperimentSpec
    query: RetrievalQuerySpec
    configuration: RetrievalConfigurationSpec


@dataclass(frozen=True)
class TrialExecution:
    status: TrialStatus
    runtime_state: TurnTerminalState | None
    delivery_state: DeliveryOutcome | None
    verification: VerifierResult
    observed_variant_settings: dict[str, JsonValue]
    metrics: dict[str, JsonValue] = field(default_factory=dict)
    findings: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_variant_settings",
            MappingProxyType(dict(self.observed_variant_settings)),
        )
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True)
class RetrievalExecution:
    status: RetrievalStatus
    ranked_results: tuple[dict[str, JsonValue], ...] = ()
    injected_results: tuple[dict[str, JsonValue], ...] = ()
    usage: dict[str, JsonValue] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    findings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


class Pack(Protocol):
    def definition(self) -> PackDefinition: ...

    async def run_trial(self, context: TrialContext) -> TrialExecution: ...
