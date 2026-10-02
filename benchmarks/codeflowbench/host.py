from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codeflow.agent.spine_runner import AgentTurnRunner
from codeflow.spine.delivery import Capabilities, DeliveryHub
from codeflow.spine.events import (
    Deliverable,
    TurnEnded,
    TurnEvent,
    TurnFailed,
)
from codeflow.spine.runner import TurnOutcome
from codeflow.spine.scheduler import OriginPools, Scheduler
from codeflow.spine.turn import TurnRequest

from .records import DeliveryOutcome, TurnTerminalState


class RecordingOutlet:
    capabilities = Capabilities()

    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.events: list[Deliverable] = []

    async def deliver(self, out: Deliverable) -> None:
        if self.fail:
            raise RuntimeError("injected delivery failure")
        self.events.append(out)


@dataclass(frozen=True)
class HostTurnObservation:
    outcome: TurnOutcome | None
    events: tuple[TurnEvent, ...]
    runtime_state: TurnTerminalState
    delivery_state: DeliveryOutcome
    failure_category: str | None


class RuntimeTrialHost:
    def __init__(
        self,
        *,
        assembly: Any,
        outlet: RecordingOutlet,
        user_concurrency: int = 1,
        system_concurrency: int = 1,
        delivery_retries: int = 0,
    ) -> None:
        self.assembly = assembly
        self.outlet = outlet
        self.runner = AgentTurnRunner(assembly.agent_loop, stream=False)
        self.hub = DeliveryHub(
            send_max_retries=delivery_retries,
            on_delivery_failure=self._delivery_failed,
        )
        self.hub.register(outlet)
        self._events: list[TurnEvent] = []
        self._delivery_failures = 0
        self.scheduler = Scheduler(
            self.runner,
            OriginPools(user=user_concurrency, system=system_concurrency),
            self._sink,
        )
        self._closed = False

    @classmethod
    async def build(
        cls,
        *,
        config: Any,
        codeflow_config: Any,
        provider: Any,
        cron_service: Any,
        outlet: RecordingOutlet,
        context_engine_factory: Any = None,
        user_concurrency: int = 1,
        system_concurrency: int = 1,
        delivery_retries: int = 0,
    ) -> RuntimeTrialHost:
        from codeflow.cli._runtime_assembly import assemble_runtime

        assembly = assemble_runtime(
            config,
            codeflow_config,
            provider=provider,
            cron_service=cron_service,
            interactive=False,
            context_engine_factory=context_engine_factory,
        )
        start_backend = getattr(assembly, "start_memory_backend", None)
        if start_backend is not None:
            await start_backend()
        return cls(
            assembly=assembly,
            outlet=outlet,
            user_concurrency=user_concurrency,
            system_concurrency=system_concurrency,
            delivery_retries=delivery_retries,
        )

    async def _delivery_failed(self, _notice) -> None:
        self._delivery_failures += 1

    async def _sink(self, event: TurnEvent) -> None:
        self._events.append(event)
        if not isinstance(event, (TurnEnded, TurnFailed)):
            from codeflow.spine.events import TurnStarted

            if not isinstance(event, TurnStarted):
                await self.hub.dispatch(event)

    async def run(self, request: TurnRequest) -> HostTurnObservation:
        start = len(self._events)
        delivered_start = len(self.outlet.events)
        failures_start = self._delivery_failures
        outcome = await self.scheduler.submit(request).result()
        await self.hub.wait_idle(request.source.channel)
        events = tuple(self._events[start:])
        terminal = _runtime_state(events, outcome)
        failure_category = _failure_category(events)
        if request.source.channel != self.outlet.name:
            delivery = DeliveryOutcome.NO_OUTLET
        elif self._delivery_failures > failures_start:
            delivery = DeliveryOutcome.DROPPED
        elif len(self.outlet.events) > delivered_start:
            delivery = DeliveryOutcome.DELIVERED
        else:
            delivery = DeliveryOutcome.DROPPED
        return HostTurnObservation(
            outcome=outcome,
            events=events,
            runtime_state=terminal,
            delivery_state=delivery,
            failure_category=failure_category,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.scheduler.shutdown(grace=5.0)
            await self.hub.wait_idle(self.outlet.name)
        finally:
            await self.hub.aclose()
            await self.assembly.close()


def _runtime_state(
    events: tuple[TurnEvent, ...],
    outcome: TurnOutcome | None,
) -> TurnTerminalState:
    failure = next((event for event in reversed(events) if isinstance(event, TurnFailed)), None)
    if failure is not None:
        if failure.cancelled:
            return TurnTerminalState.CANCELLED
        if failure.error.startswith("provider_error:"):
            return TurnTerminalState.PROVIDER_FAILED
        return TurnTerminalState.ERROR
    if outcome is not None and outcome.tool_failures:
        return TurnTerminalState.COMPLETED_WITH_TOOL_FAILURE
    return TurnTerminalState.COMPLETED


def _failure_category(
    events: tuple[TurnEvent, ...],
) -> str | None:
    failure = next((event for event in reversed(events) if isinstance(event, TurnFailed)), None)
    if failure is None:
        return None
    if failure.cancelled:
        return "cancelled"
    prefix = "provider_error:"
    if failure.error.startswith(prefix):
        return failure.error.removeprefix(prefix)
    return "runtime_error"
