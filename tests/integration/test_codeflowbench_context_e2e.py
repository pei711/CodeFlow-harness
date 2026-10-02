from __future__ import annotations

import asyncio
from pathlib import Path

from benchmarks.codeflowbench.budget import (
    ProviderBudgetError,
    ProviderRequestNotDispatchedError,
)
from benchmarks.codeflowbench.packs.context import (
    ContextPack,
    ContextTrack,
    RuntimeContextTrialRunner,
    context_engine_factory_for,
    load_context_tasks,
)
from benchmarks.codeflowbench.protocol import TrialContext
from benchmarks.codeflowbench.records import TrialKey, TrialStatus
from benchmarks.codeflowbench.schema import ExperimentSpec
from codeflow.agent.context import ContextBuilder
from codeflow.agent.tools.base import Tool
from codeflow.config.codeflow import CodeFlowConfig, ContextConfig
from codeflow.config.schema import Config
from codeflow.context_engine import TurnContext
from codeflow.memory_engine.base import TokenBudget
from codeflow.providers.base import (
    ErrorClassification,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)


class _FallbackProvider:
    api_key = "test"

    def get_default_model(self) -> str:
        return "scripted/context"

    async def chat_with_retry(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(
            content="no executable curator plan",
            finish_reason="stop",
            usage={
                "prompt_tokens": 17,
                "completion_tokens": 4,
                "total_tokens": 21,
            },
        )


def _budget() -> TokenBudget:
    return TokenBudget(
        context_length=2_000,
        reserved_output=400,
        reserved_tools=0,
        reserved_system=300,
        available_history=1_300,
    )


async def test_fifo_and_curator_factories_share_phase_a_and_swap_phase_b(
    tmp_path: Path,
) -> None:
    task = load_context_tasks(ContextTrack.CALIBRATION)[0]
    history = task.materialize_history()
    provider = _FallbackProvider()
    builder = ContextBuilder(tmp_path, start_watcher=False)
    common = {
        "workspace": tmp_path,
        "config": ContextConfig(
            fast_path_threshold=0.10,
            curator_timeout_seconds=2.0,
        ),
        "builder": builder,
        "provider": provider,
        "model": "scripted/context",
        "context_window_tokens": 2_000,
        "get_tool_definitions": lambda: [],
    }

    fifo = context_engine_factory_for("fifo_tail")(**common)
    curator = context_engine_factory_for("curator")(**common)
    fifo_result = await fifo.assemble(
        "context-calibration",
        history,
        _budget(),
        turn=TurnContext(current_message=task.final_prompt),
    )
    curator_result = await curator.assemble(
        "context-calibration",
        history,
        _budget(),
        turn=TurnContext(current_message=task.final_prompt),
    )

    assert [type(builder).__name__ for builder in fifo._phase_a] == [
        type(builder).__name__ for builder in curator._phase_a
    ]
    assert [builder.name for builder in fifo._phase_b] == ["fifo_history_manager"]
    assert [builder.name for builder in curator._phase_b] == ["curator"]
    assert fifo_result.metadata["history_manager"] == "fifo_tail"
    assert curator_result.metadata["path"] in {"slow", "fallback"}
    assert task.early_constraint not in "\n".join(str(message.get("content", "")) for message in fifo_result.messages)
    assert task.early_constraint in "\n".join(str(message.get("content", "")) for message in curator_result.messages)


class _ArtifactProvider(LLMProvider):
    def __init__(self, task) -> None:
        super().__init__()
        self.task = task
        self._step = 0
        self.max_tokens_seen: list[int] = []
        self.tool_names_seen: list[set[str]] = []

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ) -> LLMResponse:
        del model, temperature, reasoning_effort, tool_choice
        self.max_tokens_seen.append(max_tokens)
        is_curator = any("You are CodeFlow Curator" in str(message.get("content", "")) for message in messages)
        if is_curator:
            return LLMResponse(
                content="no executable curator plan",
                usage={
                    "prompt_tokens": 17,
                    "completion_tokens": 4,
                    "total_tokens": 21,
                },
            )
        self.tool_names_seen.append({str(definition["function"]["name"]) for definition in tools or []})
        self._step += 1
        usage = {
            "prompt_tokens": 40,
            "completion_tokens": 10,
            "total_tokens": 50,
        }
        if self._step == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="write-context-artifact",
                        name="write_file",
                        arguments={
                            "path": self.task.artifact_path,
                            "content": self.task.expected_path.read_text(
                                encoding="utf-8",
                            ),
                        },
                    )
                ],
                usage=usage,
            )
        return LLMResponse(content="artifact created", usage=usage)

    def get_default_model(self) -> str:
        return "scripted/context-artifact"


class _BudgetFallbackArtifactProvider(_ArtifactProvider):
    def __init__(self, task) -> None:
        super().__init__(task)
        self._curator_step = 0

    async def chat(self, messages, **kwargs) -> LLMResponse:
        is_curator = any("You are CodeFlow Curator" in str(message.get("content", "")) for message in messages)
        if not is_curator:
            return await super().chat(messages, **kwargs)
        self._curator_step += 1
        if self._curator_step == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="check-curator-budget",
                        name="curator_check_budget",
                        arguments={"include_message_ids": [0]},
                    ),
                ],
                usage={
                    "prompt_tokens": 17,
                    "completion_tokens": 4,
                    "total_tokens": 21,
                },
            )
        raise ProviderRequestNotDispatchedError(
            "estimated input tokens exceed the per-call ceiling",
        )

    def classify_error(
        self,
        exc,
        content=None,
    ) -> ErrorClassification:
        del content
        if isinstance(exc, ProviderBudgetError):
            return ErrorClassification("task_budget_exhausted")
        return super().classify_error(exc)


class _CancelledCuratorArtifactProvider(_ArtifactProvider):
    async def chat(self, messages, **kwargs) -> LLMResponse:
        is_curator = any("You are CodeFlow Curator" in str(message.get("content", "")) for message in messages)
        if is_curator:
            await asyncio.sleep(1)
        return await super().chat(messages, **kwargs)


class _PluginLeakTool(Tool):
    @property
    def name(self) -> str:
        return "plugin_leak"

    @property
    def description(self) -> str:
        return "must not enter a Context Trial"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        del kwargs
        return "leaked"


async def test_runtime_context_runner_excludes_plugin_tools_and_uses_sealed_verifier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = load_context_tasks(ContextTrack.CALIBRATION)[0]
    provider = _ArtifactProvider(task)
    plugin_tool_builds = 0

    def build_plugin_tools(*args, **kwargs):
        nonlocal plugin_tool_builds
        del args, kwargs
        plugin_tool_builds += 1
        return [_PluginLeakTool()]

    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.build_plugin_tools",
        build_plugin_tools,
    )
    config = Config()
    config.agents.defaults.model = provider.get_default_model()
    codeflow_config = CodeFlowConfig(base=config)
    runner = RuntimeContextTrialRunner(
        config=config,
        codeflow_config=codeflow_config,
        provider=provider,
    )
    pack = ContextPack(ContextTrack.CALIBRATION, runner=runner)
    definition = pack.definition()
    variant = definition.variants[1]
    experiment = ExperimentSpec(
        suite="context-runtime-e2e",
        repetitions=1,
        pack_ids=(definition.pack_id,),
        output_root=tmp_path,
        identity={
            "codeflow_commit": "0" * 40,
            "provider": "scripted",
            "model": provider.get_default_model(),
        },
    )
    context = TrialContext(
        experiment_id="context-runtime-e2e",
        plan_digest="a" * 64,
        key=TrialKey(
            experiment_id="context-runtime-e2e",
            pack_id=definition.pack_id,
            task_id=task.task_id,
            variant_id=variant.variant_id,
            repetition=0,
        ),
        block_attempt=1,
        experiment=experiment,
        task=definition.tasks[0],
        variant=variant,
    )

    execution = await pack.run_trial(context)

    assert execution.status is TrialStatus.PASSED
    assert execution.verification.state.value == "passed"
    assert execution.metrics["main_agent_input_tokens"] == 80
    assert execution.metrics["trial_total_input_tokens"] == 97
    assert execution.metrics["context_auxiliary_input_tokens"] == 17
    assert execution.metrics["usage_complete"] is True
    assert execution.metrics["early_constraint_retained"] is True
    assert execution.metrics["active_constraint_applied"] is True
    assert execution.metrics["latest_decision_applied"] is True
    assert execution.metrics["artifact_exact"] is True
    assert execution.metrics["capability_criteria_passed"] == 4
    assert execution.metrics["capability_criteria_total"] == 4
    assert execution.metrics["context_path"] in {"fast", "slow", "fallback"}
    assert execution.artifact_refs
    assert set(provider.max_tokens_seen) == {500}
    assert all(names == {"read_file", "write_file"} for names in provider.tool_names_seen)
    assert plugin_tool_builds == 0


async def test_runtime_context_runner_counts_only_dispatched_provider_calls(
    tmp_path: Path,
) -> None:
    task = load_context_tasks(ContextTrack.CALIBRATION)[0]
    provider = _BudgetFallbackArtifactProvider(task)
    config = Config()
    config.agents.defaults.model = provider.get_default_model()
    pack = ContextPack(
        ContextTrack.CALIBRATION,
        runner=RuntimeContextTrialRunner(
            config=config,
            codeflow_config=CodeFlowConfig(base=config),
            provider=provider,
        ),
    )
    definition = pack.definition()
    variant = definition.variants[1]
    experiment = ExperimentSpec(
        suite="context-runtime-budget-fallback",
        repetitions=1,
        pack_ids=(definition.pack_id,),
        output_root=tmp_path,
        identity={"variant": variant.variant_id},
    )

    execution = await pack.run_trial(
        TrialContext(
            experiment_id=experiment.suite,
            plan_digest="c" * 64,
            key=TrialKey(
                experiment_id=experiment.suite,
                pack_id=definition.pack_id,
                task_id=task.task_id,
                variant_id=variant.variant_id,
                repetition=0,
            ),
            block_attempt=1,
            experiment=experiment,
            task=definition.tasks[0],
            variant=variant,
        ),
    )

    assert execution.status is TrialStatus.PASSED
    assert execution.metrics["context_path"] == "fallback"
    assert execution.metrics["main_agent_input_tokens"] == 80
    assert execution.metrics["context_auxiliary_input_tokens"] == 17
    assert execution.metrics["trial_total_input_tokens"] == 97
    assert execution.metrics["usage_complete"] is True


async def test_runtime_context_runner_classifies_dispatched_cancelled_usage_for_retry(
    tmp_path: Path,
) -> None:
    task = load_context_tasks(ContextTrack.CALIBRATION)[0]
    provider = _CancelledCuratorArtifactProvider(task)
    config = Config()
    config.agents.defaults.model = provider.get_default_model()
    codeflow_config = CodeFlowConfig(base=config)
    codeflow_config.context.curator_timeout_seconds = 0.01
    pack = ContextPack(
        ContextTrack.CALIBRATION,
        runner=RuntimeContextTrialRunner(
            config=config,
            codeflow_config=codeflow_config,
            provider=provider,
        ),
    )
    definition = pack.definition()
    variant = definition.variants[1]
    experiment = ExperimentSpec(
        suite="context-runtime-cancelled-usage",
        repetitions=1,
        pack_ids=(definition.pack_id,),
        output_root=tmp_path,
        identity={"variant": variant.variant_id},
    )

    execution = await pack.run_trial(
        TrialContext(
            experiment_id=experiment.suite,
            plan_digest="d" * 64,
            key=TrialKey(
                experiment_id=experiment.suite,
                pack_id=definition.pack_id,
                task_id=task.task_id,
                variant_id=variant.variant_id,
                repetition=0,
            ),
            block_attempt=1,
            experiment=experiment,
            task=definition.tasks[0],
            variant=variant,
        ),
    )

    assert execution.status is TrialStatus.INFRASTRUCTURE_FAILURE
    assert execution.metrics["context_path"] == "fallback"
    assert execution.metrics["usage_complete"] is False
    assert "trial_usage_incomplete" in execution.findings


async def test_runtime_context_variants_exercise_distinct_history_paths(
    tmp_path: Path,
) -> None:
    task = load_context_tasks(ContextTrack.CALIBRATION)[0]
    retained: dict[str, bool] = {}

    for variant_index in (0, 1):
        provider = _ArtifactProvider(task)
        config = Config()
        config.agents.defaults.model = provider.get_default_model()
        pack = ContextPack(
            ContextTrack.CALIBRATION,
            runner=RuntimeContextTrialRunner(
                config=config,
                codeflow_config=CodeFlowConfig(base=config),
                provider=provider,
            ),
        )
        definition = pack.definition()
        variant = definition.variants[variant_index]
        experiment = ExperimentSpec(
            suite=f"context-runtime-path-{variant_index}",
            repetitions=1,
            pack_ids=(definition.pack_id,),
            output_root=tmp_path / str(variant_index),
            identity={"variant": variant.variant_id},
        )
        execution = await pack.run_trial(
            TrialContext(
                experiment_id=experiment.suite,
                plan_digest="b" * 64,
                key=TrialKey(
                    experiment_id=experiment.suite,
                    pack_id=definition.pack_id,
                    task_id=task.task_id,
                    variant_id=variant.variant_id,
                    repetition=0,
                ),
                block_attempt=1,
                experiment=experiment,
                task=definition.tasks[0],
                variant=variant,
            ),
        )
        assert execution.status is TrialStatus.PASSED
        retained[str(variant.settings["history_manager"])] = bool(
            execution.metrics["early_constraint_retained"],
        )

    assert retained == {
        "fifo_tail": False,
        "curator": True,
    }
