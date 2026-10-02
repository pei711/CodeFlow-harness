from __future__ import annotations

from pathlib import Path

from benchmarks.codeflowbench.campaign import (
    DEFAULT_SUITE_PATH,
    load_campaign_suite,
)
from benchmarks.codeflowbench.packs.tool_mcp import (
    DeterministicMCPTrialRunner,
    MCPRuntimeTrialRunner,
    ToolMCPPack,
    ToolMCPTrack,
    load_tool_mcp_tasks,
    run_mcp_transport_smoke,
)
from benchmarks.codeflowbench.protocol import TrialContext
from benchmarks.codeflowbench.records import TrialKey, TrialStatus, VerificationState
from benchmarks.codeflowbench.schema import ExperimentSpec
from codeflow.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)


class _InjectedProvider(LLMProvider):
    def __init__(
        self,
        *,
        tool_name: str,
        arguments: dict,
        complete_usage: bool = True,
    ) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.arguments = arguments
        self.complete_usage = complete_usage
        self.models: list[str | None] = []
        self.settings: list[tuple[int, float, str | None]] = []
        self.visible_tools: list[list[str]] = []
        self._step = 0

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
        del messages, tool_choice
        self.models.append(model)
        self.settings.append((max_tokens, temperature, reasoning_effort))
        self.visible_tools.append([definition["function"]["name"] for definition in (tools or [])])
        self._step += 1
        if self._step == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="injected-target",
                        name=self.tool_name,
                        arguments=self.arguments,
                    )
                ],
                usage={
                    "prompt_tokens": 101,
                    "completion_tokens": 7,
                    "total_tokens": 108,
                },
            )
        return LLMResponse(
            content="done",
            usage=(
                {
                    "prompt_tokens": 109,
                    "completion_tokens": 3,
                    "total_tokens": 112,
                }
                if self.complete_usage
                else {}
            ),
        )

    def get_default_model(self) -> str:
        return "delegate-default"


def _experiment(tmp_path: Path) -> ExperimentSpec:
    return ExperimentSpec(
        suite="tool-mcp-e2e",
        repetitions=1,
        pack_ids=("tool-mcp",),
        output_root=tmp_path,
        identity={
            "codeflow_commit": "0" * 40,
            "model": "scripted/tool-mcp",
            "budget_cap_cny": 100,
        },
    )


async def test_real_stdio_mcp_smoke_crosses_search_call_and_receipt(
    tmp_path: Path,
) -> None:
    result = await run_mcp_transport_smoke(tmp_path)

    assert result.transport == "stdio"
    assert result.catalog_count == 64
    assert result.search_hit_name == result.called_target_name
    assert result.receipt["tool"] == result.called_target_name.removeprefix("mcp_codeflowbench_")
    assert result.receipt["receipt"]
    assert result.transport_closed


async def test_formal_task_runs_both_variants_through_runtime_host(
    tmp_path: Path,
) -> None:
    runner = DeterministicMCPTrialRunner()
    pack = ToolMCPPack(runner=runner)
    definition = pack.definition()
    tasks = load_tool_mcp_tasks(ToolMCPTrack.FORMAL)
    task = max(tasks, key=lambda candidate: len(candidate.targets))
    task_spec = next(candidate for candidate in definition.tasks if candidate.task_id == task.task_id)
    experiment = _experiment(tmp_path)
    executions = {}

    for variant in definition.variants:
        context = TrialContext(
            experiment_id="tool-mcp-e2e",
            plan_digest="a" * 64,
            key=TrialKey(
                experiment_id="tool-mcp-e2e",
                pack_id=definition.pack_id,
                task_id=task.task_id,
                variant_id=variant.variant_id,
                repetition=0,
            ),
            block_attempt=1,
            experiment=experiment,
            task=task_spec,
            variant=variant,
        )
        executions[variant.variant_id] = await pack.run_trial(context)

    control = executions["tool-mcp-all-tools"]
    treatment = executions["tool-mcp-progressive"]
    for execution in executions.values():
        assert execution.status is TrialStatus.PASSED
        assert execution.verification.state is VerificationState.PASSED
        assert execution.metrics["mcp_transport"] == "stdio"
        assert execution.metrics["mcp_connected"] is True
        assert execution.metrics["mcp_catalog_count"] == 64
        assert execution.metrics["mcp_receipt_count"] == len(task.targets)
        assert execution.metrics["invalid_target_call_rate"] == 0.0
        assert execution.metrics["exact_target_repeat_rate"] == 0.0

    assert control.metrics["initial_visible_catalog_tool_count"] == 64
    assert treatment.metrics["initial_visible_catalog_tool_count"] == 0
    assert control.metrics["initial_visible_tool_count"] == 64
    assert treatment.metrics["initial_visible_tool_count"] == 2
    assert "understand_media" not in control.metrics["initial_visible_tool_names"]
    suite = load_campaign_suite(DEFAULT_SUITE_PATH)
    assert (
        max(
            record["conservative_serialized_input_tokens"]
            for execution in executions.values()
            for record in execution.metrics["model_call_records"]
        )
        <= suite.budget.max_input_tokens_per_call
    )
    assert (
        treatment.metrics["trial_total_estimated_visible_tool_schema_tokens"]
        < control.metrics["trial_total_estimated_visible_tool_schema_tokens"]
    )
    assert control.metrics["meta_tool_invocations"] == {
        "tool_call": 0,
        "tool_search": 0,
    }
    assert treatment.metrics["meta_tool_invocations"] == {
        "tool_call": len(task.targets),
        "tool_search": 1,
    }
    assert control.metrics["mcp_catalog_digest"] == treatment.metrics["mcp_catalog_digest"]


async def test_injected_provider_runner_records_real_decisions_and_full_usage(
    tmp_path: Path,
) -> None:
    task = load_tool_mcp_tasks(ToolMCPTrack.FORMAL)[0]
    provider = _InjectedProvider(
        tool_name=task.targets[0].runtime_name,
        arguments=dict(task.targets[0].arguments),
    )
    runner = MCPRuntimeTrialRunner(
        provider=provider,
        model="selected/provider-model",
        generation=GenerationSettings(
            max_tokens=512,
            temperature=0.0,
            reasoning_effort=None,
        ),
    )
    pack = ToolMCPPack(runner=runner)
    definition = pack.definition()
    variant = definition.variants[0]
    context = TrialContext(
        experiment_id="tool-mcp-real-runner",
        plan_digest="b" * 64,
        key=TrialKey(
            experiment_id="tool-mcp-real-runner",
            pack_id=definition.pack_id,
            task_id=task.task_id,
            variant_id=variant.variant_id,
            repetition=0,
        ),
        block_attempt=1,
        experiment=_experiment(tmp_path),
        task=definition.tasks[0],
        variant=variant,
    )

    execution = await pack.run_trial(context)

    assert execution.status is TrialStatus.PASSED
    assert provider.models == [
        "selected/provider-model",
        "selected/provider-model",
    ]
    assert provider.settings == [(512, 0.0, None), (512, 0.0, None)]
    assert execution.metrics["provider_model"] == "selected/provider-model"
    assert execution.metrics["provider_call_max_attempts"] == 2
    assert execution.metrics["generation_settings"] == {
        "temperature": 0.0,
        "max_tokens": 512,
        "reasoning_effort": None,
    }
    assert execution.metrics["main_agent_model_calls"] == 2
    assert execution.metrics["main_agent_input_tokens"] == 210
    assert execution.metrics["main_agent_output_tokens"] == 10
    assert execution.metrics["main_agent_total_tokens"] == 220
    assert execution.metrics["usage_complete"] is True
    records = execution.metrics["model_call_records"]
    assert records[0]["tool_calls"][0]["name"] == task.targets[0].runtime_name
    assert records[0]["usage"]["prompt_tokens"] == 101


async def test_incomplete_provider_usage_preserves_task_result_but_blocks_usage_claim(
    tmp_path: Path,
) -> None:
    task = load_tool_mcp_tasks(ToolMCPTrack.FORMAL)[0]
    provider = _InjectedProvider(
        tool_name=task.targets[0].runtime_name,
        arguments=dict(task.targets[0].arguments),
        complete_usage=False,
    )
    runner = MCPRuntimeTrialRunner(
        provider=provider,
        model="selected/provider-model",
    )
    pack = ToolMCPPack(runner=runner)
    definition = pack.definition()
    variant = definition.variants[0]
    context = TrialContext(
        experiment_id="tool-mcp-incomplete-usage",
        plan_digest="c" * 64,
        key=TrialKey(
            experiment_id="tool-mcp-incomplete-usage",
            pack_id=definition.pack_id,
            task_id=task.task_id,
            variant_id=variant.variant_id,
            repetition=0,
        ),
        block_attempt=1,
        experiment=_experiment(tmp_path),
        task=definition.tasks[0],
        variant=variant,
    )

    execution = await pack.run_trial(context)

    assert execution.status is TrialStatus.PASSED
    assert execution.verification.state is VerificationState.PASSED
    assert execution.metrics["usage_complete"] is False
    assert execution.metrics["main_agent_total_tokens"] is None
    assert execution.metrics["generation_settings"]["max_tokens"] == 1_500
    assert all(setting[0] == 1_500 for setting in provider.settings)
    assert "incomplete_main_agent_usage" in execution.findings
