"""Historical fixture Runtime kept separate from installed Memory backends."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmarks.codeflowbench.budget import (
    BudgetGuardedProvider,
    ProviderBudgetConfig,
    ProviderBudgetLedger,
    provider_call_budget_scope,
)
from benchmarks.codeflowbench.host import RecordingOutlet, RuntimeTrialHost
from benchmarks.codeflowbench.usage import RecordingProvider, UsageRecorder, usage_scope
from codeflow.config.codeflow import MemoryConfig, CodeFlowConfig
from codeflow.config.schema import Config
from codeflow.context_engine.assembler import ContextAssembler
from codeflow.context_engine.segments import (
    ActiveSkillsSegmentBuilder,
    BootstrapSegmentBuilder,
    IdentitySegmentBuilder,
    MemorySegmentBuilder,
    SkillsSegmentBuilder,
)
from codeflow.context_engine.segments.curator import CuratorSegmentBuilder
from codeflow.memory_engine import Memory
from codeflow.memory_engine.skill_forge import (
    LocalSkillSource,
    RouterHit,
    SkillForgeRouter,
)
from codeflow.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from codeflow.spine import ChatType, Origin, Source, TurnRequest
from codeflow.utils.helpers import estimate_prompt_tokens

from .models import CrossSessionTask

_ALWAYS_DISABLED_TOOLS = [
    "ask_user",
    "edit_file",
    "exec",
    "find",
    "grep",
    "list_dir",
    "message",
    "read_file",
    "spawn",
    "understand_media",
    "web_fetch",
    "web_search",
]


class PersistentFixtureBackend:
    def __init__(
        self,
        *,
        state_path: Path,
        task: CrossSessionTask,
        user_recall_enabled: bool,
        stage: str,
    ) -> None:
        adapter = _PersistentEverosAdapter(
            state_path=state_path,
            task=task,
            stage=stage,
        )
        self._fixture_adapter = adapter
        self.agent_id = task.workspace_id
        self._user_recall_enabled = user_recall_enabled
        self.user_recall_calls = 0
        self.suppressed_user_recall_calls = 0
        self.agent_recall_calls = 0
        self.store_calls = 0
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        top_k: int,
    ) -> list[Memory]:
        if (user_id is None) == (agent_id is None):
            return []
        if user_id is not None:
            if not self._user_recall_enabled:
                self.suppressed_user_recall_calls += 1
                return []
            self.user_recall_calls += 1
            data = await self._fixture_adapter.search(
                user_id=user_id,
                agent_id=None,
                query=query,
                top_k=top_k,
            )
            return [
                Memory(
                    text=item.episode,
                    score=float(item.score),
                    metadata={"id": item.id},
                )
                for item in data.episodes
            ]
        self.agent_recall_calls += 1
        data = await self._fixture_adapter.search(
            user_id=None,
            agent_id=agent_id,
            query=query,
            top_k=top_k,
        )
        return [
            Memory(
                text=item.content,
                score=float(item.score),
                metadata={"id": item.id, "name": item.name},
            )
            for item in data.agent_skills
        ]

    async def store(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        self.store_calls += 1
        await self._fixture_adapter.memorize(session_id, messages)

    async def feedback(self, signals: dict[str, Any]) -> None:
        del signals

    async def wait_idle(self) -> bool:
        return self._fixture_adapter.wait_idle()


class _HistoricalSkillSource:
    name = "everos"
    weight = 0.9

    def __init__(self, backend: PersistentFixtureBackend) -> None:
        self._backend = backend

    async def search(
        self,
        query: str,
        history: list[dict[str, Any]],
        k: int,
    ) -> list[RouterHit]:
        del history
        memories = await self._backend.recall(
            query,
            agent_id=self._backend.agent_id,
            top_k=k,
        )
        return [
            RouterHit(
                qualified_id=f"everos/{memory.metadata['id']}",
                name=str(memory.metadata["name"]),
                content=memory.text,
                score=memory.score,
                meta={"source": "everos"},
            )
            for memory in memories
        ]


class _PersistentEverosAdapter:
    def __init__(
        self,
        *,
        state_path: Path,
        task: CrossSessionTask,
        stage: str,
    ) -> None:
        self._state_path = state_path
        self._task = task
        self._stage = stage
        self._state = self._read_state()

    async def search(
        self,
        *,
        user_id: str | None,
        agent_id: str | None,
        query: str,
        top_k: int,
    ) -> SimpleNamespace:
        if user_id is not None:
            episodes = [
                SimpleNamespace(
                    id=str(item["id"]),
                    session_id=str(item["session_id"]),
                    episode=str(item["text"]),
                    summary="",
                    score=1.0,
                )
                for item in self._state["user_memories"]
                if item.get("workspace_id") == user_id
            ][:top_k]
            return _everos_search_data(episodes=episodes)
        query_words = set(query.lower().split())
        skills = [
            SimpleNamespace(
                id=str(item["id"]),
                name=str(item["name"]),
                content=str(item["text"]),
                confidence=1.0,
                score=1.0,
            )
            for item in self._state["agent_skills"]
            if item.get("workspace_id") == agent_id
            and query_words & set(str(item.get("retrieval_terms", "")).lower().split())
        ][:top_k]
        return _everos_search_data(agent_skills=skills)

    async def memorize(
        self,
        session_id: str,
        payload_messages: list[dict[str, Any]],
        *,
        is_final: bool = False,
    ) -> None:
        if self._stage != "learning" or not payload_messages:
            return
        self._state = {
            "user_memories": [
                {
                    "id": f"{self._task.task_id}-user-fact",
                    "session_id": session_id,
                    "workspace_id": self._task.workspace_id,
                    "text": self._task.learned_fact,
                }
            ],
            "agent_skills": [
                {
                    "id": f"{self._task.task_id}-agent-skill",
                    "name": self._task.required_skill,
                    "workspace_id": self._task.workspace_id,
                    "text": (f"SKILL_EVIDENCE:{self._task.required_skill} deployment checklist procedure"),
                    "retrieval_terms": "deployment checklist procedure apply",
                }
            ],
        }
        self._write_state()

    def wait_idle(self) -> bool:
        return self._state_path.exists()

    def _read_state(self) -> dict[str, list[dict[str, Any]]]:
        if not self._state_path.exists():
            return {
                "user_memories": [],
                "agent_skills": [],
            }
        payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("memory state must be an object")
        return {
            "user_memories": list(payload.get("user_memories", [])),
            "agent_skills": list(payload.get("agent_skills", [])),
        }

    def _write_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self._state,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        with tempfile.NamedTemporaryFile(
            dir=self._state_path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, self._state_path)


def _everos_search_data(
    *,
    episodes: list[SimpleNamespace] | None = None,
    agent_skills: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        episodes=episodes or [],
        profiles=[],
        agent_cases=[],
        agent_skills=agent_skills or [],
    )


class ScriptedCrossSessionProvider(LLMProvider):
    def __init__(
        self,
        *,
        task: CrossSessionTask,
        stage: str,
    ) -> None:
        super().__init__(api_key="codeflowbench")
        self._task = task
        self._stage = stage
        self._calls = 0
        self.memory_observed = False
        self.skill_observed = False

    def get_default_model(self) -> str:
        return "scripted/codeflowbench-memory-skill"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        del max_tokens, temperature, reasoning_effort, tool_choice
        self._calls += 1
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        self.memory_observed = self._task.learned_fact in prompt
        self.skill_observed = f"SKILL_EVIDENCE:{self._task.required_skill}" in prompt
        usage = {
            "prompt_tokens": estimate_prompt_tokens(messages, tools),
            "completion_tokens": 12,
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        if self._stage == "learning":
            return LLMResponse(
                content="Learning session recorded.",
                finish_reason="stop",
                usage=usage,
            )
        if self._calls == 1:
            artifact = {
                "task_id": self._task.task_id,
                "region": (self._task.expected_value if self.memory_observed else None),
                "skill_applied": self.skill_observed,
            }
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id=f"{self._task.task_id}-write-result",
                        name="write_file",
                        arguments={
                            "path": "result.json",
                            "content": json.dumps(
                                artifact,
                                sort_keys=True,
                            ),
                        },
                    )
                ],
                finish_reason="tool_calls",
                usage=usage,
            )
        return LLMResponse(
            content="Evaluation artifact written.",
            finish_reason="stop",
            usage=usage,
        )


def memory_skill_context_engine_factory(
    *,
    skill_sources: tuple[str, ...],
):
    def build(**kwargs: Any) -> ContextAssembler:
        builder = kwargs["builder"]
        backend = kwargs["backend"]
        memory_config = kwargs.get("memory_config") or MemoryConfig()
        router_config = kwargs["skill_forge_router_config"]
        sources = []
        if "local" in skill_sources:
            local = LocalSkillSource(
                pool=builder.skills.pool,
                registry=builder.skills.registry,
            )
            sources.append(local)
        if "everos" in skill_sources:
            everos = _HistoricalSkillSource(backend)
            sources.append(everos)
        router = SkillForgeRouter(sources=sources)
        builders = [
            IdentitySegmentBuilder(kwargs["workspace"]),
            BootstrapSegmentBuilder(kwargs["workspace"]),
            MemorySegmentBuilder(
                builder.memory,
                backend,
                user_id=memory_config.user_id,
                memory_top_k=memory_config.memory_top_k,
                enabled=True,
            ),
            ActiveSkillsSegmentBuilder(builder.skills),
            SkillsSegmentBuilder(
                router,
                skill_top_k=router_config.top_k,
            ),
            CuratorSegmentBuilder(
                workspace=kwargs["workspace"],
                config=kwargs["config"],
                provider=kwargs["provider"],
                model=kwargs["model"],
                context_window_tokens=kwargs["context_window_tokens"],
                get_tool_definitions=kwargs["get_tool_definitions"],
                now_fn=kwargs.get("now_fn"),
                memory_enabled=True,
            ),
        ]
        return ContextAssembler(
            builders,
            kwargs["get_tool_definitions"],
            now_fn=kwargs.get("now_fn"),
        )

    return build


async def run_runtime_stage(
    *,
    stage: str,
    task: CrossSessionTask,
    variant_settings: dict[str, Any],
    workspace: Path,
    state_path: Path,
    provider_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from unittest.mock import patch

    provider_spec = provider_spec or {
        "mode": "scripted",
        "provider_identity": "scripted/codeflowbench-memory-skill",
        "paid_campaign_eligible": False,
        "real_agent_task_effect_claim_eligible": False,
    }
    user_recall_enabled = variant_settings["user_memory_recall"] == "enabled"
    skill_sources = tuple(variant_settings["skill_sources"])
    backend = PersistentFixtureBackend(
        state_path=state_path,
        task=task,
        user_recall_enabled=user_recall_enabled,
        stage=stage,
    )
    (
        config,
        codeflow_config,
        delegate,
        budget_ledger,
        provider_identity,
    ) = _runtime_dependencies(
        provider_spec=provider_spec,
        workspace=workspace,
        task=task,
        stage=stage,
    )
    recorder = UsageRecorder()
    provider = RecordingProvider(delegate, recorder=recorder)
    outlet = RecordingOutlet("codeflowbench-memory-skill")
    conversation = f"{task.task_id}:{stage}"
    request = TurnRequest(
        origin=Origin.USER,
        source=Source(
            channel="codeflowbench-memory-skill",
            chat_id=conversation,
            sender_id=task.workspace_id,
            chat_type=ChatType.DM,
        ),
        text=(
            (f"LEARN_PHASE: remember {task.learned_fact}; retain deployment checklist skill {task.required_skill}.")
            if stage == "learning"
            else task.evaluation_request
        ),
        message_id=f"{conversation}-message",
        conversation=conversation,
    )
    factory = memory_skill_context_engine_factory(
        skill_sources=skill_sources,
    )
    budget_before = budget_ledger.snapshot() if budget_ledger else None
    if provider_spec["mode"] == "real":
        stage_caps = dict(provider_spec["stage_logical_call_caps"])
        budget_scope = provider_call_budget_scope(
            trial_id=f"{provider_spec['trial_id']}/{stage}",
            max_logical_calls=int(stage_caps[stage]),
            max_attempts_per_call=int(provider_spec["max_attempts_per_call"]),
            max_input_tokens_per_call=int(
                provider_spec["max_input_tokens_per_call"],
            ),
            max_output_tokens_per_call=int(
                provider_spec["max_output_tokens_per_call"],
            ),
        )
    else:
        budget_scope = contextlib.nullcontext()
    with (
        patch(
            "codeflow.cli._plugin_stack.maybe_build_memory_backend",
            return_value=backend,
        ),
        patch(
            "codeflow.cli._plugin_stack.build_plugin_tools",
            return_value=[],
        ),
    ):
        with budget_scope:
            host = await RuntimeTrialHost.build(
                config=config,
                codeflow_config=codeflow_config,
                provider=provider,
                cron_service=None,
                outlet=outlet,
                context_engine_factory=factory,
            )
            try:
                with usage_scope(call_role="main_agent"):
                    observation = await host.run(request)
                quiescent = await backend.wait_idle()
                session = host.assembly.session_manager.peek(conversation)
            finally:
                await host.close()
    aggregate = recorder.aggregate()
    budget_after = budget_ledger.snapshot() if budget_ledger else None
    scripted = delegate if isinstance(delegate, ScriptedCrossSessionProvider) else None
    failure_category = observation.failure_category
    if recorder.has_error_category("task_budget_exhausted"):
        failure_category = "task_budget_exhausted"
    return {
        "pid": os.getpid(),
        "stage": stage,
        "conversation": conversation,
        "session_message_count": (len(session.messages) if session is not None else 0),
        "runtime_state": observation.runtime_state.value,
        "delivery_state": observation.delivery_state.value,
        "failure_category": failure_category,
        "memory_hits": (observation.outcome.memory_hits if observation.outcome is not None else 0),
        "injected_skill_ids": (list(observation.outcome.injected_skill_ids) if observation.outcome is not None else []),
        "user_recall_calls": backend.user_recall_calls,
        "suppressed_user_recall_calls": (backend.suppressed_user_recall_calls),
        "agent_recall_calls": backend.agent_recall_calls,
        "store_calls": backend.store_calls,
        "backend_quiescent": quiescent,
        "backend_class": "EverosBackend",
        "backend_adapter": "injected_fixture",
        "everos_semantic_quality_claim_eligible": False,
        "provider_memory_observed": (scripted.memory_observed if scripted is not None else None),
        "provider_skill_observed": (scripted.skill_observed if scripted is not None else None),
        "provider_identity": provider_identity,
        "paid_campaign_eligible": bool(provider_spec["paid_campaign_eligible"]),
        "real_agent_task_effect_claim_eligible": bool(provider_spec["real_agent_task_effect_claim_eligible"]),
        "cost_complete": (budget_after.accounting_complete if budget_after is not None else True),
        "provider_charged_cny": (
            budget_after.provider_charged_cny - budget_before.provider_charged_cny
            if budget_after is not None and budget_before is not None
            else 0.0
        ),
        "usage": {
            "calls": aggregate.calls,
            "input_tokens": aggregate.input_tokens,
            "output_tokens": aggregate.output_tokens,
            "total_tokens": aggregate.total_tokens,
            "usage_complete": aggregate.usage_complete,
        },
    }


def _runtime_dependencies(
    *,
    provider_spec: dict[str, Any],
    workspace: Path,
    task: CrossSessionTask,
    stage: str,
) -> tuple[
    Config,
    CodeFlowConfig,
    LLMProvider,
    ProviderBudgetLedger | None,
    str,
]:
    if provider_spec["mode"] == "scripted":
        delegate = ScriptedCrossSessionProvider(task=task, stage=stage)
        config = _freeze_runtime_config(
            Config(),
            workspace=workspace,
            model=delegate.get_default_model(),
            stage=stage,
        )
        return (
            config,
            _freeze_codeflow_config(CodeFlowConfig(base=config), config, task),
            delegate,
            None,
            delegate.get_default_model(),
        )
    if provider_spec["mode"] != "real":
        raise ValueError(f"unknown provider mode: {provider_spec['mode']}")
    from codeflow.cli._helpers import make_provider

    private_config_path = Path(str(provider_spec["private_config_path"]))
    payload = json.loads(private_config_path.read_text(encoding="utf-8"))
    config = Config.model_validate(payload["config"])
    model = str(provider_spec["provider_identity"])
    config = _freeze_runtime_config(
        config,
        workspace=workspace,
        model=model,
        stage=stage,
    )
    codeflow_config = CodeFlowConfig.model_validate(payload["codeflow_config"])
    codeflow_config = _freeze_codeflow_config(codeflow_config, config, task)
    ledger = ProviderBudgetLedger(
        Path(str(provider_spec["ledger_path"])),
        ProviderBudgetConfig(**dict(provider_spec["ledger_config"])),
    )
    delegate = BudgetGuardedProvider(
        make_provider(config),
        ledger=ledger,
    )
    return config, codeflow_config, delegate, ledger, model


def _freeze_runtime_config(
    source: Config,
    *,
    workspace: Path,
    model: str,
    stage: str,
) -> Config:
    if stage not in {"learning", "evaluation"}:
        raise ValueError(f"unknown memory/skill stage: {stage}")
    config = Config.model_validate(source.model_dump(mode="json"))
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = model
    config.agents.defaults.max_tokens = 1_500
    config.agents.defaults.max_tool_iterations = 3
    config.agents.defaults.enable_personalization = False
    config.routing.enabled = False
    config.tools.restrict_to_workspace = True
    disabled_tools = list(_ALWAYS_DISABLED_TOOLS)
    if stage == "learning":
        disabled_tools.append("write_file")
    config.tools.disabled_tools = disabled_tools
    config.tools.mcp_servers = {}
    config.tools.tool_search.enabled = False
    return config


def _freeze_codeflow_config(
    codeflow_config: CodeFlowConfig,
    config: Config,
    task: CrossSessionTask,
) -> CodeFlowConfig:
    codeflow_config.base = config
    codeflow_config.memory.backend = None
    codeflow_config.memory.user_id = task.workspace_id
    codeflow_config.memory.memory_top_k = 5
    codeflow_config.skill_forge.rewrite_enabled = False
    codeflow_config.skill_forge.llm_gate_enabled = False
    codeflow_config.token_wise.smart_routing.enabled = False
    codeflow_config.runtime.checkpoint.policy = "never"
    return codeflow_config
