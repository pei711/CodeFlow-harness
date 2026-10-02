"""CLI tests for ``codeflow gateway``.

The ``gateway`` command spawns the agent loop, channel manager, cron service,
and optional background services, then runs forever. Coverage includes startup
smoke checks, configuration assembly, and deterministic shutdown ordering.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from codeflow.cli.commands import app
from codeflow.config.loader import set_config_path

runner = CliRunner()


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.json"
    set_config_path(cfg)
    yield cfg
    set_config_path(None)  # type: ignore[arg-type]


_CLEANUP_ORDER = [
    "health_close",
    "cron_stop",
    "question_cancel",
    "intake_quiesce",
    "spine_teardown",
    "channel_stop_started",
    "channel_stop_finished",
    "agent_stop",
    "runtime_close",
]


def _gateway_cleanup_dependencies(*, failures=()):
    events: list[str] = []
    errors = {name: RuntimeError(f"{name} failed") for name in failures}

    def run_step(name: str) -> None:
        events.append(name)
        if error := errors.get(name):
            raise error

    async def quiesce_intake() -> None:
        run_step("intake_quiesce")

    async def stop_channels() -> None:
        events.append("channel_stop_started")
        events.append("channel_stop_finished")
        if error := errors.get("channel_stop"):
            raise error

    async def teardown_spine() -> None:
        run_step("spine_teardown")

    async def close_runtime() -> None:
        run_step("runtime_close")

    dependencies = {
        "health_server": SimpleNamespace(close=lambda: run_step("health_close")),
        "cron": SimpleNamespace(stop=lambda: run_step("cron_stop")),
        "question_broker": SimpleNamespace(cancel_all=lambda: run_step("question_cancel")),
        "channels": SimpleNamespace(quiesce_intake=quiesce_intake, stop_all=stop_channels),
        "gw_teardown": teardown_spine,
        "agent": SimpleNamespace(stop=lambda: run_step("agent_stop")),
        "runtime": SimpleNamespace(close=close_runtime),
    }
    return dependencies, events, errors


def test_gateway_help_works() -> None:
    """``codeflow gateway --help`` lists the documented options."""
    r = runner.invoke(app, ["gateway", "--help"])
    assert r.exit_code == 0
    assert "Start the CodeFlow gateway" in r.stdout
    assert "--port" in r.stdout
    assert "--workspace" in r.stdout
    assert "--verbose" in r.stdout
    assert "--config" in r.stdout


def test_gateway_config_short_alias_removed() -> None:
    """``-c`` no longer binds ``--config`` (UN-41); only the long form remains."""
    bad = runner.invoke(app, ["gateway", "-c", "/tmp/whatever.json"])
    assert bad.exit_code != 0

    r = runner.invoke(app, ["gateway", "--help"])
    assert r.exit_code == 0
    assert "--config" in r.stdout


def test_gateway_without_api_key_exits_with_error(tmp_config: Path) -> None:
    """With no provider configured, gateway must exit non-zero — and crucially
    must not raise a crash-class exception (NameError / AttributeError /
    ImportError). Those would indicate a regression like a missing import.
    """
    from codeflow.config.loader import save_config
    from codeflow.config.schema import Config

    save_config(Config())

    r = runner.invoke(app, ["gateway"])
    if r.exception is not None:
        assert not isinstance(r.exception, (NameError, AttributeError, ImportError)), (
            f"Crash-class exception leaked through: {r.exception!r}"
        )
    assert r.exit_code != 0


@pytest.mark.parametrize("channel_validation_fails", [False, True])
def test_gateway_validates_channels_before_shared_runtime_assembly(
    tmp_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    channel_validation_fails: bool,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from codeflow.cli import _runtime_assembly, gateway_commands
    from codeflow.config.paths import RuntimePaths

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = MagicMock()
    config.workspace_path = workspace
    config.gateway.port = 18_790
    config.gateway.log = SimpleNamespace(
        rotation="10 MB",
        retention=7,
        level="INFO",
        console_level="INFO",
    )
    for name in ("feishu", "qq", "wecom"):
        getattr(config.channels, name).enabled = False
    codeflow_config = MagicMock()
    provider = object()
    router = object()
    cron_service = MagicMock()
    cron_service.status.return_value = {"jobs": 0}
    runtime = SimpleNamespace(
        agent_loop=MagicMock(),
        session_manager=object(),
    )
    calls: list[tuple[tuple, dict]] = []

    monkeypatch.setattr(
        gateway_commands,
        "load_runtime_config",
        lambda config_path, workspace_path: config,
    )
    monkeypatch.setattr(gateway_commands, "make_provider", lambda cfg: provider)
    monkeypatch.setattr(
        gateway_commands,
        "print_deprecated_memory_window_notice",
        lambda cfg: None,
    )
    monkeypatch.setattr(
        gateway_commands,
        "sync_workspace_templates",
        lambda workspace_path: None,
    )
    monkeypatch.setattr(
        "codeflow.cli._log_file.redirect_loguru_to_file",
        lambda *args, **kwargs: tmp_path / "gateway.log",
    )
    monkeypatch.setattr(
        "codeflow.cli._gateway_lock.acquire",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "codeflow.config.codeflow.load_codeflow_config",
        lambda: codeflow_config,
    )
    monkeypatch.setattr(
        "codeflow.config.paths.get_cron_dir",
        lambda: tmp_path / "cron",
    )
    monkeypatch.setattr(
        "codeflow.proactive_engine.schedulers.cron.service.CronService",
        lambda *args, **kwargs: cron_service,
    )
    monkeypatch.setattr(
        gateway_commands,
        "build_model_routing",
        lambda cfg, initial_provider: (router, initial_provider),
    )

    def _assemble(*args, **kwargs):
        calls.append((args, kwargs))
        return runtime

    monkeypatch.setattr(_runtime_assembly, "assemble_runtime", _assemble)

    class _ChannelValidationFailure(RuntimeError):
        pass

    class _StopAfterAssembly(RuntimeError):
        pass

    channels = SimpleNamespace(enabled_channels=[], channels={})

    def _build_channels(cfg):
        if channel_validation_fails:
            raise _ChannelValidationFailure
        return channels

    monkeypatch.setattr(
        "codeflow.channels.manager.ChannelManager",
        _build_channels,
    )

    def _stop_async_run(coro):
        coro.close()
        raise _StopAfterAssembly

    monkeypatch.setattr(gateway_commands.asyncio, "run", _stop_async_run)

    result = runner.invoke(
        app,
        ["gateway", "--config", str(tmp_config)],
    )

    expected_exception = _ChannelValidationFailure if channel_validation_fails else _StopAfterAssembly
    assert isinstance(result.exception, expected_exception)
    assert len(calls) == (0 if channel_validation_fails else 1)
    if channel_validation_fails:
        return
    assert calls[0][0] == (config, codeflow_config)
    assert calls[0][1] == {
        "provider": provider,
        "cron_service": cron_service,
        "router": router,
        "interactive": True,
        "paths": RuntimePaths(
            workspace=config.workspace_path,
            state=config.workspace_path,
        ),
    }


def test_gateway_refuses_second_instance(tmp_config: Path, monkeypatch) -> None:
    """When the instance lock is already held, gateway exits 1 with a clear
    message and never builds the agent/channel stack."""
    from codeflow.config.loader import save_config
    from codeflow.config.schema import Config

    save_config(Config())

    from codeflow.cli import _gateway_lock

    def _raise(now: float):
        raise _gateway_lock.GatewayAlreadyRunningError(
            _gateway_lock.LockInfo(pid=4242, started_at=0.0, config_path=str(tmp_config))
        )

    monkeypatch.setattr(_gateway_lock, "acquire", _raise)

    r = runner.invoke(app, ["gateway"])
    assert r.exit_code == 1
    assert "already running for this instance" in r.stdout
    assert "4242" in r.stdout


async def test_gateway_quiesces_intake_before_spine_and_stops_transports_afterward() -> None:
    from codeflow.cli.gateway_commands import _cleanup_gateway

    dependencies, events, _errors = _gateway_cleanup_dependencies()

    await _cleanup_gateway(run_error=None, **dependencies)

    assert events == _CLEANUP_ORDER


async def test_gateway_attempts_every_cleanup_and_raises_the_first_failure() -> None:
    from codeflow.cli.gateway_commands import _cleanup_gateway

    failures = {
        "health_close",
        "cron_stop",
        "question_cancel",
        "intake_quiesce",
        "channel_stop",
        "spine_teardown",
        "agent_stop",
        "runtime_close",
    }
    dependencies, events, errors = _gateway_cleanup_dependencies(failures=failures)

    with pytest.raises(RuntimeError) as exc_info:
        await _cleanup_gateway(run_error=None, **dependencies)

    assert exc_info.value is errors["health_close"]
    assert events == _CLEANUP_ORDER


@pytest.mark.parametrize(
    "run_error",
    [RuntimeError("gateway run failed"), asyncio.CancelledError()],
)
async def test_gateway_preserves_run_failure_over_cleanup_failures(
    run_error: BaseException,
) -> None:
    from codeflow.cli.gateway_commands import _cleanup_gateway

    dependencies, events, _errors = _gateway_cleanup_dependencies(
        failures={"health_close", "cron_stop", "runtime_close"}
    )

    with pytest.raises(BaseException) as exc_info:
        await _cleanup_gateway(run_error=run_error, **dependencies)

    assert exc_info.value is run_error
    assert events == _CLEANUP_ORDER


async def test_gateway_preserves_run_failure_over_cleanup_cancellation() -> None:
    from codeflow.cli.gateway_commands import _cleanup_gateway

    dependencies, events, _errors = _gateway_cleanup_dependencies()
    run_error = RuntimeError("gateway run failed")

    async def cancelled_quiesce() -> None:
        events.append("intake_quiesce")
        raise asyncio.CancelledError

    dependencies["channels"].quiesce_intake = cancelled_quiesce

    with pytest.raises(RuntimeError) as exc_info:
        await _cleanup_gateway(run_error=run_error, **dependencies)

    assert exc_info.value is run_error
    assert events == _CLEANUP_ORDER


async def test_gateway_cancellation_at_intake_barrier_finishes_barrier_before_spine() -> None:
    from codeflow.channels.intake import Intake
    from codeflow.channels.manager import ChannelManager
    from codeflow.cli.gateway_commands import _cleanup_gateway

    events: list[str] = []
    barrier_started = asyncio.Event()
    publish_cancelled = asyncio.Event()
    release_publish = asyncio.Event()
    intake = Intake("telegram", SimpleNamespace(allow_from=["*"]))

    async def submit(req) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            publish_cancelled.set()
            await release_publish.wait()
            events.append("publish_finished")

    intake.set_submit(submit)
    original_wait_idle = intake.wait_idle

    async def observed_wait_idle() -> None:
        barrier_started.set()
        await original_wait_idle()

    intake.wait_idle = observed_wait_idle  # type: ignore[method-assign]

    class _Channel:
        async def stop(self) -> None:
            events.append("transport_stop")

    channel = _Channel()
    channel.intake = intake
    channels = object.__new__(ChannelManager)
    channels.channels = {"telegram": channel}
    health_error = RuntimeError("health close failed")

    def close_health() -> None:
        events.append("health_close")
        raise health_error

    publish = asyncio.create_task(intake.publish("user", "c", "blocked"))
    await asyncio.sleep(0)
    cleanup = asyncio.create_task(
        _cleanup_gateway(
            run_error=None,
            health_server=SimpleNamespace(close=close_health),
            cron=SimpleNamespace(stop=lambda: events.append("cron_stop")),
            question_broker=None,
            channels=channels,
            gw_teardown=lambda: events.append("spine_teardown"),
            agent=SimpleNamespace(stop=lambda: events.append("agent_stop")),
            runtime=SimpleNamespace(close=lambda: events.append("runtime_close")),
        )
    )
    await barrier_started.wait()
    cleanup.cancel()
    await publish_cancelled.wait()
    cleanup.cancel()
    await asyncio.sleep(0)
    assert "spine_teardown" not in events
    release_publish.set()

    try:
        with pytest.raises(asyncio.CancelledError):
            await cleanup
    finally:
        release_publish.set()
        if not publish.done():
            publish.cancel()
        await asyncio.gather(publish, return_exceptions=True)

    assert events.index("publish_finished") < events.index("spine_teardown")
    assert events[-4:] == ["spine_teardown", "transport_stop", "agent_stop", "runtime_close"]


def test_gateway_log_config_defaults() -> None:
    from codeflow.config.schema import GatewayConfig

    cfg = GatewayConfig()
    log = cfg.log
    assert not hasattr(cfg, "heartbeat")
    assert log.rotation == "10 MB"
    assert log.retention == 7
    assert log.level == "INFO"
    assert log.console_level == "INFO"


def test_gateway_log_config_overrides_parse() -> None:
    from codeflow.config.schema import GatewayConfig

    cfg = GatewayConfig.model_validate(
        {
            "log": {
                "rotation": "00:00",
                "retention": "14 days",
                "level": "DEBUG",
                "console_level": "WARNING",
            }
        }
    )
    assert cfg.log.rotation == "00:00"
    assert cfg.log.retention == "14 days"
    assert cfg.log.level == "DEBUG"
    assert cfg.log.console_level == "WARNING"


def test_gateway_channels_excludes_tui_when_no_im_enabled() -> None:

    from unittest.mock import MagicMock

    from codeflow.cli.gateway_commands import _build_gateway_channels

    cfg = MagicMock()
    for name in ("feishu", "qq", "wecom"):
        ch = MagicMock()
        ch.enabled = False
        setattr(cfg.channels, name, ch)
    assert _build_gateway_channels(cfg) == set()


def test_gateway_channels_excludes_tui_alongside_enabled_im() -> None:
    from unittest.mock import MagicMock

    from codeflow.cli.gateway_commands import _build_gateway_channels

    cfg = MagicMock()
    for name in ("feishu", "qq", "wecom"):
        ch = MagicMock()
        ch.enabled = name == "feishu"
        setattr(cfg.channels, name, ch)
    result = _build_gateway_channels(cfg)
    assert "tui" not in result
    assert "feishu" in result
    assert "qq" not in result


def test_stop_dispatch_cancels_both_scheduler_and_subagents() -> None:
    """The gateway ``/stop`` path must fan out to BOTH the scheduler lane cancel
    and the subagent-session cancel, summing their counts.

    ``_inbound_dispatch`` is a closure nested inside the gateway serve command
    with no import seam, so this pins the /stop branch of the command source:
    dropping either cancel call (or the summed count) breaks this test.
    """
    import inspect

    from codeflow.cli import gateway_commands

    src = inspect.getsource(gateway_commands.register)
    stop_branch = src.split('if cmd == "/stop":', 1)[1].split('elif cmd == "/restart":', 1)[0]
    assert "cancel_conversation(cid)" in stop_branch
    assert "cancel_by_session(cid)" in stop_branch
    assert "stopped +=" in stop_branch


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

from types import SimpleNamespace

from codeflow.cli.gateway_commands import build_model_routing
from codeflow.config.schema import ModelEndpoint, RoutingConfig
from codeflow.providers.per_model_provider import PerModelProvider
from codeflow.routing.knn_router import KNNModelRouter
from codeflow.routing.router import ModelRouter


class _FakeProvider:
    def get_default_model(self):
        return "default-model"


def _routing_config_obj(routing):
    return SimpleNamespace(
        routing=routing,
        providers=SimpleNamespace(openrouter=SimpleNamespace(api_key="")),
        agents=SimpleNamespace(defaults=SimpleNamespace(model="default-model")),
    )


def test_build_routing_disabled_returns_same_provider():
    prov = _FakeProvider()
    router, out = build_model_routing(_routing_config_obj(RoutingConfig(enabled=False)), prov)
    assert router is None
    assert out is prov


def test_build_routing_knn_wraps_provider():
    routing = RoutingConfig(
        enabled=True,
        backend="knn",
        embedding_endpoint="http://e/embed",
        models=[
            ModelEndpoint(model="small", api_base="http://a/v1"),
            ModelEndpoint(model="large", api_base="http://b/v1"),
        ],
    )
    router, out = build_model_routing(_routing_config_obj(routing), _FakeProvider())
    assert isinstance(router, KNNModelRouter)
    assert isinstance(out, PerModelProvider)


def test_build_routing_ecoclaw_with_key_keeps_provider():
    routing = RoutingConfig(enabled=True, backend="ecoclaw", api_key="sk-or-x")
    prov = _FakeProvider()
    router, out = build_model_routing(_routing_config_obj(routing), prov)
    assert isinstance(router, ModelRouter)
    assert out is prov


def test_build_routing_ecoclaw_no_key_disabled():
    routing = RoutingConfig(enabled=True, backend="ecoclaw", api_key="")
    prov = _FakeProvider()
    router, out = build_model_routing(_routing_config_obj(routing), prov)
    assert router is None
    assert out is prov
