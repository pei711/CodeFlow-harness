"""Tests for codeflow.channels.manager.ChannelManager — spec-based init
(incl. the missing-dependency / ImportError path), per-channel failure
isolation, allow_from validation, and status accessors. Outbound delivery moved
to the spine outlets (no longer the manager's job)."""

import asyncio
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest
from loguru import logger

from codeflow.channels.contract import Capabilities, ChannelSpec
from codeflow.channels.manager import ChannelManager, _missing_dep_hint


@contextmanager
def _logs(level="ERROR"):
    lines: list[str] = []
    sink_id = logger.add(lambda m: lines.append(str(m)), level=level)
    try:
        yield lines
    finally:
        logger.remove(sink_id)


class _FakeChannel:
    def __init__(self, config):
        self.config = config
        self._running = False
        self.transcription_api_key = ""

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:  # pragma: no cover - 未覆盖
        self._running = True

    async def stop(self) -> None:  # pragma: no cover - 未覆盖
        self._running = False

    async def send(self, chat_id, content, media=None) -> None:  # pragma: no cover
        pass


def _spec(factory, display_name="Fake", interactive_login=False) -> ChannelSpec:
    return ChannelSpec(
        display_name=display_name,
        factory=factory,
        capabilities=Capabilities(interactive_login=interactive_login),
    )


def _config(channels=None):
    chan = SimpleNamespace()
    for name, section in (channels or {}).items():
        setattr(chan, name, section)
    return SimpleNamespace(
        providers=SimpleNamespace(groq=SimpleNamespace(api_key="gk")),
        channels=chan,
    )


def _manager(monkeypatch, specs, config) -> ChannelManager:
    monkeypatch.setattr("codeflow.channels.registry.discover_specs", lambda: specs)
    return ChannelManager(config)


def test_init_builds_enabled_channel_and_sets_groq_key(monkeypatch):
    mgr = _manager(
        monkeypatch,
        {"fake": _spec(_FakeChannel)},
        _config({"fake": SimpleNamespace(enabled=True, allow_from=["*"])}),
    )
    assert mgr.enabled_channels == ["fake"]
    assert mgr.channels["fake"].transcription_api_key == "gk"


def test_init_skips_disabled_channel(monkeypatch):
    mgr = _manager(
        monkeypatch,
        {"fake": _spec(_FakeChannel)},
        _config({"fake": SimpleNamespace(enabled=False, allow_from=["*"])}),
    )
    assert mgr.channels == {}


def test_init_disables_channel_on_missing_dependency(monkeypatch):
    """A channel whose factory can't import its SDK is disabled, not fatal."""

    def boom(config):
        raise ImportError("No module named 'botpy'")

    mgr = _manager(
        monkeypatch,
        {"fake": _spec(boom)},
        _config({"fake": SimpleNamespace(enabled=True, allow_from=["*"])}),
    )
    assert "fake" not in mgr.channels


def test_empty_allow_from_disables_the_channel_loudly(monkeypatch):
    """A deny-all allowlist is a misconfiguration: drop that channel with a
    loud error instead of aborting the process."""
    with _logs() as lines:
        mgr = _manager(
            monkeypatch,
            {"fake": _spec(_FakeChannel)},
            _config({"fake": SimpleNamespace(enabled=True, allow_from=[])}),
        )
    assert mgr.channels == {}
    error = "".join(lines)
    assert "fake channel disabled" in error
    assert "empty allowFrom denies every sender" in error


def test_factory_crash_disables_only_that_channel(monkeypatch):
    """A factory raising something other than ImportError must not escape and
    kill the Gateway; the other channel still constructs."""

    def boom(config):
        raise RuntimeError("SDK handshake exploded")

    with _logs() as lines:
        mgr = _manager(
            monkeypatch,
            {"broken": _spec(boom), "healthy": _spec(_FakeChannel)},
            _config(
                {
                    "broken": SimpleNamespace(enabled=True, allow_from=["*"]),
                    "healthy": SimpleNamespace(enabled=True, allow_from=["*"]),
                }
            ),
        )
    assert mgr.enabled_channels == ["healthy"]
    error = "".join(lines)
    assert "broken channel disabled" in error
    assert "RuntimeError: SDK handshake exploded" in error


def test_empty_allow_from_leaves_other_channels_running(monkeypatch):
    mgr = _manager(
        monkeypatch,
        {"denyall": _spec(_FakeChannel), "healthy": _spec(_FakeChannel)},
        _config(
            {
                "denyall": SimpleNamespace(enabled=True, allow_from=[]),
                "healthy": SimpleNamespace(enabled=True, allow_from=["*"]),
            }
        ),
    )
    assert mgr.enabled_channels == ["healthy"]


def test_gateway_still_constructs_when_every_channel_fails(monkeypatch):
    """Both failure modes at once still yield a usable manager for the Gateway
    to hold; no channel means no inbound, not a dead process."""

    def boom(config):
        raise RuntimeError("nope")

    mgr = _manager(
        monkeypatch,
        {"broken": _spec(boom), "denyall": _spec(_FakeChannel)},
        _config(
            {
                "broken": SimpleNamespace(enabled=True, allow_from=["*"]),
                "denyall": SimpleNamespace(enabled=True, allow_from=[]),
            }
        ),
    )
    assert mgr.enabled_channels == []
    assert mgr.get_status() == {}
    assert mgr.get_channel("broken") is None


@pytest.mark.parametrize(
    "first_error",
    [RuntimeError("first stop failed"), asyncio.CancelledError()],
    ids=["exception", "cancelled"],
)
async def test_stop_all_attempts_every_channel_and_raises_first_failure(
    first_error: BaseException,
) -> None:
    events: list[str] = []
    later_error = RuntimeError("later stop failed")

    class _Channel:
        def __init__(self, name: str, error: BaseException | None = None) -> None:
            self.name = name
            self.error = error

        async def stop(self) -> None:
            events.append(self.name)
            if self.error is not None:
                raise self.error

    manager = object.__new__(ChannelManager)
    manager.channels = {
        "first": _Channel("first", first_error),
        "second": _Channel("second", later_error),
        "last": _Channel("last"),
    }

    with _logs() as lines:
        with pytest.raises(BaseException) as exc_info:
            await manager.stop_all()

    assert exc_info.value is first_error
    assert events == ["first", "second", "last"]
    logged = "".join(lines)
    assert "Error stopping first" in logged
    assert "Error stopping second" in logged


async def test_stop_all_times_out_one_transport_and_attempts_the_rest(monkeypatch) -> None:
    monkeypatch.setattr("codeflow.channels.manager._CHANNEL_STOP_TIMEOUT_S", 0.01, raising=False)
    events: list[str] = []

    class _Channel:
        def __init__(self, name: str, *, blocks: bool = False) -> None:
            self.name = name
            self.blocks = blocks

        async def stop(self) -> None:
            events.append(self.name)
            if self.blocks:
                await asyncio.Event().wait()

    manager = object.__new__(ChannelManager)
    manager.channels = {
        "blocked": _Channel("blocked", blocks=True),
        "healthy": _Channel("healthy"),
    }

    with pytest.raises(TimeoutError, match="blocked.*timed out"):
        await asyncio.wait_for(manager.stop_all(), timeout=0.2)

    assert events == ["blocked", "healthy"]


async def test_stop_all_preserves_later_cancellation_over_earlier_failure() -> None:
    events: list[str] = []
    second_started = asyncio.Event()

    class _Channel:
        def __init__(self, name: str) -> None:
            self.name = name

        async def stop(self) -> None:
            events.append(self.name)
            if self.name == "first":
                raise RuntimeError("first stop failed")
            if self.name == "second":
                second_started.set()
                await asyncio.Event().wait()

    manager = object.__new__(ChannelManager)
    manager.channels = {name: _Channel(name) for name in ("first", "second", "third")}
    stopping = asyncio.create_task(manager.stop_all())
    await second_started.wait()
    stopping.cancel()

    with pytest.raises(asyncio.CancelledError):
        await stopping

    assert stopping.cancelled()
    assert events == ["first", "second", "third"]


async def test_quiesce_intake_seals_every_channel_before_waiting() -> None:
    events: list[str] = []
    intakes = []

    class _Intake:
        def __init__(self, name: str) -> None:
            self.name = name
            self.sealed = False
            intakes.append(self)

        def seal(self) -> None:
            self.sealed = True
            events.append(f"seal:{self.name}")

        async def wait_idle(self) -> None:
            assert all(intake.sealed for intake in intakes)
            events.append(f"wait:{self.name}")

    manager = object.__new__(ChannelManager)
    manager.channels = {
        "first": SimpleNamespace(intake=_Intake("first")),
        "second": SimpleNamespace(intake=_Intake("second")),
    }

    await manager.quiesce_intake()

    assert events == ["seal:first", "seal:second", "wait:first", "wait:second"]


async def test_quiesce_intake_preserves_caller_cancellation_over_barrier_failure() -> None:
    wait_started = asyncio.Event()
    release_wait = asyncio.Event()

    class _FailingIntake:
        def __init__(self) -> None:
            self.sealed = False
            self.cancel_calls = 0

        def seal(self) -> None:
            self.sealed = True

        async def wait_idle(self) -> None:
            wait_started.set()
            await release_wait.wait()
            raise RuntimeError("wait_idle failed")

        def cancel_inflight(self) -> int:
            self.cancel_calls += 1
            return 0

    intake = _FailingIntake()
    manager = object.__new__(ChannelManager)
    manager.channels = {"telegram": SimpleNamespace(intake=intake)}
    quiescing = asyncio.create_task(manager.quiesce_intake())
    await wait_started.wait()
    quiescing.cancel()
    await asyncio.sleep(0)

    try:
        assert not quiescing.done()
        release_wait.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.wait_for(quiescing, timeout=0.2)
    finally:
        release_wait.set()
        if not quiescing.done():
            quiescing.cancel()
        await asyncio.gather(quiescing, return_exceptions=True)

    assert quiescing.cancelled()
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "wait_idle failed"
    assert intake.sealed
    assert intake.cancel_calls == 1


_EDITABLE_JSON = '{"url": "file:///src", "dir_info": {"editable": true}}'
_WHEEL_JSON = '{"url": "https://x/codeflow_harness-0.1.2.whl", "archive_info": {}}'


def _patch_direct_url(monkeypatch, read_text_result, calls=None):
    class _Dist:
        def read_text(self, name):
            return read_text_result

    def fake_distribution(package):
        if calls is not None:
            calls.append(package)
        return _Dist()

    monkeypatch.setattr("codeflow.channels.manager.distribution", fake_distribution)


@pytest.mark.parametrize("modname", ["feishu", "qq", "wecom"])
def test_hint_editable_names_the_channel_extra(monkeypatch, modname):
    """Editable checkout -> `uv sync --extra channel-<name>`, name interpolated."""
    calls = []
    _patch_direct_url(monkeypatch, _EDITABLE_JSON, calls)
    assert _missing_dep_hint(modname) == f"Run: uv sync --extra channel-{modname}"
    assert calls == ["codeflow-harness"]


@pytest.mark.parametrize(
    "raw",
    [
        _WHEEL_JSON,
        None,
        '{"url": "file:///x", "dir_info": {}}',
        "{}",
        "{not valid json",
    ],
    ids=["wheel", "absent", "dir_info_no_editable", "empty", "malformed"],
)
def test_hint_non_editable_points_to_installer(monkeypatch, raw):
    """Any non-editable / malformed direct_url.json -> installer hint, never raises."""
    _patch_direct_url(monkeypatch, raw)
    monkeypatch.setattr("codeflow.channels.manager.sys.platform", "linux")
    hint = _missing_dep_hint("wecom")
    assert "uv sync" not in hint
    assert "install.sh" in hint
    assert "CODEFLOW_GITHUB_TOKEN" in hint


def test_hint_package_not_found_points_to_installer(monkeypatch):
    """CodeFlow distribution not found -> installer hint, no exception."""

    def _raise(pkg):
        raise PackageNotFoundError(pkg)

    monkeypatch.setattr("codeflow.channels.manager.distribution", _raise)
    monkeypatch.setattr("codeflow.channels.manager.sys.platform", "darwin")
    assert "install.sh" in _missing_dep_hint("qq")


@pytest.mark.parametrize(
    "platform, marker",
    [("win32", "install.ps1"), ("darwin", "install.sh"), ("linux", "install.sh")],
)
def test_hint_installer_matches_os(monkeypatch, platform, marker):
    """Wheel install picks the installer for the running OS (irm vs curl)."""
    _patch_direct_url(monkeypatch, _WHEEL_JSON)
    monkeypatch.setattr("codeflow.channels.manager.sys.platform", platform)
    assert marker in _missing_dep_hint("feishu")


@pytest.mark.parametrize(
    "direct_url, platform, expected",
    [
        (_EDITABLE_JSON, "linux", "uv sync --extra channel-feishu"),
        (_WHEEL_JSON, "linux", "install.sh"),
        (_WHEEL_JSON, "win32", "install.ps1"),
    ],
    ids=["editable", "wheel-unix", "wheel-win"],
)
def test_init_warning_carries_install_hint(monkeypatch, direct_url, platform, expected):
    """A channel disabled by ImportError logs the mode-correct install hint."""
    from loguru import logger

    _patch_direct_url(monkeypatch, direct_url)
    monkeypatch.setattr("codeflow.channels.manager.sys.platform", platform)

    def boom(config):
        raise ImportError("No module named 'lark_oapi'")

    lines: list[str] = []
    sink_id = logger.add(lambda m: lines.append(str(m)), level="WARNING")
    try:
        _manager(
            monkeypatch,
            {"feishu": _spec(boom)},
            _config({"feishu": SimpleNamespace(enabled=True, allow_from=["*"])}),
        )
    finally:
        logger.remove(sink_id)

    warning = "".join(lines)
    assert "feishu channel disabled" in warning
    assert expected in warning


def test_get_status_and_get_channel(monkeypatch):
    mgr = _manager(
        monkeypatch,
        {"fake": _spec(_FakeChannel)},
        _config({"fake": SimpleNamespace(enabled=True, allow_from=["*"])}),
    )
    mgr.channels["fake"]._running = True
    assert mgr.get_status() == {"fake": {"enabled": True, "running": True}}
    assert mgr.get_channel("fake") is mgr.channels["fake"]
    assert mgr.get_channel("nope") is None
