"""Tests for the channel contract (channels/contract.py): Capabilities,
the Channel / Supports* protocols (runtime-checkable), ChannelSpec, and the
capability-proof helper."""

from codeflow.channels import (
    Capabilities,
    Channel,
    ChannelSpec,
    SupportsLogin,
    SupportsStreaming,
)
from codeflow.channels.contract import capability_violations
from codeflow.channels.intake import Intake


class _Min:
    name = "min"
    capabilities = Capabilities()
    intake = Intake("min", object())

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send(self, chat_id, content, media=None) -> None: ...


class _WithLogin(_Min):
    capabilities = Capabilities(interactive_login=True)

    async def login(self, force: bool = False) -> bool:
        return True


class _WithStreaming(_Min):
    capabilities = Capabilities(streaming=True)

    async def send_stream_chunk(self, chat_id, stream_id, delta, *, done=False) -> None: ...


def test_capabilities_defaults_all_false():
    c = Capabilities()
    assert (c.interactive_login, c.streaming) == (False, False)


def test_channel_spec_fields_and_factory():
    spec = ChannelSpec(
        display_name="WeChat",
        factory=lambda cfg, bus: _Min(),
        capabilities=Capabilities(interactive_login=True),
    )
    assert spec.display_name == "WeChat" and spec.capabilities.interactive_login is True
    assert isinstance(spec.factory("cfg", "bus"), _Min)


def test_channel_spec_capabilities_default_empty():
    spec = ChannelSpec(display_name="X", factory=lambda cfg, bus: _Min())
    assert spec.capabilities == Capabilities()


def test_min_satisfies_channel_protocol():
    assert isinstance(_Min(), Channel)


def test_supports_protocols_are_opt_in():
    assert not isinstance(_Min(), SupportsLogin)
    assert not isinstance(_Min(), SupportsStreaming)
    assert isinstance(_WithLogin(), SupportsLogin)
    assert isinstance(_WithStreaming(), SupportsStreaming)


def test_capability_violations_consistent():
    assert capability_violations(_Min()) == []
    assert capability_violations(_WithLogin()) == []
    assert capability_violations(_WithStreaming()) == []


def test_capability_violations_declared_but_missing():

    bad_login = _Min()
    bad_login.capabilities = Capabilities(interactive_login=True)
    assert any("interactive_login" in m for m in capability_violations(bad_login))

    bad_stream = _Min()
    bad_stream.capabilities = Capabilities(streaming=True)
    assert any("streaming" in m for m in capability_violations(bad_stream))


def test_capability_violations_implemented_but_undeclared():

    class _SneakyLogin(_Min):
        async def login(self, force: bool = False) -> bool:
            return True

    assert any("SupportsLogin" in m for m in capability_violations(_SneakyLogin()))

    class _SneakyStream(_Min):
        async def send_stream_chunk(self, chat_id, stream_id, delta, *, done=False) -> None: ...

    assert any("SupportsStreaming" in m for m in capability_violations(_SneakyStream()))
