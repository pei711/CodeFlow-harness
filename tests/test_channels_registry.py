"""Tests for codeflow.channels.registry — adapter auto-discovery (spec-based)."""

from codeflow.channels.contract import Capabilities, ChannelSpec
from codeflow.channels.registry import discover_channel_names, discover_specs


def test_discover_channel_names_lists_adapter_packages():
    names = discover_channel_names()
    assert names == ["feishu", "qq", "wecom"]
    assert "__pycache__" not in names


def test_discover_specs_returns_channel_specs():
    specs = discover_specs()
    assert specs
    assert set(specs) <= set(discover_channel_names())
    for spec in specs.values():
        assert isinstance(spec, ChannelSpec)
        assert callable(spec.factory)
        assert isinstance(spec.display_name, str) and spec.display_name
        assert isinstance(spec.capabilities, Capabilities)


def test_discover_specs_exposes_only_retained_channels():
    specs = discover_specs()
    assert set(specs) == {"feishu", "qq", "wecom"}
    assert all(not spec.capabilities.interactive_login for spec in specs.values())


def test_discover_specs_is_cheap():
    """Discovery imports only each spec.py — never the channel SDKs (those are
    deferred into the spec factories)."""
    import subprocess
    import sys

    code = (
        "import sys; from codeflow.channels.registry import discover_specs; discover_specs();"
        "pulled = {m for m in ('botpy', 'lark_oapi', 'wecom_aibot_sdk') if m in sys.modules};"
        "assert not pulled, f'discovery pulled in channel SDKs: {pulled}'"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
