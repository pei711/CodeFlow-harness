"""Plugin ``tools`` contribution point, registry, CLI stack, and rendering."""

from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from codeflow.plugin import (
    Contributes,
    DiscoveredPlugin,
    PluginConflictError,
    PluginContext,
    PluginManifest,
    PluginNotFoundError,
    PluginRegistry,
    ServiceLocator,
    Source,
    ToolContribution,
)

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def _install_test_module(name: str, attrs: dict[str, object]) -> None:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


@pytest.fixture(autouse=True)
def _cleanup_modules():
    snapshot = set(sys.modules)
    yield
    for k in set(sys.modules) - snapshot:
        sys.modules.pop(k, None)


def _discovered_with_tools(
    plugin_id: str,
    tools: list[tuple[str, str]],
) -> DiscoveredPlugin:
    mf = PluginManifest(
        id=plugin_id,
        version="0.1.0",
        enabled_by_default=True,
        contributes=Contributes(
            tools=[ToolContribution(name=n, factory=f) for n, f in tools],
        ),
    )
    return DiscoveredPlugin(manifest=mf, source=Source.USER, location=None)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestManifestTools:
    def test_parses_tools_contribution(self) -> None:
        toml = textwrap.dedent("""
            [plugin]
            id = "media-tools"
            version = "0.1.0"
            [[plugin.contributes.tools]]
            name = "inspect_media"
            factory = "_test_media_tools:make_inspect_media_tool"
        """)
        mf = PluginManifest.from_toml_str(toml)
        assert [t.name for t in mf.contributes.tools] == ["inspect_media"]
        assert mf.contributes.tools[0].factory.endswith(":make_inspect_media_tool")

    def test_default_tools_empty(self) -> None:
        mf = PluginManifest(id="p", version="0.1.0")
        assert mf.contributes.tools == []

    def test_bad_factory_ref_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolContribution(name="t", factory="not-a-ref")

    def test_duplicate_tool_names_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate tool name"):
            PluginManifest(
                id="p",
                version="0.1.0",
                contributes=Contributes(
                    tools=[
                        ToolContribution(name="dup", factory="m:a"),
                        ToolContribution(name="dup", factory="m:b"),
                    ],
                ),
            )

    def test_backend_and_tool_may_share_name(self) -> None:

        from codeflow.plugin import MemoryBackendContribution

        mf = PluginManifest(
            id="p",
            version="0.1.0",
            contributes=Contributes(
                memory_backends=[MemoryBackendContribution(name="x", factory="m:b")],
                tools=[ToolContribution(name="x", factory="m:t")],
            ),
        )
        assert mf.contributes.tools[0].name == "x"


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestRegistryTools:
    def test_activates_and_resolves_tool(self) -> None:
        def fake_factory(ctx):
            return "tool-instance"

        _install_test_module("_tp_tools_a", {"make_tool": fake_factory})
        reg = PluginRegistry()
        reg.activate([_discovered_with_tools("alpha", [("inspect_media", "_tp_tools_a:make_tool")])])

        assert reg.tool_names() == ["inspect_media"]
        assert reg.tool_plugin_id("inspect_media") == "alpha"
        assert reg.get_tool_factory("inspect_media") is fake_factory

    def test_build_tool_invokes_factory_with_context(self, tmp_path: Path) -> None:
        captured = {}

        def fake_factory(ctx: PluginContext):
            captured["config"] = ctx.config
            return "built"

        _install_test_module("_tp_tools_b", {"make_tool": fake_factory})
        reg = PluginRegistry()
        reg.activate([_discovered_with_tools("p", [("t", "_tp_tools_b:make_tool")])])

        out = reg.build_tool(
            "t",
            config={"k": 1},
            services=ServiceLocator(workspace=tmp_path),
        )
        assert out == "built"
        assert captured["config"] == {"k": 1}

    def test_tool_name_conflict_across_plugins(self) -> None:
        def f(ctx):
            return None

        _install_test_module("_tp_tools_c", {"make_tool": f})
        reg = PluginRegistry()
        with pytest.raises(PluginConflictError, match="tool 'dup'"):
            reg.activate(
                [
                    _discovered_with_tools("one", [("dup", "_tp_tools_c:make_tool")]),
                    _discovered_with_tools("two", [("dup", "_tp_tools_c:make_tool")]),
                ]
            )

    def test_unknown_tool_raises(self) -> None:
        reg = PluginRegistry()
        with pytest.raises(PluginNotFoundError):
            reg.get_tool_factory("nope")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestBuildPluginTools:
    def _config(self, plugin_config: dict | None = None):
        from codeflow.config.codeflow import CodeFlowConfig, PluginsConfig

        return CodeFlowConfig(plugins=PluginsConfig(config=dict(plugin_config or {})))

    def test_builds_tools_from_registry(self, tmp_path: Path) -> None:
        from codeflow.cli._plugin_stack import build_plugin_tools

        seen = {}

        def fake_factory(ctx):
            seen["config"] = ctx.config
            return f"tool::{ctx.config.get('flag')}"

        _install_test_module("_tp_tools_d", {"make_tool": fake_factory})
        reg = PluginRegistry()
        reg.activate([_discovered_with_tools("myplugin", [("t1", "_tp_tools_d:make_tool")])])

        cfg = self._config({"myplugin": {"flag": "on"}})
        tools = build_plugin_tools(tmp_path, cfg, registry=reg)
        assert tools == ["tool::on"]
        assert seen["config"] == {"flag": "on"}

    def test_empty_when_no_tools(self, tmp_path: Path) -> None:
        from codeflow.cli._plugin_stack import build_plugin_tools

        assert build_plugin_tools(tmp_path, self._config(), registry=PluginRegistry()) == []

    def test_failing_factory_is_skipped(self, tmp_path: Path) -> None:
        from codeflow.cli._plugin_stack import build_plugin_tools

        def boom(ctx):
            raise RuntimeError("nope")

        _install_test_module("_tp_tools_e", {"make_tool": boom})
        reg = PluginRegistry()
        reg.activate([_discovered_with_tools("p", [("t", "_tp_tools_e:make_tool")])])

        assert build_plugin_tools(tmp_path, self._config(), registry=reg) == []

    def test_none_factory_is_skipped(self, tmp_path: Path) -> None:
        from codeflow.cli._plugin_stack import build_plugin_tools

        def opt_out(ctx):
            return None

        _install_test_module("_tp_tools_f", {"make_tool": opt_out})
        reg = PluginRegistry()
        reg.activate([_discovered_with_tools("p", [("t", "_tp_tools_f:make_tool")])])
        assert build_plugin_tools(tmp_path, self._config(), registry=reg) == []


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestRenderAttachments:
    def test_non_image_surfaced_as_note(self, tmp_path: Path) -> None:
        from codeflow.context_engine.segments import render

        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4 data")
        out = render.build_user_content("summarize this", [str(pdf)])
        assert isinstance(out, str)
        assert "report.pdf" in out
        assert "report.pdf" in out
        assert "summarize this" in out

    def test_image_inlined_as_block(self, tmp_path: Path) -> None:
        import base64 as _b64

        from codeflow.context_engine.segments import render

        png = tmp_path / "a.png"
        png.write_bytes(
            _b64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            )
        )
        out = render.build_user_content("look", [str(png)])
        assert isinstance(out, list)
        assert out[0]["type"] == "image_url"
        assert out[-1] == {"type": "text", "text": "look"}

    def test_mixed_image_and_doc(self, tmp_path: Path) -> None:
        import base64 as _b64

        from codeflow.context_engine.segments import render

        png = tmp_path / "a.png"
        png.write_bytes(
            _b64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            )
        )
        pdf = tmp_path / "d.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        out = render.build_user_content("q", [str(png), str(pdf)])
        assert isinstance(out, list)
        assert out[0]["type"] == "image_url"
        text_block = out[-1]["text"]
        assert "d.pdf" in text_block

    def test_no_media_returns_text(self) -> None:
        from codeflow.context_engine.segments import render

        assert render.build_user_content("hi", None) == "hi"
