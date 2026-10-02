"""Installed plugin-stack assembly and retired backend errors."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from codeflow.cli._plugin_stack import MynaSetupError, build_plugin_registry, maybe_build_memory_backend
from codeflow.config.codeflow import CodeFlowConfig, MemoryConfig, PluginsConfig
from codeflow.memory_engine import MemoryBackend
from codeflow.plugin import (
    Contributes,
    DiscoveredPlugin,
    MemoryBackendContribution,
    PluginIdentityError,
    PluginManifest,
    PluginNotFoundError,
    PluginRegistry,
    Source,
)


def _config(memory_backend: str | None = "myna") -> CodeFlowConfig:
    return CodeFlowConfig(
        memory=MemoryConfig(backend=memory_backend),
        plugins=PluginsConfig(),
    )


def _registry(
    *,
    plugin_id: str = "myna-memory",
    factory_ref: str = "myna.integrations.codeflow:make_backend",
    start_error: Exception | None = None,
):
    module_name = factory_ref.partition(":")[0]
    module = types.ModuleType(module_name)

    class Backend:
        async def recall(self, query, *, user_id=None, agent_id=None, top_k=5):
            return []

        async def store(self, session_id, messages):
            return None

        async def feedback(self, signals):
            return None

        async def start(self):
            if start_error is not None:
                raise start_error
            return None

        async def stop(self):
            return None

    module.make_backend = lambda _context: Backend()
    sys.modules[module_name] = module
    registry = PluginRegistry()
    registry.activate(
        [
            DiscoveredPlugin(
                manifest=PluginManifest(
                    id=plugin_id,
                    version="0.1.1rc3",
                    codeflow=">=0.1,<0.2",
                    enabled_by_default=True,
                    contributes=Contributes(
                        memory_backends=[MemoryBackendContribution(name="myna", factory=factory_ref)]
                    ),
                ),
                source=Source.ENTRY_POINTS,
                location=None,
            )
        ]
    )
    return registry


def test_default_config_builds_validated_myna_backend(tmp_path: Path) -> None:
    backend = maybe_build_memory_backend(tmp_path, _config(), registry=_registry())
    assert isinstance(backend, MemoryBackend)


def test_memory_backend_none_returns_none(tmp_path: Path) -> None:
    assert maybe_build_memory_backend(tmp_path, _config(None)) is None


async def test_myna_uninitialized_error_has_setup_remediation(tmp_path: Path) -> None:
    class ConfigurationInvalid(RuntimeError):
        code = "configuration_invalid"

    backend = maybe_build_memory_backend(
        tmp_path,
        _config(),
        registry=_registry(start_error=ConfigurationInvalid("invalid")),
    )
    assert backend is not None

    with pytest.raises(MynaSetupError, match=r"myna init.*myna doctor --live"):
        await backend.start()


def test_unknown_backend_raises(tmp_path: Path) -> None:
    with pytest.raises(PluginNotFoundError):
        maybe_build_memory_backend(tmp_path, _config("nonexistent"))


@pytest.mark.parametrize("retired", ["codecairn", "everos"])
def test_retired_backend_fails_without_rewrite(
    tmp_path: Path,
    retired: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    data = {"memory": {"backend": retired}}
    config_path.write_text(json.dumps(data), encoding="utf-8")
    config = _config(retired)
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.build_plugin_registry",
        lambda _config: pytest.fail("retired backend must be rejected before plugin discovery"),
    )

    with pytest.raises(
        PluginNotFoundError,
        match="set memory.backend to 'myna'.*or set it to null",
    ):
        maybe_build_memory_backend(tmp_path, config)

    assert json.loads(config_path.read_text(encoding="utf-8")) == data


@pytest.mark.parametrize(
    ("plugin_id", "factory_ref"),
    [
        ("other-memory", "_test_myna_backend:make_backend"),
        ("myna-memory", "myna.integrations.codeflow.backend:make_backend"),
    ],
)
def test_myna_backend_requires_frozen_public_manifest_identity(
    tmp_path: Path,
    plugin_id: str,
    factory_ref: str,
) -> None:
    registry = _registry(plugin_id=plugin_id, factory_ref=factory_ref)

    with pytest.raises(PluginIdentityError, match="Myna plugin manifest identity"):
        maybe_build_memory_backend(tmp_path, _config(), registry=registry)


def test_myna_manifest_identity_is_validated_before_factory_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = tmp_path / "plugins" / "myna-memory"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "codeflow-plugin.toml").write_text(
        """
[plugin]
id = "myna-memory"
version = "0.1.1rc3"
codeflow = ">=0.1,<0.2"
enabled_by_default = true

[[plugin.contributes.memory_backends]]
name = "myna"
factory = "_must_not_import:make_backend"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.plugin_discovery_sources",
        lambda: {
            "bundled_dir": None,
            "user_dir": tmp_path / "plugins",
            "project_dir": None,
            "entry_points_group": None,
        },
    )

    with pytest.raises(PluginIdentityError, match="Myna plugin manifest identity"):
        build_plugin_registry(_config())

    assert "_must_not_import" not in sys.modules
