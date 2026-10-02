"""CFG-1 — CodeFlowConfig: plugins / memory / skill_router sections + migration."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import pytest

from codeflow.config.codeflow import (
    CodeFlowConfig,
    MemoryConfig,
    PluginsConfig,
    SkillForgeConfig,
    SkillForgeRouterConfig,
    load_codeflow_config,
)
from codeflow.config.loader import EXTENSION_KEYS

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestDefaults:
    def test_plugins_defaults(self) -> None:
        c = PluginsConfig()
        assert c.disabled == []
        assert c.config == {}

    def test_memory_defaults(self) -> None:
        c = MemoryConfig()
        assert c.backend == "myna"
        assert c.user_id == "default"
        assert c.memory_top_k == 5

    def test_memory_backend_none_disables(self) -> None:
        c = MemoryConfig(backend=None)
        assert c.backend is None

    def test_skill_router_defaults(self) -> None:
        c = SkillForgeRouterConfig()
        assert c.enabled is True
        assert c.local_min_score == 0.0
        assert c.top_k == 5

    def test_skill_forge_public_defaults(self) -> None:
        c = SkillForgeConfig()
        assert c.rewrite_enabled is False
        assert c.llm_gate_enabled is False
        assert c.embedding_model == "default"
        assert c.embedding_url == "http://localhost:1357"
        assert c.embedding_api_key is None
        assert c.reranker_model == "default"
        assert c.reranker_url == "http://localhost:1357"
        assert c.reranker_api_key is None
        assert c.mass_library_db is None

        exported = c.model_dump_json()
        assert not re.search(
            r"https?://(?:10\.|127\.|192\.168\.|172\.(?:1[6-9]|2\d|3[0-1])\.)",
            exported,
        )

    def test_root_default_factories_wired(self) -> None:
        c = CodeFlowConfig()
        assert isinstance(c.plugins, PluginsConfig)
        assert isinstance(c.memory, MemoryConfig)
        assert isinstance(c.skill_forge.router, SkillForgeRouterConfig)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestKeyAliasing:
    def test_camel_keys_accepted(self) -> None:
        c = MemoryConfig.model_validate(
            {
                "userId": "alice",
                "memoryTopK": 7,
            }
        )
        assert c.user_id == "alice"
        assert c.memory_top_k == 7

    def test_snake_keys_accepted(self) -> None:
        c = MemoryConfig.model_validate(
            {
                "user_id": "bob",
                "memory_top_k": 3,
            }
        )
        assert c.user_id == "bob"
        assert c.memory_top_k == 3


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestExtensionKeys:
    def test_plugins_listed(self) -> None:
        assert "plugins" in EXTENSION_KEYS

    def test_memory_listed(self) -> None:
        assert "memory" in EXTENSION_KEYS

    def test_skill_router_nested_under_skill_forge(self) -> None:

        assert "skillRouter" not in EXTENSION_KEYS
        assert "skill_router" not in EXTENSION_KEYS
        assert "skillForge" in EXTENSION_KEYS


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, body: dict) -> Path:
    """Write a config file + return its path."""
    p = tmp_path / "config.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


class TestLoaderIntegration:
    def test_loads_new_sections_from_file(self, tmp_path: Path) -> None:
        path = _write_config(
            tmp_path,
            {
                "plugins": {"disabled": ["test-memory"], "config": {}},
                "memory": {
                    "backend": "example",
                    "userId": "alice",
                    "memoryTopK": 10,
                },
                "skillRouter": {
                    "topK": 8,
                    "mass": {"endpoint": "http://mass.internal:9001"},
                },
            },
        )
        cfg = load_codeflow_config(path)
        assert cfg.plugins.disabled == ["test-memory"]
        assert cfg.plugins.config == {}
        assert cfg.memory.backend == "example"
        assert cfg.memory.user_id == "alice"
        assert cfg.memory.memory_top_k == 10
        assert cfg.skill_forge.router.top_k == 8

    def test_missing_sections_use_defaults(self, tmp_path: Path) -> None:
        path = _write_config(tmp_path, {})
        cfg = load_codeflow_config(path)

        assert cfg.plugins.disabled == []
        assert cfg.memory.backend == "myna"
        assert cfg.skill_forge.router.enabled is True

    def test_explicit_null_section_uses_defaults(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_config(
            tmp_path,
            {
                "plugins": None,
                "memory": None,
                "skillForge": None,
            },
        )
        cfg = load_codeflow_config(path)

        assert isinstance(cfg.plugins, PluginsConfig)
        assert isinstance(cfg.memory, MemoryConfig)
        assert isinstance(cfg.skill_forge.router, SkillForgeRouterConfig)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestMassLibraryDbDeprecation:
    def test_legacy_only_emits_deprecation_warning(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_config(
            tmp_path,
            {
                "skill_forge": {"mass_library_db": "/tmp/old.db"},
            },
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_codeflow_config(path)
        deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deps, "expected at least one DeprecationWarning"
        assert "mass_library_db" in str(deps[0].message)
        assert "configured filesystem sources" in str(deps[0].message)

    def test_no_legacy_field_no_warning(self, tmp_path: Path) -> None:
        path = _write_config(
            tmp_path,
            {
                "skill_router": {"mass": {"endpoint": "http://m"}},
            },
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_codeflow_config(path)
        deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deps == []


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestStrictness:
    def test_unknown_field_in_plugins_rejected(self) -> None:
        with pytest.raises(Exception):
            PluginsConfig.model_validate(
                {
                    "disabled": [],
                    "config": {},
                    "unknown_field": True,
                }
            )

    def test_unknown_field_in_memory_rejected(self) -> None:
        with pytest.raises(Exception):
            MemoryConfig.model_validate({"backend": "x", "typo": 1})

    @pytest.mark.parametrize(
        "value",
        [
            -0.1,
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            False,
        ],
    )
    def test_local_min_score_requires_nonnegative_finite_number_not_bool(
        self,
        value: object,
    ) -> None:
        with pytest.raises(ValueError):
            SkillForgeRouterConfig(local_min_score=value)  # type: ignore[arg-type]
