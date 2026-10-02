"""PG-2 — multi-source plugin discovery + conflict resolution."""

from __future__ import annotations

import textwrap
from importlib import metadata
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from codeflow.plugin import (
    DiscoveredPlugin,
    PluginCompatibilityError,
    PluginDiscovery,
    PluginIdentityError,
    Source,
)


def _write_manifest(root: Path, plugin_id: str, *, extra: str = "") -> Path:
    """Drop a minimal valid manifest at ``root/<plugin_id>/codeflow-plugin.toml``."""
    sub = root / plugin_id
    sub.mkdir(parents=True, exist_ok=True)
    body = textwrap.dedent(f"""
        [plugin]
        id = "{plugin_id}"
        version = "0.1.0"
        {extra}
    """)
    path = sub / "codeflow-plugin.toml"
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestSingleSource:
    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        d = PluginDiscovery(bundled_dir=tmp_path)
        assert d.discover() == []

    def test_nonexistent_dir_returns_empty_list(self, tmp_path: Path) -> None:
        d = PluginDiscovery(bundled_dir=tmp_path / "missing")
        assert d.discover() == []

    def test_finds_single_manifest(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, "foo")
        d = PluginDiscovery(bundled_dir=tmp_path)
        out = d.discover()
        assert len(out) == 1
        assert out[0].manifest.id == "foo"
        assert out[0].source == Source.BUNDLED
        assert out[0].location is not None
        assert out[0].location.name == "codeflow-plugin.toml"

    def test_finds_multiple_manifests_sorted_by_id(self, tmp_path: Path) -> None:
        for pid in ("zeta", "alpha", "mid"):
            _write_manifest(tmp_path, pid)
        d = PluginDiscovery(user_dir=tmp_path)
        ids = [p.manifest.id for p in d.discover()]
        assert ids == ["alpha", "mid", "zeta"]

    def test_subdir_without_manifest_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "not-a-plugin").mkdir()
        _write_manifest(tmp_path, "real")
        d = PluginDiscovery(bundled_dir=tmp_path)
        assert [p.manifest.id for p in d.discover()] == ["real"]

    def test_malformed_manifest_skipped_silently(
        self,
        tmp_path: Path,
        caplog,
    ) -> None:
        sub = tmp_path / "broken"
        sub.mkdir()
        (sub / "codeflow-plugin.toml").write_text(
            "not valid toml [[[",
            encoding="utf-8",
        )
        _write_manifest(tmp_path, "ok")
        d = PluginDiscovery(bundled_dir=tmp_path)
        out = d.discover()

        assert [p.manifest.id for p in out] == ["ok"]

    def test_incompatible_manifest_fails_closed(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, "future", extra='codeflow = ">=0.2,<0.3"')

        with pytest.raises(PluginCompatibilityError, match=r"requires CodeFlow >=0.2,<0.3.*installed CodeFlow is 0.1.7"):
            PluginDiscovery(bundled_dir=tmp_path, codeflow_version="0.1.7").discover()


class TestEntryPointIdentity:
    @staticmethod
    def _entry_point(tmp_path: Path, *, plugin_id: str = "myna-memory", version: str = "0.1.1rc3"):
        relative = PurePosixPath("myna/integrations/codeflow/codeflow-plugin.toml")
        manifest = tmp_path / relative
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            textwrap.dedent(f"""
                [plugin]
                id = "{plugin_id}"
                version = "{version}"
                codeflow = ">=0.1,<0.2"
                enabled_by_default = true

                [[plugin.contributes.memory_backends]]
                name = "myna"
                factory = "myna.integrations.codeflow:make_backend"
            """),
            encoding="utf-8",
        )
        distribution = SimpleNamespace(
            name="myna-memory",
            version="0.1.1rc3",
            files=[relative],
            locate_file=lambda path: tmp_path / path,
        )
        return SimpleNamespace(
            name="myna",
            value="myna.integrations.codeflow",
            dist=distribution,
        )

    def test_reads_installed_manifest_without_importing_plugin_package(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        entry_point = self._entry_point(tmp_path)
        monkeypatch.setattr(metadata, "entry_points", lambda **_: (entry_point,))

        discovered = PluginDiscovery(entry_points_group="codeflow.plugins", codeflow_version="0.1.7").discover()

        assert discovered[0].manifest.id == "myna-memory"
        assert discovered[0].manifest.version == "0.1.1rc3"
        assert discovered[0].manifest.codeflow == ">=0.1,<0.2"
        assert discovered[0].manifest.contributes.memory_backends[0].name == "myna"

    @pytest.mark.parametrize(
        ("plugin_id", "version", "message"),
        [
            ("other-memory", "0.1.1rc3", "distribution 'myna-memory'.*manifest id 'other-memory'"),
            ("myna-memory", "9.9.9", "distribution version 0.1.1rc3.*manifest version 9.9.9"),
        ],
    )
    def test_distribution_and_manifest_identity_must_match(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        plugin_id: str,
        version: str,
        message: str,
    ) -> None:
        entry_point = self._entry_point(tmp_path, plugin_id=plugin_id, version=version)
        monkeypatch.setattr(metadata, "entry_points", lambda **_: (entry_point,))

        with pytest.raises(PluginIdentityError, match=message):
            PluginDiscovery(entry_points_group="codeflow.plugins", codeflow_version="0.1.7").discover()


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestConflictResolution:
    def test_bundled_shadows_user(self, tmp_path: Path, caplog) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        _write_manifest(bundled, "example")
        _write_manifest(user, "example")
        d = PluginDiscovery(bundled_dir=bundled, user_dir=user)
        out = d.discover()
        assert len(out) == 1

        assert out[0].source == Source.BUNDLED

    def test_user_shadows_project(self, tmp_path: Path) -> None:
        user = tmp_path / "user"
        project = tmp_path / "project"
        _write_manifest(user, "myplug")
        _write_manifest(project, "myplug")
        d = PluginDiscovery(user_dir=user, project_dir=project)
        out = d.discover()
        assert len(out) == 1
        assert out[0].source == Source.USER

    def test_priority_order_full_chain(self, tmp_path: Path) -> None:
        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        project = tmp_path / "project"

        _write_manifest(bundled, "x")
        _write_manifest(user, "x")
        _write_manifest(project, "x")

        _write_manifest(bundled, "b-only")
        _write_manifest(user, "u-only")
        _write_manifest(project, "p-only")
        d = PluginDiscovery(
            bundled_dir=bundled,
            user_dir=user,
            project_dir=project,
        )
        out = d.discover()
        by_id = {p.manifest.id: p.source for p in out}
        assert by_id == {
            "b-only": Source.BUNDLED,
            "u-only": Source.USER,
            "p-only": Source.PROJECT,
            "x": Source.BUNDLED,
        }


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestSubdirNameMismatch:
    def test_id_in_manifest_wins(self, tmp_path: Path) -> None:

        sub = tmp_path / "wrong-dirname"
        sub.mkdir()
        (sub / "codeflow-plugin.toml").write_text(
            textwrap.dedent("""
            [plugin]
            id = "correct"
            version = "0.1"
        """),
            encoding="utf-8",
        )
        d = PluginDiscovery(bundled_dir=tmp_path)
        out = d.discover()
        assert out[0].manifest.id == "correct"


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestDiscoveredPluginRecord:
    def test_record_is_frozen(self, tmp_path: Path) -> None:
        from dataclasses import FrozenInstanceError

        import pytest

        _write_manifest(tmp_path, "x")
        out = PluginDiscovery(bundled_dir=tmp_path).discover()
        rec: DiscoveredPlugin = out[0]
        with pytest.raises(FrozenInstanceError):
            rec.source = Source.USER  # type: ignore[misc]
