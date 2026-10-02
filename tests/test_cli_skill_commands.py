"""CLI tests for ``codeflow skills``.

Two subcommands are covered with mocked ``SkillService`` so the tests
stay self-contained.
"""

from __future__ import annotations

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


# ============================================================================

# ============================================================================


def test_skills_help_lists_all_subcommands() -> None:
    r = runner.invoke(app, ["skills", "--help"])
    assert r.exit_code == 0
    for sub in ("list", "get"):
        assert sub in r.stdout, f"missing subcommand in --help: {sub}"


def test_skill_commands_use_current_project_state(
    tmp_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codeflow.cli.skill_commands import _build_skill_service

    project = tmp_path / "project"
    project.mkdir()
    product_home = tmp_path / "codeflow-home"
    monkeypatch.setenv("CODEFLOW_HOME", str(product_home))
    monkeypatch.chdir(project)

    service = _build_skill_service()

    assert service.registry.workspace_skills.parent.parent == product_home / "projects"
    assert service.registry.workspace_skills.parent.name.startswith("project-")
    assert list(project.iterdir()) == []


@pytest.mark.parametrize("subcmd", ["list", "get"])
def test_skills_subcommand_help_works(subcmd: str) -> None:
    """Every skills subcommand exposes ``--help`` without crashing."""
    r = runner.invoke(app, ["skills", subcmd, "--help"])
    assert r.exit_code == 0


# ============================================================================

# ============================================================================


def _make_meta(name: str, source: str = "builtin", desc: str = "stub") -> SimpleNamespace:
    return SimpleNamespace(name=name, source=source, description=desc)


def test_skill_list_renders_table(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``skills list`` prints a table when the registry returns metas."""
    fake_svc = SimpleNamespace(
        gather_all_skills=lambda: [_make_meta("alpha"), _make_meta("beta", source="workspace")],
    )
    monkeypatch.setattr("codeflow.cli.skill_commands._build_skill_service", lambda: fake_svc)

    r = runner.invoke(app, ["skills", "list"])
    assert r.exit_code == 0
    assert "alpha" in r.stdout
    assert "beta" in r.stdout
    assert "Skills" in r.stdout


def test_skill_list_empty_message(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty registry prints the ``No skills found`` notice."""
    fake_svc = SimpleNamespace(gather_all_skills=lambda: [])
    monkeypatch.setattr("codeflow.cli.skill_commands._build_skill_service", lambda: fake_svc)

    r = runner.invoke(app, ["skills", "list"])
    assert r.exit_code == 0
    assert "No skills found" in r.stdout


def test_skill_list_filters_by_source(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--source workspace`` only shows metas matching that source."""
    fake_svc = SimpleNamespace(
        gather_all_skills=lambda: [
            _make_meta("alpha", source="builtin"),
            _make_meta("beta", source="workspace"),
        ],
    )
    monkeypatch.setattr("codeflow.cli.skill_commands._build_skill_service", lambda: fake_svc)

    r = runner.invoke(app, ["skills", "list", "--source", "workspace"])
    assert r.exit_code == 0
    assert "beta" in r.stdout
    assert "alpha" not in r.stdout


def test_skill_get_known_skill(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``skills get <name>`` prints the metadata fields."""
    fake_svc = SimpleNamespace(
        get_skill_metadata=lambda _: {"name": "alpha", "source": "builtin"},
        load_skill=lambda _: "# SKILL.md body",
    )
    monkeypatch.setattr("codeflow.cli.skill_commands._build_skill_service", lambda: fake_svc)

    r = runner.invoke(app, ["skills", "get", "alpha"])
    assert r.exit_code == 0
    assert "alpha" in r.stdout
    assert "builtin" in r.stdout


def test_skill_get_unknown_skill_exits_1(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_svc = SimpleNamespace(
        get_skill_metadata=lambda _: None,
        load_skill=lambda _: None,
    )
    monkeypatch.setattr("codeflow.cli.skill_commands._build_skill_service", lambda: fake_svc)

    r = runner.invoke(app, ["skills", "get", "ghost-skill"])
    assert r.exit_code == 1
    assert "Skill not found" in r.stdout


def test_skill_get_with_body_renders_markdown(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--with-body`` prints the SKILL.md body section."""
    fake_svc = SimpleNamespace(
        get_skill_metadata=lambda _: {"source": "builtin"},
        load_skill=lambda _: "# Title\nbody text",
    )
    monkeypatch.setattr("codeflow.cli.skill_commands._build_skill_service", lambda: fake_svc)

    r = runner.invoke(app, ["skills", "get", "alpha", "--with-body"])
    assert r.exit_code == 0
    assert "SKILL.md" in r.stdout
