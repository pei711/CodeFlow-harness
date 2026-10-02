import ast
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_python_distribution_exposes_only_the_codeflow_namespace() -> None:
    import codeflow

    assert codeflow.__version__ != "0.0.0+unknown"


def test_public_distribution_cli_and_plugin_identity() -> None:
    from typer.testing import CliRunner

    from codeflow.cli.commands import app

    repo_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert metadata["name"] == "codeflow-harness"
    assert metadata["scripts"] == {"codeflow": "codeflow.cli.commands:run"}

    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    version_result = runner.invoke(app, ["--version"])
    assert help_result.exit_code == 0
    assert "CodeFlow" in help_result.stdout
    assert version_result.exit_code == 0
    assert "CodeFlow v" in version_result.stdout

    assert not (repo_root / "codeflow" / "plugin" / "memory" / "everos").exists()
    assert not (repo_root / "codeflow" / "config" / "update_everos.py").exists()


def test_distribution_has_no_unpublished_or_retired_memory_dependency() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads(
        (repo_root / "pyproject.toml").read_text(encoding="utf-8"),
    )

    dependencies = metadata["project"]["dependencies"]
    assert not any("codecairn" in dependency.lower() for dependency in dependencies)
    assert not any("myna" in dependency.lower() for dependency in dependencies)
    assert not any("file://" in dependency.lower() for dependency in dependencies)


def test_retired_memory_implementation_is_absent_from_executable_surfaces() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    retired = "code" + "cairn"

    benchmark_files = tuple(
        path
        for path in (repo_root / "benchmarks").rglob("*")
        if path.is_file() and path.suffix in {".json", ".md", ".py", ".yaml"}
    )
    assert not any(retired in path.as_posix().lower() for path in benchmark_files)
    assert not any(retired in path.read_text(encoding="utf-8").lower() for path in benchmark_files)

    retired_test_consumers = [path for path in (repo_root / "tests").rglob("test_*.py") if retired in path.name.lower()]
    assert retired_test_consumers == []

    makefile = (repo_root / "Makefile").read_text(encoding="utf-8")
    assert "CODEFLOW_SCORECARD_MEMORY_" not in makefile
    assert "--memory-summary" not in makefile
    assert "--memory-handoff" not in makefile


def test_codeflow_only_imports_myna_through_its_public_plugin_surface() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for root in (repo_root / "codeflow", repo_root / "scripts"):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("myna.") and alias.name != "myna.integrations.codeflow":
                            violations.append(f"{path.relative_to(repo_root)}:{node.lineno}:{alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imported = {alias.name for alias in node.names}
                    public_integration = module == "myna.integrations.codeflow" and imported <= {
                        "descriptor",
                        "open_codeflow_integration",
                    }
                    if module.startswith("myna") and not public_integration:
                        violations.append(f"{path.relative_to(repo_root)}:{node.lineno}:{module}")

    assert violations == []


def test_clean_defaults_use_codeflow_state(tmp_path: Path, monkeypatch) -> None:
    from codeflow.config import loader
    from codeflow.config.paths import get_workspace_path

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(loader, "_current_config_path", None)

    assert loader.get_config_path() == tmp_path / ".codeflow" / "config.json"
    assert loader.load_config().agents.defaults.workspace == "~/.codeflow/workspace"
    assert get_workspace_path() == tmp_path / ".codeflow" / "workspace"


def test_foreground_runtime_uses_current_directory_without_polluting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codeflow.config.paths import resolve_foreground_paths
    from codeflow.config.schema import Config

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("CODEFLOW_HOME", str(tmp_path / "codeflow-home"))
    monkeypatch.chdir(project)

    paths = resolve_foreground_paths(Config())

    assert paths.workspace == project.resolve()
    assert paths.state.parent == tmp_path / "codeflow-home" / "projects"
    assert paths.state.name.startswith("project-")
    assert list(project.iterdir()) == []


def test_foreground_runtime_preserves_explicit_workspace(
    tmp_path: Path,
) -> None:
    from codeflow.config.paths import resolve_foreground_paths
    from codeflow.config.schema import Config

    workspace = tmp_path / "explicit-workspace"
    paths = resolve_foreground_paths(Config(), workspace=str(workspace))

    assert paths.workspace == workspace
    assert paths.state == workspace


def test_project_state_identity_is_stable_and_workspace_specific(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codeflow.product import get_project_state_dir

    monkeypatch.setenv("CODEFLOW_HOME", str(tmp_path / "codeflow-home"))
    first = tmp_path / "first" / "project"
    second = tmp_path / "second" / "project"

    assert get_project_state_dir(first) == get_project_state_dir(first)
    assert get_project_state_dir(first) != get_project_state_dir(second)
    assert get_project_state_dir(first).parent == tmp_path / "codeflow-home" / "projects"


def test_foreground_runtime_preserves_configured_workspace(
    tmp_path: Path,
) -> None:
    from codeflow.config.paths import resolve_foreground_paths
    from codeflow.config.schema import Config

    workspace = tmp_path / "configured-workspace"
    config = Config()
    config.agents.defaults.workspace = str(workspace)

    paths = resolve_foreground_paths(config)

    assert paths.workspace == workspace
    assert paths.state == workspace


def test_service_runtime_keeps_configured_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codeflow.config.paths import resolve_service_paths
    from codeflow.config.schema import Config

    product_home = tmp_path / "codeflow-home"
    monkeypatch.setenv("CODEFLOW_HOME", str(product_home))

    paths = resolve_service_paths(Config())

    assert paths.workspace == product_home / "workspace"
    assert paths.state == product_home / "workspace"


def test_codeflow_home_scopes_global_runtime_state(tmp_path: Path, monkeypatch) -> None:
    product_home = tmp_path / "state"
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("CODEFLOW_HOME", str(product_home))
    monkeypatch.chdir(project)

    from codeflow.cli._plugin_stack import plugin_discovery_sources
    from codeflow.config import loader
    from codeflow.config.paths import get_cli_history_path, get_cron_dir, get_workspace_path
    from codeflow.plugin import PluginDiscovery
    from codeflow.routing.cache import BenchmarkCache
    from codeflow.routing.knn_router import KNNModelRouter
    from codeflow.token_wise.usage_tracker import UsageTracker
    from codeflow.tracing.config import state_dir

    monkeypatch.setattr(loader, "_current_config_path", None)
    routing_config = SimpleNamespace(
        k=1,
        lambda_cost=0.0,
        embedding_endpoint=None,
        models=[],
        min_similarity=0.0,
        min_similar_neighbors=1,
        min_memory_size=1,
        min_margin=0.0,
        memory_path="",
    )

    assert loader.get_config_path() == product_home / "config.json"
    assert loader.load_config().workspace_path == product_home / "workspace"
    assert get_workspace_path() == product_home / "workspace"
    assert get_cli_history_path() == product_home / ".codeflow_history"
    assert get_cron_dir() == product_home / "cron"
    assert state_dir() == product_home / "traces"
    assert UsageTracker(persist=False).telemetry_dir == product_home / "telemetry"
    assert BenchmarkCache()._cache_path == product_home / "routing" / "benchmark-cache.json"
    assert KNNModelRouter(routing_config)._emb_cache_path("tasks.json").parent == product_home / "knn_embcache"
    sources = plugin_discovery_sources()
    assert sources["user_dir"] == product_home / "plugins"
    assert sources["project_dir"] is None
    assert sources["entry_points_group"] == "codeflow.plugins"
    discovered = PluginDiscovery(
        user_dir=sources["user_dir"],
        project_dir=sources["project_dir"],
        entry_points_group=None,
    ).discover()
    assert discovered == []


def test_blank_codeflow_home_uses_default_state_root(tmp_path: Path, monkeypatch) -> None:
    from codeflow.product import get_product_home

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("CODEFLOW_HOME", "  ")

    assert get_product_home() == tmp_path / ".codeflow"


@pytest.mark.asyncio
async def test_checkpoint_uses_codeflow_state(tmp_path: Path) -> None:
    from codeflow.agent.loop.checkpoint import CheckpointService

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "task.txt").write_text("retained workspace content\n", encoding="utf-8")

    checkpoint_id, changed = await CheckpointService(workspace).commit_turn("identity isolation")

    assert checkpoint_id is not None
    assert changed == ["task.txt"]
    assert (workspace / ".codeflow" / "shadow.git" / "HEAD").is_file()
