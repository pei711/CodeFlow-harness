from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OPT_IN_MARKERS = (
    "real_llm",
    "llm_judge",
    "real_vm",
    "real_channel",
    "external_runtime",
    "e2e",
)
OPT_IN_EXPRESSION = "not (" + " or ".join(OPT_IN_MARKERS) + ")"


def test_make_test_retained_excludes_every_opt_in_marker() -> None:
    result = subprocess.run(
        ["make", "-n", "test-retained"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "pytest" in result.stdout
    assert "--frozen" in result.stdout
    assert "--all-extras" in result.stdout
    assert "--exact" in result.stdout
    assert "--strict-markers" in result.stdout
    assert OPT_IN_EXPRESSION in result.stdout


def test_pytest_defaults_require_explicit_opt_in() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    addopts = config["tool"]["pytest"]["ini_options"]["addopts"]

    assert addopts == ["-m", OPT_IN_EXPRESSION]


def test_retained_tests_use_an_isolated_home(tmp_path: Path) -> None:
    from codeflow.config.loader import get_config_path
    from tests.conftest import _is_external_environment

    isolated_home = tmp_path / "home"
    assert Path.home() == isolated_home
    assert get_config_path() == isolated_home / ".codeflow" / "config.json"
    assert _is_external_environment("ANTHROPIC_API_KEY_CODEFLOW")
    assert _is_external_environment("OPENAI_API_KEY_CODEFLOW")
    assert _is_external_environment("SSH_AUTH_SOCK")
    assert _is_external_environment("HTTPS_PROXY")
    assert not _is_external_environment("GIT_AUTHOR_NAME")
    assert not [name for name in os.environ if _is_external_environment(name)]


def test_exception_ledger_is_empty_and_cannot_waive_v_d0() -> None:
    ledger = tomllib.loads((REPO_ROOT / "tests" / "fixtures" / "exception-ledger.toml").read_text(encoding="utf-8"))

    assert ledger["policy"]["v_d0_exceptions_allowed"] is False
    assert ledger.get("diagnostic", []) == []
    assert ledger.get("dependency", []) == []
