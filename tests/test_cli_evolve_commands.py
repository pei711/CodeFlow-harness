from __future__ import annotations

from typer.testing import CliRunner

from codeflow.cli.commands import app

runner = CliRunner()


def test_evolve_help_uses_existing_launcher() -> None:
    result = runner.invoke(app, ["evolve", "--help"])

    assert result.exit_code == 0
    assert "Beta" in result.stdout
    assert "opt-in" in result.stdout
    assert "manual activation" in result.stdout
    assert "start or resume an evolution run" in result.stdout
    assert "validate config/models/bench setup" in result.stdout
    assert "terminate now and unseal" in result.stdout
    assert "resume" not in {
        line.split()[0] for line in result.stdout.splitlines() if line.startswith("  ") and line.strip()
    }


def test_evolve_forwards_arguments_to_existing_launcher(monkeypatch) -> None:
    calls: list[list[str] | None] = []

    def _main(argv: list[str] | None = None) -> int:
        calls.append(argv)
        return 7

    monkeypatch.setattr("codeflow.evolver.cli.main", _main)

    result = runner.invoke(app, ["evolve", "check", "--config", "run.yaml", "--smoke"])

    assert result.exit_code == 7
    assert calls == [["check", "--config", "run.yaml", "--smoke"]]


def test_evolve_exposes_only_the_approved_commands() -> None:
    help_result = runner.invoke(app, ["evolve", "--help"])
    resume_result = runner.invoke(app, ["evolve", "resume", "--config", "run.yaml"])

    assert "{run,check,status,finalize}" in help_result.stdout
    assert resume_result.exit_code == 2
    assert "invalid choice: 'resume'" in resume_result.output
