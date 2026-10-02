from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

_FIXTURE_BENCH = Path(__file__).with_name("_evolver_process_bench.py")
_INTERRUPT_ROUND_ENV = "EVOLVER_FIXTURE_INTERRUPT_ROUND"


def _run(
    config_path: Path,
    cwd: Path,
    command: str,
    *args: str,
    interrupt_round: int | None = None,
) -> subprocess.CompletedProcess[str]:
    cwd.mkdir()
    env = dict(os.environ)
    if interrupt_round is None:
        env.pop(_INTERRUPT_ROUND_ENV, None)
    else:
        env[_INTERRUPT_ROUND_ENV] = str(interrupt_round)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "codeflow.cli.commands",
            "evolve",
            command,
            "--config",
            str(config_path),
            *args,
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _create_subject_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "subject"
    entry_dir = repo / "benchmarks" / "appworld" / "evolve"
    entry_dir.mkdir(parents=True)
    for package_dir in (
        repo / "benchmarks",
        repo / "benchmarks" / "appworld",
        entry_dir,
    ):
        (package_dir / "__init__.py").write_text("")
    shutil.copyfile(_FIXTURE_BENCH, entry_dir / "entry.py")
    source = repo / "src" / "subject.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n")

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "CodeFlow Test",
        "GIT_AUTHOR_EMAIL": "codeflow-test@example.invalid",
        "GIT_COMMITTER_NAME": "CodeFlow Test",
        "GIT_COMMITTER_EMAIL": "codeflow-test@example.invalid",
    }
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "fixture"],
    ):
        subprocess.run(
            command,
            cwd=repo,
            env=env,
            capture_output=True,
            check=True,
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return repo, revision


def _write_config(
    tmp_path: Path,
    repo: Path,
    revision: str,
) -> tuple[Path, Path]:
    work_dir = tmp_path / "evolution-run"
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "bench": "appworld",
                "repo_root": str(repo),
                "base_sha": revision,
                "work_dir": str(work_dir),
                "models": {
                    "driver": {
                        "provider": "openai_compat",
                        "base_url": "http://127.0.0.1:1/v1",
                        "model": "unused-offline-fixture",
                        "timeout": 0.05,
                        "retry_delays": [],
                    }
                },
                "funnel": {
                    "k_confirm": 1,
                    "budget": {
                        "max_why_per_round": 1,
                        "candidates_per_why": 1,
                        "recombinations_per_round": 0,
                    },
                    "termination": {
                        "patience": 10,
                        "max_rounds": 3,
                    },
                },
            }
        )
    )
    return config_path, work_dir


def test_evolution_run_resumes_statuses_and_finalizes_across_processes(
    tmp_path: Path,
) -> None:
    repo, revision = _create_subject_repo(tmp_path)
    config_path, work_dir = _write_config(tmp_path, repo, revision)

    started = _run(
        config_path,
        tmp_path / "process-start",
        "run",
        interrupt_round=2,
    )
    assert started.returncode == 130, (started.stdout, started.stderr)
    journal_path = work_dir / "journal" / "rounds.jsonl"
    assert len(journal_path.read_text().splitlines()) == 1
    assert json.loads((work_dir / "run_meta.json").read_text())["unsealed_at"] is None

    resumed = _run(
        config_path,
        tmp_path / "process-resume",
        "run",
        interrupt_round=3,
    )
    assert resumed.returncode == 130, (resumed.stdout, resumed.stderr)
    assert len(journal_path.read_text().splitlines()) == 2

    status = _run(
        config_path,
        tmp_path / "process-status",
        "status",
    )
    assert status.returncode == 0, (status.stdout, status.stderr)
    assert "phase 2: 2 completed round(s)" in status.stdout
    assert "retention" not in status.stdout.lower()
    summary_path = work_dir / "evolution_summary.json"
    summary_before_finalize = summary_path.read_bytes()
    summary = json.loads(summary_before_finalize)
    assert summary["outcome_counts"] == {
        "accepted": 0,
        "rejected": 2,
        "failed": 0,
        "inconclusive": 0,
    }

    repeated_status = _run(
        config_path,
        tmp_path / "process-status-repeated",
        "status",
    )
    assert repeated_status.returncode == 0, (repeated_status.stdout, repeated_status.stderr)
    assert summary_path.read_bytes() == summary_before_finalize

    finalized = _run(
        config_path,
        tmp_path / "process-finalize",
        "finalize",
        "--yes",
    )
    assert finalized.returncode == 0, (finalized.stdout, finalized.stderr)
    assert json.loads((work_dir / "retention.json").read_text()) == {
        "best_round": 2,
        "retention": 1.0,
    }
    run_meta = json.loads((work_dir / "run_meta.json").read_text())
    assert run_meta["unsealed_at"]
    assert run_meta["finalize_reason"] == "user_finalized"
    assert summary_path.read_bytes() == summary_before_finalize

    repeated_finalize = _run(
        config_path,
        tmp_path / "process-finalize-repeated",
        "finalize",
        "--yes",
    )
    assert repeated_finalize.returncode == 0, (
        repeated_finalize.stdout,
        repeated_finalize.stderr,
    )
    assert summary_path.read_bytes() == summary_before_finalize
