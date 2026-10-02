from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_PROBE = Path(__file__).with_name("_session_continuity_probe.py")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _probe(workspace: Path, action: str, *args: str) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(_PROBE),
            action,
            "--workspace",
            str(workspace),
            *args,
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_session_lifecycle_survives_processes_and_isolates_forks(
    tmp_path: Path,
) -> None:
    product_home = tmp_path / "codeflow-home"
    workspace = product_home / "workspace"
    seeded = _probe(workspace, "seed")
    diverged = _probe(
        workspace,
        "diverge",
        "--parent-key",
        seeded["parent_key"],
        "--child-key",
        seeded["child_key"],
    )
    verified = _probe(
        workspace,
        "verify",
        "--parent-key",
        seeded["parent_key"],
        "--child-key",
        seeded["child_key"],
        "--export-path",
        seeded["export_path"],
    )

    assert diverged["parent_contents_before_delete"] == [
        "shared question",
        "shared answer",
        "parent only",
    ]
    assert verified == {
        "child_contents": [
            "shared question",
            "shared answer",
            "child only",
        ],
        "child_parent": seeded["parent_key"],
        "export_verified": True,
        "parent_not_found": True,
    }

    resume = subprocess.run(
        [
            sys.executable,
            "-m",
            "codeflow.cli.commands",
            "sessions",
            "resume",
            seeded["parent_key"].partition(":")[2],
        ],
        cwd=_REPO_ROOT,
        env={**os.environ, "CODEFLOW_HOME": str(product_home)},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert resume.returncode == 1
    assert "No session matching id/prefix" in resume.stdout
