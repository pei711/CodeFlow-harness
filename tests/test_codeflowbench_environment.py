from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from benchmarks.codeflowbench.campaign import (
    CampaignMode,
    _experiment_spec,
    load_campaign_suite,
)
from benchmarks.codeflowbench.canonical import canonical_digest, canonical_json
from benchmarks.codeflowbench.environment import (
    EnvironmentIdentityError,
    capture_environment_identity,
    validate_environment_identity,
)
from codeflow.sandbox.config import SandboxConfig

_SUITE_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "codeflowbench" / "suites" / "agent_application_ship_1.yaml"
)


def test_environment_identity_is_stable_and_excludes_machine_paths(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    lock_bytes = b"version = 1\nrevision = 3\n"
    (first_root / "uv.lock").write_bytes(lock_bytes)
    (second_root / "uv.lock").write_bytes(lock_bytes)
    host_mount = tmp_path / "private-host-volume"
    sandbox = SandboxConfig(
        backend="boxlite",
        extra_volumes=[
            [str(host_mount), "/mnt/data", "ro"],
        ],
    )

    first = capture_environment_identity(
        first_root,
        sandbox_config=sandbox,
    )
    second = capture_environment_identity(
        second_root,
        sandbox_config=sandbox,
    )
    serialized = canonical_json(first)

    assert first == second
    assert first["dependency_lock"] == {
        "file": "uv.lock",
        "sha256": hashlib.sha256(lock_bytes).hexdigest(),
    }
    assert first["execution"] == {
        "configured_backend": "boxlite",
        "sandbox_identity": "boxlite_microvm",
    }
    assert first["python"]["implementation"]
    assert first["python"]["version"]
    assert first["os"]["system"]
    assert first["os"]["release"]
    assert first["hardware"]["machine"]
    assert first["hardware"]["architecture"]
    assert str(first_root) not in serialized
    assert str(second_root) not in serialized
    assert str(host_mount) not in serialized
    assert "/mnt/data" not in serialized


def test_environment_identity_changes_when_uv_lock_changes(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text("version = 1\n", encoding="utf-8")
    before = capture_environment_identity(tmp_path)

    lock_path.write_text("version = 2\n", encoding="utf-8")
    after = capture_environment_identity(tmp_path)

    assert before != after
    assert before["dependency_lock"]["sha256"] != after["dependency_lock"]["sha256"]


def test_environment_identity_rejects_missing_uv_lock(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        EnvironmentIdentityError,
        match="cannot seal uv.lock identity",
    ):
        capture_environment_identity(tmp_path)


def test_environment_identity_rejects_absolute_machine_path() -> None:
    with pytest.raises(
        EnvironmentIdentityError,
        match="non-canonical environment identity",
    ):
        validate_environment_identity(
            {
                "schema": "codeflow.codeflowbench.environment.v1",
                "python": {
                    "implementation": "CPython",
                    "version": "3.13.0",
                },
                "os": {
                    "system": "Darwin",
                    "release": "25.0",
                },
                "hardware": {
                    "machine": "/private/host",
                    "architecture": "64bit",
                },
                "dependency_lock": {
                    "file": "uv.lock",
                    "sha256": "a" * 64,
                },
                "execution": {
                    "configured_backend": "none",
                    "sandbox_identity": "direct_host",
                },
            },
        )


def test_environment_identity_changes_canonical_experiment_identity(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (second_root / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    suite = load_campaign_suite(_SUITE_PATH)
    common = {
        "suite": suite,
        "mode": CampaignMode.CALIBRATION,
        "suite_digest": "a" * 64,
        "claim_rules_digest": "b" * 64,
        "runtime_config_digest": "c" * 64,
        "codeflow_commit": "d" * 40,
        "worktree_clean": True,
    }
    first_environment = capture_environment_identity(first_root)
    second_environment = capture_environment_identity(second_root)

    first = _experiment_spec(
        **common,
        output_root=tmp_path / "output-a",
        environment_identity=first_environment,
    )
    same = _experiment_spec(
        **common,
        output_root=tmp_path / "output-b",
        environment_identity=first_environment,
    )
    drifted = _experiment_spec(
        **common,
        output_root=tmp_path / "output-a",
        environment_identity=second_environment,
    )

    assert canonical_digest(first.canonical_payload()) == canonical_digest(
        same.canonical_payload(),
    )
    assert canonical_digest(first.canonical_payload()) != canonical_digest(
        drifted.canonical_payload(),
    )
