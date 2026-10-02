from __future__ import annotations

import hashlib
import platform
import re
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .canonical import CanonicalizationError, validate_plan_value

ENVIRONMENT_IDENTITY_SCHEMA = "codeflow.codeflowbench.environment.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EnvironmentIdentityError(RuntimeError):
    pass


def capture_environment_identity(
    repository_root: Path,
    *,
    sandbox_config: object | None = None,
) -> dict[str, Any]:
    lock_path = Path(repository_root) / "uv.lock"
    try:
        lock_digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise EnvironmentIdentityError(
            f"cannot seal uv.lock identity: {exc}",
        ) from exc

    backend = _configured_backend(sandbox_config)
    identity = {
        "schema": ENVIRONMENT_IDENTITY_SCHEMA,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
        },
        "hardware": {
            "machine": platform.machine(),
            "architecture": f"{struct.calcsize('P') * 8}bit",
        },
        "dependency_lock": {
            "file": "uv.lock",
            "sha256": lock_digest,
        },
        "execution": {
            "configured_backend": backend,
            "sandbox_identity": _sandbox_identity(backend),
        },
    }
    validate_environment_identity(identity)
    return identity


def validate_environment_identity(
    identity: Mapping[str, Any],
) -> None:
    try:
        validate_plan_value(identity)
    except CanonicalizationError as exc:
        raise EnvironmentIdentityError(
            f"non-canonical environment identity: {exc}",
        ) from exc
    if set(identity) != {
        "schema",
        "python",
        "os",
        "hardware",
        "dependency_lock",
        "execution",
    }:
        raise EnvironmentIdentityError(
            "environment identity fields are invalid",
        )
    if identity.get("schema") != ENVIRONMENT_IDENTITY_SCHEMA:
        raise EnvironmentIdentityError(
            "environment identity schema is invalid",
        )
    _require_string_fields(
        identity.get("python"),
        {"implementation", "version"},
        "python",
    )
    _require_string_fields(
        identity.get("os"),
        {"system", "release"},
        "os",
    )
    _require_string_fields(
        identity.get("hardware"),
        {"machine", "architecture"},
        "hardware",
    )
    dependency_lock = _require_string_fields(
        identity.get("dependency_lock"),
        {"file", "sha256"},
        "dependency_lock",
    )
    if dependency_lock["file"] != "uv.lock" or not _SHA256.fullmatch(
        dependency_lock["sha256"],
    ):
        raise EnvironmentIdentityError(
            "dependency lock identity is invalid",
        )
    execution = _require_string_fields(
        identity.get("execution"),
        {"configured_backend", "sandbox_identity"},
        "execution",
    )
    expected_sandbox = _sandbox_identity(
        execution["configured_backend"],
    )
    if execution["sandbox_identity"] != expected_sandbox:
        raise EnvironmentIdentityError(
            "execution Sandbox identity is inconsistent",
        )


def _configured_backend(sandbox_config: object | None) -> str:
    if sandbox_config is None:
        return "unresolved"
    backend = getattr(sandbox_config, "backend", None)
    if not isinstance(backend, str) or not backend:
        raise EnvironmentIdentityError(
            "configured execution backend is unavailable",
        )
    return backend


def _sandbox_identity(backend: str) -> str:
    if backend == "none":
        return "direct_host"
    if backend in {"auto", "boxlite"}:
        return "boxlite_microvm"
    if backend == "unresolved":
        return "unresolved"
    raise EnvironmentIdentityError(
        f"unsupported configured execution backend: {backend!r}",
    )


def _require_string_fields(
    value: object,
    expected: set[str],
    field_name: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise EnvironmentIdentityError(
            f"{field_name} environment identity is invalid",
        )
    normalized = dict(value)
    if any(not isinstance(item, str) or not item for item in normalized.values()):
        raise EnvironmentIdentityError(
            f"{field_name} environment identity is invalid",
        )
    return normalized
