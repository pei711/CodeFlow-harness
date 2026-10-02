from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol

from .records import VerificationState, VerifierResult


class Verifier(Protocol):
    async def verify(self, workspace: Path) -> VerifierResult: ...


@dataclass(frozen=True)
class VerifierSeal:
    path: Path
    digest: str

    @classmethod
    def capture(cls, path: Path) -> VerifierSeal:
        path = Path(path)
        return cls(path=path, digest=_file_digest(path))

    def intact(self) -> bool:
        return self.path.is_file() and _file_digest(self.path) == self.digest


@dataclass(frozen=True)
class VerifierExecution:
    result: VerifierResult
    infrastructure_error: str | None = None


@dataclass(frozen=True)
class JsonArtifactVerifier:
    expected_path: Path
    artifact_path: str
    forbidden_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_path",
            require_normalized_relative_path(
                self.artifact_path,
                field_name="artifact_path",
            ),
        )
        object.__setattr__(
            self,
            "forbidden_paths",
            tuple(
                require_normalized_relative_path(
                    path,
                    field_name="forbidden_paths",
                )
                for path in self.forbidden_paths
            ),
        )

    async def verify(self, workspace: Path) -> VerifierResult:
        workspace = Path(workspace).resolve()
        artifact = _resolve_workspace_path(workspace, self.artifact_path)
        if artifact is None:
            return VerifierResult(
                state=VerificationState.FAILED,
                findings=(f"artifact_path_outside_workspace:{self.artifact_path}",),
            )
        resolved_forbidden = tuple(
            (relative, _resolve_workspace_path(workspace, relative)) for relative in self.forbidden_paths
        )
        escaped_forbidden = tuple(relative for relative, resolved in resolved_forbidden if resolved is None)
        if escaped_forbidden:
            return VerifierResult(
                state=VerificationState.FAILED,
                findings=tuple(f"forbidden_path_outside_workspace:{path}" for path in escaped_forbidden),
            )
        forbidden = [
            relative for relative, resolved in resolved_forbidden if resolved is not None and resolved.exists()
        ]
        if forbidden:
            return VerifierResult(
                state=VerificationState.FAILED,
                findings=tuple(f"forbidden_path:{path}" for path in forbidden),
            )
        if not artifact.is_file():
            return VerifierResult(
                state=VerificationState.FAILED,
                findings=(f"missing_artifact:{self.artifact_path}",),
            )
        try:
            expected = json.loads(Path(self.expected_path).read_text())
            actual = json.loads(artifact.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return VerifierResult(
                state=VerificationState.FAILED,
                findings=(f"invalid_json:{type(exc).__name__}",),
            )
        if actual != expected:
            return VerifierResult(
                state=VerificationState.FAILED,
                findings=("artifact_mismatch",),
            )
        return VerifierResult(state=VerificationState.PASSED)


def require_normalized_relative_path(
    value: str,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(
            f"{field_name} must be a normalized relative path",
        )
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or "\\" in value
        or path.as_posix() != value
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise ValueError(
            f"{field_name} must be a normalized relative path",
        )
    return value


def _resolve_workspace_path(
    workspace: Path,
    relative: str,
) -> Path | None:
    candidate = workspace.joinpath(
        *PurePosixPath(relative).parts,
    ).resolve()
    if not candidate.is_relative_to(workspace):
        return None
    return candidate


async def run_sealed_verifier(
    verifier: Verifier,
    *,
    workspace: Path,
    seal: VerifierSeal,
) -> VerifierExecution:
    if not seal.intact():
        return VerifierExecution(
            result=VerifierResult(state=VerificationState.NOT_RUN),
            infrastructure_error="verifier_digest_changed",
        )
    try:
        result = await verifier.verify(Path(workspace))
    except BaseException as exc:
        return VerifierExecution(
            result=VerifierResult(state=VerificationState.NOT_RUN),
            infrastructure_error=f"verifier_crashed:{type(exc).__name__}",
        )
    if not seal.intact():
        return VerifierExecution(
            result=VerifierResult(state=VerificationState.NOT_RUN),
            infrastructure_error="verifier_digest_changed",
        )
    return VerifierExecution(result=result)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
