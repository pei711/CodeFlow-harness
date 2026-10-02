from __future__ import annotations

import argparse
import configparser
import hashlib
import http.client
import json
import os
import posixpath
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import Parser
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]


class VerificationError(RuntimeError):
    pass


class _GatewayProbeError(RuntimeError):
    def __init__(self, outcome: str, detail: str) -> None:
        self.outcome = outcome
        super().__init__(detail)


@dataclass(frozen=True)
class WheelSnapshot:
    path: Path
    files: dict[str, str]
    metadata: str
    entry_points: str
    dist_info: str


@dataclass(frozen=True)
class InstallPlan:
    name: str
    root: Path
    requirement: str
    required_imports: tuple[str, ...]
    forbidden_imports: tuple[str, ...] = ()


_REQUIRED_FILES = {
    "codeflow/__init__.py",
    "codeflow/templates/AGENTS.md",
    "codeflow/templates/SOUL.md",
    "codeflow/tracing/viewer/server.js",
    "codeflow/tracing/viewer/ui/app.js",
}
_RETIRED_MEMORY_DEPENDENCIES = ("codecairn", "everalgo", "everos")

_EXTRA_IMPORTS = {
    "channel-feishu": ("lark_oapi",),
    "channel-qq": ("botpy",),
    "channel-wecom": ("wecom_aibot_sdk",),
    "channels": ("lark_oapi", "botpy", "wecom_aibot_sdk"),
    "sandbox": ("boxlite",),
}
_OPTIONAL_IMPORTS = tuple(sorted({module for modules in _EXTRA_IMPORTS.values() for module in modules}))
_REQUIRED_INSTALL_TARGETS = (
    "base",
    "channel-feishu",
    "channel-qq",
    "channel-wecom",
    "channels",
    "sandbox",
)
_RETAINED_CHANNEL_EXTRAS = frozenset({"channel-feishu", "channel-qq", "channel-wecom"})
_REMOVED_CHANNELS = frozenset(
    {"telegram", "slack", "discord", "whatsapp", "matrix", "mochat", "dingtalk", "email", "weixin"}
)
_REMOVED_CHANNEL_EXTRAS = frozenset(f"channel-{name}" for name in _REMOVED_CHANNELS)
_REMOVED_CHANNEL_DEPENDENCIES = (
    "dingtalk-stream",
    "matrix-nio",
    "mistune",
    "msgpack",
    "nh3",
    "python-socketio",
    "python-socks",
    "python-telegram-bot",
    "qrcode",
    "slack-sdk",
    "slackify-markdown",
    "socksio",
    "websocket-client",
    "websockets",
)
_PUBLIC_CLI_COMMANDS = frozenset(
    {
        "channels",
        "cron",
        "doctor",
        "evolve",
        "gateway",
        "onboard",
        "plugins",
        "provider",
        "run",
        "sessions",
        "skills",
        "status",
        "tracing",
    }
)
_REGISTERED_CLI_COMMANDS = _PUBLIC_CLI_COMMANDS | {"sandbox"}
_TUI_PROBE_TIMEOUT_SECONDS = 30
_ENV_ALLOWLIST = {
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "NIX_SSL_CERT_FILE",
    "PATH",
    "PATHEXT",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "WINDIR",
}


def _required_attribution_files() -> tuple[str, ...]:
    files = ["LICENSE", "NOTICES.md"]
    licenses = REPO_ROOT / "LICENSES"
    if licenses.is_dir():
        files.extend(path.relative_to(REPO_ROOT).as_posix() for path in sorted(licenses.iterdir()) if path.is_file())
    return tuple(files)


def _prepare_output_root(output_root: Path) -> Path:
    resolved = output_root.expanduser().resolve()
    repo = REPO_ROOT.resolve()
    if resolved == repo or repo in resolved.parents:
        raise VerificationError("output root must be outside the source checkout")
    if resolved.exists():
        if not resolved.is_dir():
            raise VerificationError("output root must be a directory")
        if any(resolved.iterdir()):
            raise VerificationError("output root must be empty")
    else:
        resolved.mkdir(parents=True)
    return resolved


def _wheel_snapshot(wheel: Path) -> WheelSnapshot:
    try:
        with zipfile.ZipFile(wheel) as archive:
            if corrupt := archive.testzip():
                raise VerificationError(f"wheel CRC check failed for {corrupt}")
            infos = [info for info in archive.infolist() if not info.is_dir()]
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise VerificationError("wheel contains duplicate paths")
            for name in names:
                path = PurePosixPath(name)
                if name.startswith(("/", "\\")) or "\\" in name or ".." in path.parts:
                    raise VerificationError(f"wheel contains unsafe path: {name}")
            metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata_paths) != 1:
                raise VerificationError("wheel must contain exactly one METADATA file")
            dist_info = metadata_paths[0].removesuffix("/METADATA")
            entry_points_path = f"{dist_info}/entry_points.txt"
            if entry_points_path not in names:
                raise VerificationError("wheel is missing entry_points.txt")
            files = {info.filename: hashlib.sha256(archive.read(info)).hexdigest() for info in infos}
            metadata = archive.read(metadata_paths[0]).decode("utf-8")
            entry_points = archive.read(entry_points_path).decode("utf-8")
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise VerificationError(f"cannot inspect wheel {wheel}: {exc}") from exc
    return WheelSnapshot(
        path=wheel,
        files=files,
        metadata=metadata,
        entry_points=entry_points,
        dist_info=dist_info,
    )


def _validate_wheel(snapshot: WheelSnapshot, entrypoint: str) -> None:
    names = set(snapshot.files)
    forbidden = [
        name
        for name in names
        if "node_modules" in PurePosixPath(name).parts
        or "__pycache__" in PurePosixPath(name).parts
        or name.endswith((".pyc", ".pyo"))
    ]
    if forbidden:
        raise VerificationError(f"wheel contains forbidden files such as {sorted(forbidden)[0]}")

    tui_files = {name for name in names if name.startswith("codeflow/ui-tui/")}
    expected_tui = {"codeflow/ui-tui/dist/entry.js"}
    if tui_files != expected_tui:
        raise VerificationError(f"wheel must contain exactly one TUI bundle, found {sorted(tui_files)}")

    removed_channel_runtime = sorted(name for name in names if _is_removed_channel_path(name))
    if removed_channel_runtime:
        raise VerificationError(f"wheel contains removed Channel runtime: {removed_channel_runtime[0]}")
    removed_media_generation_runtime = sorted(name for name in names if _is_removed_media_generation_path(name))
    if removed_media_generation_runtime:
        raise VerificationError(
            f"wheel contains removed media generation runtime: {removed_media_generation_runtime[0]}"
        )
    removed_skill_hub_runtime = sorted(name for name in names if _is_removed_skill_hub_path(name))
    if removed_skill_hub_runtime:
        raise VerificationError(f"wheel contains removed Skill Hub runtime: {removed_skill_hub_runtime[0]}")
    removed_deep_research_runtime = sorted(name for name in names if _is_removed_deep_research_path(name))
    if removed_deep_research_runtime:
        raise VerificationError(f"wheel contains removed Deep Research runtime: {removed_deep_research_runtime[0]}")
    removed_sentinel_runtime = sorted(name for name in names if _is_removed_sentinel_path(name))
    if removed_sentinel_runtime:
        raise VerificationError(f"wheel contains removed Sentinel runtime: {removed_sentinel_runtime[0]}")
    removed_heartbeat_runtime = sorted(name for name in names if _is_removed_heartbeat_path(name))
    if removed_heartbeat_runtime:
        raise VerificationError(f"wheel contains removed Heartbeat runtime: {removed_heartbeat_runtime[0]}")
    removed_cli_runtime = sorted(name for name in names if _is_removed_cli_path(name))
    if removed_cli_runtime:
        raise VerificationError(f"wheel contains removed CLI runtime: {removed_cli_runtime[0]}")
    removed_everos_runtime = sorted(
        name
        for name in names
        if name.startswith("codeflow/plugin/memory/everos/")
        or name
        in {
            "codeflow/config/update_everos.py",
            "codeflow/memory_engine/skill_forge/everos_source.py",
            "codeflow/tracing/viewer/everos-deposits.js",
        }
    )
    if removed_everos_runtime:
        raise VerificationError(f"wheel contains removed EverOS runtime: {removed_everos_runtime[0]}")

    local_artifacts = sorted(name for name in names if _is_local_build_or_report_artifact(name))
    if local_artifacts:
        raise VerificationError(f"wheel contains local build or report artifact: {local_artifacts[0]}")

    missing_required = sorted(_REQUIRED_FILES - names)
    if missing_required:
        raise VerificationError(f"wheel is missing required resources: {missing_required}")
    required_attribution = {f"{snapshot.dist_info}/licenses/{name}" for name in _required_attribution_files()}
    missing_attribution = sorted(required_attribution - names)
    if missing_attribution:
        raise VerificationError(f"wheel is missing required attribution files: {missing_attribution}")

    metadata = Parser().parsestr(snapshot.metadata)
    if metadata.get("Name", "").lower() != "codeflow-harness":
        raise VerificationError("wheel METADATA does not describe the codeflow-harness distribution")
    provided_extras = set(metadata.get_all("Provides-Extra", []))
    if removed_extras := sorted(provided_extras.intersection(_REMOVED_CHANNEL_EXTRAS)):
        raise VerificationError(f"wheel advertises removed Channel extra: {removed_extras[0]}")
    channel_extras = {name for name in provided_extras if name.startswith("channel-")}
    if channel_extras != _RETAINED_CHANNEL_EXTRAS:
        raise VerificationError(
            f"wheel must advertise exactly the retained Channel extras: {sorted(_RETAINED_CHANNEL_EXTRAS)}"
        )
    requirements = tuple(value.lower() for value in metadata.get_all("Requires-Dist", []))
    for package in _RETIRED_MEMORY_DEPENDENCIES:
        if any(re.match(rf"^{package}(?:\W|$)", requirement) for requirement in requirements):
            raise VerificationError(f"wheel depends directly on removed package: {package}")
    if any(re.match(r"^myna-memory(?:\W|$)", requirement) for requirement in requirements):
        raise VerificationError("wheel depends on unpublished myna-memory distribution")
    if any(" @ file:" in requirement or " @ /" in requirement for requirement in requirements):
        raise VerificationError("wheel contains a local filesystem dependency")
    for package in _REMOVED_CHANNEL_DEPENDENCIES:
        if any(re.match(rf"^{re.escape(package)}(?:\W|$)", requirement) for requirement in requirements):
            raise VerificationError(f"wheel depends on removed Channel package: {package}")

    parser = configparser.ConfigParser()
    parser.read_string(snapshot.entry_points)
    expected_target = "codeflow.cli.commands:run"
    if parser.get("console_scripts", entrypoint, fallback="").strip() != expected_target:
        raise VerificationError(f"console entrypoint {entrypoint!r} does not target {expected_target}")


def _is_local_build_or_report_artifact(name: str) -> bool:
    path = PurePosixPath(name)
    parts = tuple(part.lower() for part in path.parts)
    basename = parts[-1]
    suffix = path.suffix.lower()
    node_source = parts[:1] == ("ui-tui",) or suffix in {
        ".cts",
        ".jsx",
        ".map",
        ".mts",
        ".ts",
        ".tsx",
    }
    report_asset = (
        any(part in {"coverage", "htmlcov", "reports", "test-results"} for part in parts)
        or basename
        in {
            ".coverage",
            "coverage.json",
            "coverage.xml",
            "distribution-report.json",
            "junit.xml",
            "lcov.info",
            "pytest.xml",
        }
        or ("report" in path.stem.lower() and suffix in {".htm", ".html", ".json", ".pdf", ".png", ".svg", ".xml"})
    )
    return (
        basename in {"package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock", ".npmrc"}
        or basename.startswith(("npm-debug", "yarn-debug", "yarn-error"))
        or node_source
        or report_asset
    )


def _is_removed_channel_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    candidates = (parts, parts[1:])
    return any(
        candidate[:1] == ("bridge",)
        or (
            candidate[:3] == ("codeflow", "channels", "adapters")
            and len(candidate) > 3
            and candidate[3] in _REMOVED_CHANNELS
        )
        for candidate in candidates
    )


def _is_removed_media_generation_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return ("codeflow", "agent", "tools", "media_gen.py") in (parts, parts[1:])


def _is_removed_skill_hub_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    candidates = (parts, parts[1:])
    return any(
        candidate[:2] == ("codeflow", "skill_hub")
        or candidate == ("codeflow", "agent", "tools", "skill_hub.py")
        or candidate == ("codeflow", "memory_engine", "skill_forge", "hub_source.py")
        for candidate in candidates
    )


def _is_removed_deep_research_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    candidates = (parts, parts[1:])
    return any(
        candidate
        in {
            ("codeflow", "agent", "tools", "deep_research.py"),
            ("codeflow", "cli", "deep_research_commands.py"),
            ("codeflow", "config", "update_tools.py"),
        }
        for candidate in candidates
    )


def _is_removed_sentinel_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    candidates = (parts, parts[1:])
    return any(
        candidate[:3] == ("codeflow", "proactive_engine", "sentinel")
        or candidate
        in {
            ("codeflow", "cli", "sentinel_commands.py"),
            ("codeflow", "cli", "_proactive_stack.py"),
            ("codeflow", "memory_engine", "consolidate", "attention.py"),
            ("codeflow", "memory_engine", "consolidate", "behaviors.py"),
            ("codeflow", "memory_engine", "consolidate", "behaviors_extractor.py"),
        }
        for candidate in candidates
    )


def _is_removed_heartbeat_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    candidates = (parts, parts[1:])
    return any(
        candidate[:4] == ("codeflow", "proactive_engine", "schedulers", "heartbeat")
        or candidate
        in {
            ("codeflow", "proactive_engine", "system_events.py"),
            ("codeflow", "proactive_engine", "wake.py"),
            ("codeflow", "templates", "HEARTBEAT.md"),
        }
        for candidate in candidates
    )


def _is_removed_cli_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return ("codeflow", "cli", "upgrade_commands.py") in (parts, parts[1:])


def _validate_sdist(sdist: Path) -> None:
    try:
        with tarfile.open(sdist, "r:gz") as archive:
            members = archive.getmembers()
            files = {
                PurePosixPath(*PurePosixPath(member.name).parts[1:]).as_posix()
                for member in members
                if member.isfile() and len(PurePosixPath(member.name).parts) > 1
            }
            removed = sorted(member.name for member in members if _is_removed_channel_path(member.name))
            removed_media_generation = sorted(
                member.name for member in members if _is_removed_media_generation_path(member.name)
            )
            removed_skill_hub = sorted(member.name for member in members if _is_removed_skill_hub_path(member.name))
            removed_deep_research = sorted(
                member.name for member in members if _is_removed_deep_research_path(member.name)
            )
            removed_sentinel = sorted(member.name for member in members if _is_removed_sentinel_path(member.name))
            removed_heartbeat = sorted(member.name for member in members if _is_removed_heartbeat_path(member.name))
            removed_cli = sorted(member.name for member in members if _is_removed_cli_path(member.name))
    except (OSError, tarfile.TarError) as exc:
        raise VerificationError(f"cannot inspect sdist {sdist}: {exc}") from exc
    if removed:
        raise VerificationError(f"sdist contains removed Channel runtime: {removed[0]}")
    if removed_media_generation:
        raise VerificationError(f"sdist contains removed media generation runtime: {removed_media_generation[0]}")
    if removed_skill_hub:
        raise VerificationError(f"sdist contains removed Skill Hub runtime: {removed_skill_hub[0]}")
    if removed_deep_research:
        raise VerificationError(f"sdist contains removed Deep Research runtime: {removed_deep_research[0]}")
    if removed_sentinel:
        raise VerificationError(f"sdist contains removed Sentinel runtime: {removed_sentinel[0]}")
    if removed_heartbeat:
        raise VerificationError(f"sdist contains removed Heartbeat runtime: {removed_heartbeat[0]}")
    if removed_cli:
        raise VerificationError(f"sdist contains removed CLI runtime: {removed_cli[0]}")
    missing_attribution = sorted(set(_required_attribution_files()) - files)
    if missing_attribution:
        raise VerificationError(f"sdist is missing required attribution files: {missing_attribution}")


def _build_install_plans(
    environments_root: Path,
    wheel: Path,
    extras: tuple[str, ...],
    snapshot: WheelSnapshot,
) -> list[InstallPlan]:
    if extras != _REQUIRED_INSTALL_TARGETS:
        raise VerificationError(f"extras must be exactly the canonical targets: {_REQUIRED_INSTALL_TARGETS}")

    metadata = Parser().parsestr(snapshot.metadata)
    provided = set(metadata.get_all("Provides-Extra", []))
    plans: list[InstallPlan] = []
    for name in extras:
        if name == "base":
            plans.append(
                InstallPlan(
                    name=name,
                    root=environments_root / name,
                    requirement=str(wheel),
                    required_imports=(),
                    forbidden_imports=_OPTIONAL_IMPORTS,
                )
            )
            continue
        if name not in provided:
            raise VerificationError(f"wheel does not provide requested extra {name!r}")
        if name not in _EXTRA_IMPORTS:
            raise VerificationError(f"requested extra {name!r} has no import probe")
        plans.append(
            InstallPlan(
                name=name,
                root=environments_root / name,
                requirement=f"{wheel}[{name}]",
                required_imports=_EXTRA_IMPORTS[name],
                forbidden_imports=tuple(sorted(set(_OPTIONAL_IMPORTS) - set(_EXTRA_IMPORTS[name]))),
            )
        )
    return plans


def _isolated_environment(home: Path, cache_root: Path) -> dict[str, str]:
    home.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    env = {name: value for name, value in os.environ.items() if name in _ENV_ALLOWLIST or name.startswith("LC_")}
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(home),
            "NPM_CONFIG_USERCONFIG": str(home / ".npmrc"),
            "PIP_CONFIG_FILE": os.devnull,
            "USERPROFILE": str(home),
            "UV_CACHE_DIR": str(cache_root / "uv"),
            "UV_NO_CONFIG": "1",
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_DATA_HOME": str(home / ".local" / "share"),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "npm_config_cache": str(cache_root / "npm"),
            "PYTHONPATH": "",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return env


def _git_output(arguments: list[str], env: dict[str, str]) -> bytes:
    executable = shutil.which("git")
    if executable is None:
        raise VerificationError("git executable is not available")
    try:
        completed = subprocess.run(
            [executable, "-c", "core.fsmonitor=false", *arguments],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise VerificationError(f"git command could not run: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise VerificationError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _source_paths(env: dict[str, str]) -> tuple[str, ...]:
    output = _git_output(["ls-files", "--cached", "--others", "--exclude-standard", "-z"], env)
    paths = tuple(sorted({os.fsdecode(raw) for raw in output.split(b"\0") if raw}))
    for value in paths:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
            raise VerificationError(f"git reported an unsafe source path: {value}")
    return paths


def _manifest_entries(root: Path, paths: tuple[str, ...]) -> list[dict]:
    source_paths = set(paths)
    entries: list[dict] = []
    for value in paths:
        path = root / value
        if path.is_symlink():
            target = os.readlink(path)
            target_path = PurePosixPath(target)
            resolved_target = PurePosixPath(posixpath.normpath(str(PurePosixPath(value).parent / target_path)))
            if (
                target_path.is_absolute()
                or resolved_target.is_absolute()
                or ".." in resolved_target.parts
                or str(resolved_target) not in source_paths
            ):
                raise VerificationError(f"source symlink escapes the source manifest: {value} -> {target}")
            digest = hashlib.sha256(os.fsencode(target)).hexdigest()
            entries.append({"path": value, "type": "symlink", "mode": 0, "sha256": digest})
            continue
        if not path.exists():
            entries.append({"path": value, "type": "missing", "mode": 0, "sha256": None})
            continue
        if not path.is_file():
            raise VerificationError(f"source path is not a regular file: {value}")
        mode = stat.S_IMODE(path.stat().st_mode)
        entries.append({"path": value, "type": "file", "mode": mode, "sha256": _sha256_file(path)})
    return entries


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _capture_source_state(env: dict[str, str]) -> dict:
    paths = _source_paths(env)
    manifest = _manifest_entries(REPO_ROOT, paths)
    status_raw = _git_output(["status", "--porcelain=v1", "-z", "--untracked-files=all"], env)
    status_entries = [os.fsdecode(part) for part in status_raw.split(b"\0") if part]
    changed_paths = sorted({entry[3:] if len(entry) >= 3 and entry[2] == " " else entry for entry in status_entries})
    untracked_paths = tuple(
        sorted(
            os.fsdecode(raw)
            for raw in _git_output(["ls-files", "--others", "--exclude-standard", "-z"], env).split(b"\0")
            if raw
        )
    )
    untracked_set = set(untracked_paths)
    tracked_diff = _git_output(["diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--"], env)
    commit = _git_output(["rev-parse", "HEAD"], env).decode("ascii").strip()
    tree = _git_output(["rev-parse", "HEAD^{tree}"], env).decode("ascii").strip()
    source_manifest_sha256 = _canonical_sha256(manifest)
    public = {
        "commit": commit,
        "tree": tree,
        "clean": not status_raw,
        "changed_paths": changed_paths,
        "status_sha256": hashlib.sha256(status_raw).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked_manifest_sha256": _canonical_sha256([entry for entry in manifest if entry["path"] in untracked_set]),
        "source_manifest_sha256": source_manifest_sha256,
        "source_file_count": sum(entry["type"] != "missing" for entry in manifest),
    }
    public["fingerprint_sha256"] = _canonical_sha256(public)
    return {**public, "_paths": paths, "_manifest": manifest}


def _public_source_state(state: dict) -> dict:
    return {name: value for name, value in state.items() if not name.startswith("_")}


def _copy_source_snapshot(destination: Path, state: dict) -> None:
    if destination.exists():
        raise VerificationError(f"source snapshot destination already exists: {destination}")
    destination.mkdir(parents=True)
    paths = tuple(state["_paths"])
    for value in paths:
        source = REPO_ROOT / value
        target = destination / value
        if source.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.readlink(source))
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    copied_manifest = _manifest_entries(destination, paths)
    copied_sha256 = _canonical_sha256(copied_manifest)
    if copied_sha256 != state["source_manifest_sha256"]:
        raise VerificationError("source changed while the isolated snapshot was being copied")


def _assert_source_unchanged(before: dict, env: dict[str, str]) -> dict:
    after = _capture_source_state(env)
    if after["fingerprint_sha256"] != before["fingerprint_sha256"]:
        raise VerificationError("source checkout changed during verification")
    return after


def _verify_wheel_equivalence(direct: WheelSnapshot, rebuilt: WheelSnapshot) -> None:
    def comparable(snapshot: WheelSnapshot) -> dict[str, str]:
        return {name: digest for name, digest in snapshot.files.items() if name != f"{snapshot.dist_info}/RECORD"}

    direct_files = comparable(direct)
    rebuilt_files = comparable(rebuilt)
    if direct_files == rebuilt_files:
        return
    missing = sorted(set(direct_files) - set(rebuilt_files))
    unexpected = sorted(set(rebuilt_files) - set(direct_files))
    changed = sorted(
        name for name in set(direct_files) & set(rebuilt_files) if direct_files[name] != rebuilt_files[name]
    )
    raise VerificationError(
        f"sdist wheel differs from direct wheel; missing={missing}, unexpected={unexpected}, changed={changed}"
    )


def _validate_doctor_probe(completed: object, expected_config_path: Path) -> dict:
    returncode = getattr(completed, "returncode", None)
    stdout = getattr(completed, "stdout", "")
    if returncode != 1:
        raise VerificationError(f"unconfigured doctor probe must exit 1, got {returncode}")
    try:
        report = json.loads(stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise VerificationError("doctor --json did not emit valid JSON") from exc
    paths = report.get("paths") if isinstance(report, dict) else None
    if not isinstance(paths, dict) or paths.get("config_exists") is not False:
        raise VerificationError("doctor probe did not report an absent config")
    if paths.get("config_path") != str(expected_config_path):
        raise VerificationError("doctor probe did not report the isolated CodeFlow config path")
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: list[str],
    *,
    cwd: Path,
    log_dir: Path,
    records: list[dict],
    env: dict[str, str],
    expected: tuple[int, ...] = (0,),
    timeout: int | float = 1800,
) -> subprocess.CompletedProcess[str]:
    print(f"+ {shlex.join(command)}")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        elapsed = time.monotonic() - started
        stdout = _command_output(getattr(exc, "stdout", ""))
        stderr = _command_output(getattr(exc, "stderr", ""))
        outcome = "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "spawn_error"
        _record_command(
            command=command,
            cwd=cwd,
            log_dir=log_dir,
            records=records,
            expected=expected,
            elapsed=elapsed,
            exit_code=None,
            stdout=stdout,
            stderr=stderr,
            accepted=False,
            outcome=outcome,
            error=str(exc),
        )
        raise VerificationError(f"command could not run: {shlex.join(command)}: {exc}") from exc
    elapsed = time.monotonic() - started
    accepted = completed.returncode in expected
    _record_command(
        command=command,
        cwd=cwd,
        log_dir=log_dir,
        records=records,
        expected=expected,
        elapsed=elapsed,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        accepted=accepted,
        outcome="completed",
    )
    if not accepted:
        detail = (completed.stderr or completed.stdout).strip()[-4000:]
        raise VerificationError(
            f"command exited {completed.returncode}, expected {expected}: {shlex.join(command)}\n{detail}"
        )
    return completed


def _command_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _record_command(
    *,
    command: list[str],
    cwd: Path,
    log_dir: Path,
    records: list[dict],
    expected: tuple[int, ...],
    elapsed: float,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    accepted: bool,
    outcome: str,
    error: str | None = None,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{len(records) + 1:02d}-command.log"
    log_path.write_text(
        f"$ {shlex.join(command)}\n"
        f"cwd: {cwd}\n"
        f"outcome: {outcome}\n"
        f"exit: {exit_code}\n"
        f"elapsed_seconds: {elapsed:.3f}\n\n"
        f"[stdout]\n{stdout}\n\n"
        f"[stderr]\n{stderr}\n" + (f"\n[error]\n{error}\n" if error else ""),
        encoding="utf-8",
    )
    records.append(
        {
            "command": command,
            "cwd": str(cwd),
            "exit_code": exit_code,
            "expected_exit_codes": list(expected),
            "accepted": accepted,
            "outcome": outcome,
            "elapsed_seconds": round(elapsed, 3),
            "log": str(log_path),
            "log_sha256": _sha256_file(log_path),
            **({"error": error} if error else {}),
        }
    )


def _only_artifact(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise VerificationError(f"expected exactly one {label} in {directory}, found {matches}")
    return matches[0]


def _build_artifacts(
    source_root: Path,
    output_root: Path,
    records: list[dict],
    env: dict[str, str],
) -> tuple[Path, Path]:
    log_dir = output_root / "logs"
    _run(
        ["npm", "ci", "--prefix", "ui-tui"],
        cwd=source_root,
        log_dir=log_dir,
        records=records,
        env=env,
    )
    _run(
        ["npm", "run", "build", "--prefix", "ui-tui"],
        cwd=source_root,
        log_dir=log_dir,
        records=records,
        env=env,
    )
    dist = output_root / "dist"
    dist.mkdir()
    _run(
        ["uv", "build", "--wheel", "--out-dir", str(dist), "--no-create-gitignore", "."],
        cwd=source_root,
        log_dir=log_dir,
        records=records,
        env=env,
    )
    _run(
        ["uv", "build", "--sdist", "--out-dir", str(dist), "--no-create-gitignore", "."],
        cwd=source_root,
        log_dir=log_dir,
        records=records,
        env=env,
    )
    return _only_artifact(dist, "*.whl", "wheel"), _only_artifact(dist, "*.tar.gz", "sdist")


def _environment_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _environment_entrypoint(root: Path, entrypoint: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return root / ("Scripts" if os.name == "nt" else "bin") / f"{entrypoint}{suffix}"


def _validate_import_probe(completed: subprocess.CompletedProcess[str], plan: InstallPlan) -> dict:
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"import probe for {plan.name} did not emit valid JSON") from exc
    present = result.get("present", {})
    missing = [name for name in plan.required_imports if present.get(name) is not True]
    leaked = [name for name in plan.forbidden_imports if present.get(name) is not False]
    if missing or leaked:
        raise VerificationError(f"import probe for {plan.name} failed; missing={missing}, leaked={leaked}")
    return result


def _validate_package_probe(completed: subprocess.CompletedProcess[str], plan: InstallPlan) -> dict:
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"package probe for {plan.name} did not emit valid JSON") from exc
    root = plan.root.resolve()
    package = Path(result.get("package", "")).resolve()
    tui = Path(result.get("tui", "")).resolve()
    if root not in package.parents:
        raise VerificationError(f"{plan.name} imported CodeFlow's internal package outside its environment: {package}")
    if root not in tui.parents or tui.name != "entry.js":
        raise VerificationError(f"{plan.name} resolved the TUI outside its installed wheel: {tui}")
    public_commands = set(result.get("public_commands", []))
    registered_commands = set(result.get("registered_commands", []))
    if public_commands != _PUBLIC_CLI_COMMANDS:
        raise VerificationError(
            f"{plan.name} public CLI drifted; expected={sorted(_PUBLIC_CLI_COMMANDS)}, got={sorted(public_commands)}"
        )
    if registered_commands != _REGISTERED_CLI_COMMANDS:
        raise VerificationError(
            f"{plan.name} registered CLI drifted; "
            f"expected={sorted(_REGISTERED_CLI_COMMANDS)}, got={sorted(registered_commands)}"
        )
    return result


def _validate_codeflow_module_probe(completed: subprocess.CompletedProcess[str], plan: InstallPlan) -> None:
    if completed.returncode != 0 or "CodeFlow v" not in completed.stdout:
        raise VerificationError(f"{plan.name} cannot run the CodeFlow module entrypoint")


def _validate_product_paths(
    report: dict,
    *,
    probe_home: Path,
    probe_cwd: Path,
) -> None:
    resolved = probe_cwd.expanduser().resolve()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", resolved.name).strip(".-") or "workspace"
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    expected = {
        "product_home": probe_home / ".codeflow",
        "workspace_state": probe_home / ".codeflow" / "projects" / f"{slug}-{digest}",
        "plugin_user": probe_home / ".codeflow" / "plugins",
    }
    mismatches = {
        name: {"expected": str(path), "actual": report.get(name)}
        for name, path in expected.items()
        if Path(report.get(name, "")) != path
    }
    if report.get("plugin_project") is not None:
        mismatches["plugin_project"] = {
            "expected": None,
            "actual": report.get("plugin_project"),
        }
    if report.get("plugin_entrypoints") != "codeflow.plugins":
        mismatches["plugin_entrypoints"] = {
            "expected": "codeflow.plugins",
            "actual": report.get("plugin_entrypoints"),
        }
    if mismatches:
        raise VerificationError(f"installed CodeFlow state roots drifted: {mismatches}")


def _gateway_probe_config() -> dict:
    return {
        "agents": {
            "defaults": {
                "model": "ollama/codeflow-distribution-probe",
                "provider": "ollama",
            }
        },
        "channels": {
            "feishu": {"enabled": False},
            "qq": {"enabled": False},
            "wecom": {"enabled": False},
        },
        "tools": {
            "sandbox": {"backend": "none"},
        },
        "plugins": {"disabled": []},
        "memory": {"backend": None},
        "skillForge": {
            "enabled": False,
            "router": {"enabled": False},
        },
        "runtime": {"checkpoint": {"policy": "never"}},
        "tracing": {"enabled": False},
    }


def _validate_gateway_health(status: int, content_type: str, body: bytes) -> dict:
    if status != 200:
        raise VerificationError(f"gateway health returned HTTP {status}")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise VerificationError(f"gateway health returned content type {content_type!r}")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError("gateway health did not return valid JSON") from exc
    if payload != {"status": "ok"}:
        raise VerificationError(f"gateway health returned unexpected payload: {payload!r}")
    return payload


def _allocate_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_gateway_health(port: int) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
    try:
        connection.request("GET", "/health", headers={"Connection": "close"})
        response = connection.getresponse()
        body = response.read()
        return _validate_gateway_health(
            response.status,
            response.getheader("Content-Type", ""),
            body,
        )
    finally:
        connection.close()


def _wait_for_gateway_health(
    process: subprocess.Popen,
    port: int,
    *,
    deadline: float,
) -> tuple[dict, int]:
    attempts = 0
    last_error = "connection was not attempted"
    while True:
        returncode = process.poll()
        if returncode is not None:
            raise _GatewayProbeError(
                "early_exit",
                f"gateway exited with {returncode} before health became ready",
            )
        attempts += 1
        try:
            return _request_gateway_health(port), attempts
        except VerificationError as exc:
            raise _GatewayProbeError("health_mismatch", str(exc)) from exc
        except (OSError, http.client.HTTPException) as exc:
            last_error = str(exc) or exc.__class__.__name__
        now = time.monotonic()
        if now >= deadline:
            raise _GatewayProbeError(
                "readiness_timeout",
                f"gateway health did not become ready: {last_error}",
            )
        time.sleep(min(0.1, max(0.0, deadline - now)))


def _signal_gateway_process(process: subprocess.Popen, sig: signal.Signals) -> None:
    if os.name == "nt":
        if sig == signal.SIGINT and hasattr(signal, "CTRL_BREAK_EVENT"):
            process.send_signal(signal.CTRL_BREAK_EVENT)
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
        return
    os.killpg(process.pid, sig)


def _kill_gateway_process(process: subprocess.Popen) -> None:
    if os.name == "nt":
        process.kill()
    else:
        os.killpg(process.pid, signal.SIGKILL)


def _stop_gateway_process(
    process: subprocess.Popen,
    *,
    shutdown_timeout: float,
    cleanup_timeout: float,
) -> tuple[str | None, int | None, bool]:
    if process.poll() is not None:
        return None, process.returncode, False
    try:
        _signal_gateway_process(process, signal.SIGINT)
    except ProcessLookupError:
        return None, process.poll(), False
    try:
        return "SIGINT", process.wait(timeout=shutdown_timeout), False
    except subprocess.TimeoutExpired:
        pass

    try:
        _signal_gateway_process(process, signal.SIGTERM)
    except ProcessLookupError:
        return "SIGINT", process.poll(), True
    try:
        process.wait(timeout=cleanup_timeout)
    except subprocess.TimeoutExpired:
        try:
            _kill_gateway_process(process)
        except ProcessLookupError:
            pass
        process.wait(timeout=cleanup_timeout)
    return "SIGINT", process.returncode, True


def _probe_installed_gateway(
    executable: Path,
    *,
    root: Path,
    cwd: Path,
    env: dict[str, str],
    log_dir: Path,
    records: list[dict],
    readiness_timeout: float = 30.0,
    stability_seconds: float = 1.0,
    shutdown_timeout: float = 10.0,
    cleanup_timeout: float = 3.0,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / "workspace"
    workspace.mkdir()
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps(_gateway_probe_config(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        port = _allocate_loopback_port()
    except OSError as exc:
        raise VerificationError(f"gateway probe port allocation failed: {exc}") from exc

    command = [
        str(executable),
        "gateway",
        "--config",
        str(config_path),
        "--workspace",
        str(workspace),
        "--port",
        str(port),
    ]
    child_env = dict(env)
    child_env["CODEFLOW_TRACING"] = "0"
    stdout_path = root / "stdout.log"
    stderr_path = root / "stderr.log"
    started = time.monotonic()
    process: subprocess.Popen | None = None
    failure: _GatewayProbeError | None = None
    health_attempts = 0
    startup_seconds: float | None = None
    shutdown_signal: str | None = None
    shutdown_exit_code: int | None = None
    forced = False

    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open(
            "w",
            encoding="utf-8",
        ) as stderr_handle,
    ):
        popen_options: dict = {}
        if os.name == "nt":
            popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_options["start_new_session"] = True
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=child_env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                **popen_options,
            )
        except OSError as exc:
            failure = _GatewayProbeError("spawn_error", str(exc))
        else:
            try:
                _, attempts = _wait_for_gateway_health(
                    process,
                    port,
                    deadline=started + readiness_timeout,
                )
                health_attempts += attempts
                startup_seconds = time.monotonic() - started
                stable_deadline = time.monotonic() + stability_seconds
                while time.monotonic() < stable_deadline:
                    returncode = process.poll()
                    if returncode is not None:
                        raise _GatewayProbeError(
                            "early_exit_after_health",
                            f"gateway exited with {returncode} during the stability window",
                        )
                    time.sleep(min(0.1, max(0.0, stable_deadline - time.monotonic())))
                try:
                    _request_gateway_health(port)
                except (OSError, http.client.HTTPException, VerificationError) as exc:
                    raise _GatewayProbeError(
                        "health_mismatch",
                        f"gateway health did not remain stable: {exc}",
                    ) from exc
                health_attempts += 1
            except _GatewayProbeError as exc:
                failure = exc
            finally:
                shutdown_signal, shutdown_exit_code, forced = _stop_gateway_process(
                    process,
                    shutdown_timeout=shutdown_timeout,
                    cleanup_timeout=cleanup_timeout,
                )

    elapsed = time.monotonic() - started
    if failure is None and forced:
        failure = _GatewayProbeError(
            "shutdown_timeout",
            "gateway required forced termination after SIGINT",
        )
    expected_exit_codes = (0, 130, -int(signal.SIGINT))
    if failure is None and shutdown_exit_code not in expected_exit_codes:
        failure = _GatewayProbeError(
            "shutdown_exit_error",
            f"gateway exited with {shutdown_exit_code} after SIGINT",
        )

    report = {
        "endpoint": f"http://127.0.0.1:{port}/health",
        "port": port,
        "config": str(config_path),
        "workspace": str(workspace),
        "health_attempts": health_attempts,
        "startup_seconds": round(startup_seconds, 3) if startup_seconds is not None else None,
        "stable_seconds": stability_seconds,
        "shutdown_signal": shutdown_signal,
        "shutdown_exit_code": shutdown_exit_code,
        "forced": forced,
    }
    stdout = stdout_path.read_text(encoding="utf-8")
    stderr = stderr_path.read_text(encoding="utf-8")
    accepted = failure is None
    _record_command(
        command=command,
        cwd=cwd,
        log_dir=log_dir,
        records=records,
        expected=expected_exit_codes,
        elapsed=elapsed,
        exit_code=shutdown_exit_code,
        stdout=stdout,
        stderr=stderr,
        accepted=accepted,
        outcome="healthy_and_stopped" if accepted else failure.outcome,
        error=str(failure) if failure is not None else None,
    )
    records[-1]["gateway"] = report
    if failure is not None:
        raise VerificationError(f"installed gateway probe failed ({failure.outcome}): {failure}") from failure
    return report


def _probe_base_runtime_surfaces(
    plan: InstallPlan,
    *,
    executable: Path,
    probe_cwd: Path,
    probe_env: dict[str, str],
    log_dir: Path,
    records: list[dict],
) -> dict:
    if plan.name != "base":
        return {}
    _run(
        [str(executable), "--check"],
        cwd=probe_cwd,
        env=probe_env,
        log_dir=log_dir,
        records=records,
        timeout=_TUI_PROBE_TIMEOUT_SECONDS,
    )
    gateway = _probe_installed_gateway(
        executable,
        root=plan.root / "gateway-probe",
        cwd=probe_cwd,
        env=probe_env,
        log_dir=log_dir,
        records=records,
    )
    return {"gateway": gateway}


def _install_environment(
    plan: InstallPlan,
    *,
    entrypoint: str,
    probe_cwd: Path,
    log_dir: Path,
    records: list[dict],
    package_env: dict[str, str],
) -> dict:
    _run(
        ["uv", "venv", "--python", sys.executable, "--no-project", str(plan.root)],
        cwd=probe_cwd,
        log_dir=log_dir,
        records=records,
        env=package_env,
    )
    python = _environment_python(plan.root)
    executable = _environment_entrypoint(plan.root, entrypoint)
    _run(
        ["uv", "pip", "install", "--python", str(python), "--strict", plan.requirement],
        cwd=probe_cwd,
        log_dir=log_dir,
        records=records,
        env=package_env,
    )
    _run(
        ["uv", "pip", "check", "--python", str(python)],
        cwd=probe_cwd,
        log_dir=log_dir,
        records=records,
        env=package_env,
    )

    probe_env = _isolated_environment(plan.root / "probe-home", plan.root / "probe-cache")
    probe_home = Path(probe_env["HOME"])
    import_code = _import_probe_code(plan.required_imports, plan.forbidden_imports)
    import_completed = _run(
        [str(python), "-I", "-c", import_code],
        cwd=probe_cwd,
        env=probe_env,
        log_dir=log_dir,
        records=records,
    )
    import_report = _validate_import_probe(import_completed, plan)

    package_code = (
        "import json,codeflow,sys,typer; "
        "from codeflow.cli._plugin_stack import plugin_discovery_sources; "
        "from codeflow.cli.commands import app; "
        "from codeflow.cli.tui_commands import resolve_dist_entry; "
        "from codeflow.product import get_product_home,get_project_state_dir; "
        "root=typer.main.get_command(app); "
        "entry=resolve_dist_entry(); "
        "sources=plugin_discovery_sources(); "
        "print(json.dumps({'package': codeflow.__file__, 'tui': str(entry) if entry else '', "
        "'public_commands': sorted(name for name, command in root.commands.items() if not command.hidden), "
        "'registered_commands': sorted(root.commands), "
        "'product_home': str(get_product_home()), "
        "'workspace_state': str(get_project_state_dir(__import__('pathlib').Path.cwd())), "
        "'plugin_user': str(sources['user_dir']), "
        "'plugin_project': str(sources['project_dir']) if sources['project_dir'] is not None else None, "
        "'plugin_entrypoints': sources['entry_points_group'], "
        "'python': sys.version.split()[0], 'python_executable': sys.executable}))"
    )
    package_completed = _run(
        [str(python), "-I", "-c", package_code],
        cwd=probe_cwd,
        env=probe_env,
        log_dir=log_dir,
        records=records,
    )
    package_report = _validate_package_probe(package_completed, plan)
    _validate_product_paths(
        package_report,
        probe_home=probe_home,
        probe_cwd=probe_cwd,
    )
    codeflow_module = _run(
        [str(python), "-I", "-m", "codeflow", "--version"],
        cwd=probe_cwd,
        env=probe_env,
        log_dir=log_dir,
        records=records,
    )
    _validate_codeflow_module_probe(codeflow_module, plan)
    _run(
        [str(executable), "--help"],
        cwd=probe_cwd,
        env=probe_env,
        log_dir=log_dir,
        records=records,
    )
    doctor = _run(
        [str(executable), "doctor", "--json"],
        cwd=probe_cwd,
        env=probe_env,
        log_dir=log_dir,
        records=records,
        expected=(1,),
    )
    doctor_report = _validate_doctor_probe(
        doctor,
        probe_home / ".codeflow" / "config.json",
    )
    _run(
        [str(executable), "plugins", "--verbose"],
        cwd=probe_cwd,
        env=probe_env,
        log_dir=log_dir,
        records=records,
    )
    runtime_surfaces = _probe_base_runtime_surfaces(
        plan,
        executable=executable,
        probe_cwd=probe_cwd,
        probe_env=probe_env,
        log_dir=log_dir,
        records=records,
    )
    return {
        "name": plan.name,
        "root": str(plan.root),
        "python": str(python),
        "entrypoint": str(executable),
        "imports": import_report,
        "package": package_report,
        "doctor": doctor_report,
        **runtime_surfaces,
    }


def _import_probe_code(required: tuple[str, ...], forbidden: tuple[str, ...]) -> str:
    return (
        "import contextlib,importlib,importlib.util,io,json\n"
        f"required={list(required)!r}\n"
        f"forbidden={list(forbidden)!r}\n"
        "present={}\n"
        "errors={}\n"
        "captured_stdout=io.StringIO()\n"
        "captured_stderr=io.StringIO()\n"
        "with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):\n"
        "    for name in required:\n"
        "        try:\n"
        "            importlib.import_module(name)\n"
        "        except Exception as exc:\n"
        "            present[name]=False\n"
        "            errors[name]=f'{type(exc).__name__}: {exc}'\n"
        "        else:\n"
        "            present[name]=True\n"
        "    for name in forbidden:\n"
        "        present[name]=importlib.util.find_spec(name) is not None\n"
        "print(json.dumps({'present': present, 'errors': errors, "
        "'captured_stdout': captured_stdout.getvalue(), 'captured_stderr': captured_stderr.getvalue()}))\n"
    )


def _verify_sdist_round_trip(
    sdist: Path,
    direct: WheelSnapshot,
    *,
    entrypoint: str,
    output_root: Path,
    records: list[dict],
    env: dict[str, str],
) -> WheelSnapshot:
    rebuilt_dir = output_root / "sdist-wheel"
    rebuilt_dir.mkdir()
    _run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(rebuilt_dir),
            "--no-create-gitignore",
            str(sdist),
        ],
        cwd=output_root,
        log_dir=output_root / "logs",
        records=records,
        env=env,
    )
    rebuilt = _wheel_snapshot(_only_artifact(rebuilt_dir, "*.whl", "sdist-built wheel"))
    _validate_wheel(rebuilt, entrypoint)
    _verify_wheel_equivalence(direct, rebuilt)
    return rebuilt


def _build_distribution_handoff(wheel: Path, environments: list[dict]) -> dict:
    base_environments = [environment for environment in environments if environment.get("name") == "base"]
    if len(base_environments) != 1:
        raise VerificationError("distribution handoff requires exactly one base environment")
    base = base_environments[0]
    root = Path(base.get("root", ""))
    python = Path(base.get("python", ""))
    entrypoint = Path(base.get("entrypoint", ""))
    missing = [
        name
        for name, path, expected_type in (
            ("wheel", wheel, "file"),
            ("base environment", root, "directory"),
            ("base python", python, "file"),
            ("base entrypoint", entrypoint, "file"),
        )
        if (expected_type == "file" and not path.is_file()) or (expected_type == "directory" and not path.is_dir())
    ]
    if missing:
        raise VerificationError(f"distribution handoff paths are missing: {missing}")
    return {
        "wheel": str(wheel),
        "base_environment": {
            "root": str(root),
            "python": str(python),
            "entrypoint": str(entrypoint),
        },
    }


def _write_report(output_root: Path, report: dict) -> Path:
    path = output_root / "distribution-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and verify an isolated CodeFlow distribution.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--extras", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root: Path | None = None
    records: list[dict] = []
    source_state: dict | None = None
    try:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", args.entrypoint):
            raise VerificationError("entrypoint contains unsupported characters")
        extras = tuple(part.strip() for part in args.extras.split(",") if part.strip())
        output_root = _prepare_output_root(args.output_root)
        command_env = _isolated_environment(output_root / "command-home", output_root / "cache")
        source_state = _capture_source_state(command_env)
        source_root = output_root / "source"
        _copy_source_snapshot(source_root, source_state)
        wheel, sdist = _build_artifacts(source_root, output_root, records, command_env)
        direct = _wheel_snapshot(wheel)
        _validate_wheel(direct, args.entrypoint)
        _validate_sdist(sdist)

        plans = _build_install_plans(output_root, wheel, extras, direct)
        probe_cwd = output_root / "probe-cwd"
        probe_cwd.mkdir()
        environments = [
            _install_environment(
                plan,
                entrypoint=args.entrypoint,
                probe_cwd=probe_cwd,
                log_dir=output_root / "logs",
                records=records,
                package_env=command_env,
            )
            for plan in plans
        ]
        rebuilt = _verify_sdist_round_trip(
            sdist,
            direct,
            entrypoint=args.entrypoint,
            output_root=output_root,
            records=records,
            env=command_env,
        )
        _assert_source_unchanged(source_state, command_env)
        source_report = _public_source_state(source_state)
        source_report.update(
            {
                "checkout": str(REPO_ROOT),
                "snapshot": str(source_root),
                "unchanged_during_verification": True,
            }
        )
        report = {
            "schema_version": 3,
            "status": "passed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "source_sha": source_state["commit"],
            "source": source_report,
            "artifacts": {
                "wheel": {
                    "path": str(wheel),
                    "sha256": _sha256_file(wheel),
                    "files": len(direct.files),
                },
                "sdist": {"path": str(sdist), "sha256": _sha256_file(sdist)},
                "sdist_wheel": {
                    "path": str(rebuilt.path),
                    "sha256": _sha256_file(rebuilt.path),
                    "files": len(rebuilt.files),
                },
            },
            "handoff": _build_distribution_handoff(wheel, environments),
            "environments": environments,
            "commands": records,
        }
        report_path = _write_report(output_root, report)
        print(f"Distribution verification passed: {report_path}")
        return 0
    except VerificationError as exc:
        if output_root is not None and output_root.exists():
            _write_report(
                output_root,
                {
                    "schema_version": 3,
                    "status": "failed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                    "source": _public_source_state(source_state) if source_state else None,
                    "commands": records,
                },
            )
        print(f"Distribution verification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
