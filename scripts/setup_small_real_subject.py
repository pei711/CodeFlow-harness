"""Materialize the disposable small-real Evolution Run subject repo.

Copies ``benchmarks/evolver/subject_template/`` to ``benchmarks/evolver/subject/``
(gitignored), runs ``git init``, and creates one deterministic initial commit,
so every operator starts a small-real run from the same root commit.

Rerun behavior is explicit:

- missing subject        -> created
- clean subject + --reuse    -> kept as is
- clean subject + --recreate -> deleted and recreated (same commit sha)
- clean subject, no flag     -> refused; pick --reuse or --recreate
- dirty subject          -> always refused; inspect or remove it by hand first
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = REPO_ROOT / "benchmarks" / "evolver" / "subject_template"
DEFAULT_SUBJECT = REPO_ROOT / "benchmarks" / "evolver" / "subject"

AUTHOR_NAME = "Small Real Subject"
AUTHOR_EMAIL = "small-real-subject@example.invalid"
AUTHOR_DATE = "2026-01-01T00:00:00 +0000"
COMMIT_MESSAGE = "chore: seed small-real subject from template"
GIT = shutil.which("git")


def _git(subject: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    if GIT is None:
        raise RuntimeError("git executable not found on PATH")
    proc = subprocess.run([GIT, "-C", str(subject), *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {subject}: {proc.stderr.strip()}")
    return proc


def _is_dirty(subject: Path) -> bool:
    proc = _git(subject, "status", "--porcelain", "-uall")
    return bool(proc.stdout.strip())


def _head_sha(subject: Path) -> str:
    return _git(subject, "rev-parse", "HEAD").stdout.strip()


def _ignored_paths(subject: Path) -> list[str]:
    proc = _git(subject, "status", "--porcelain=v1", "--ignored=matching", "--untracked-files=all")
    return [line[3:] for line in proc.stdout.splitlines() if line.startswith("!! ")]


def _matches_template(subject: Path, template: Path) -> bool:
    top = Path(_git(subject, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if top != subject:
        return False
    tracked = _git(subject, "ls-files", "-z").stdout.split("\0")
    tracked = sorted(path for path in tracked if path)
    expected = sorted(
        str(path.relative_to(template))
        for path in template.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts
    )
    if tracked != expected:
        return False
    return all((subject / path).read_bytes() == (template / path).read_bytes() for path in expected)


def _create(template: Path, subject: Path) -> str:
    if GIT is None:
        raise RuntimeError("git executable not found on PATH")
    shutil.copytree(
        template,
        subject,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
    )
    _git(subject, "init", "-q", "-b", "main")
    _git(subject, "config", "user.name", AUTHOR_NAME)
    _git(subject, "config", "user.email", AUTHOR_EMAIL)
    _git(subject, "add", "-A")
    env_args = [
        "-c",
        f"user.name={AUTHOR_NAME}",
        "-c",
        f"user.email={AUTHOR_EMAIL}",
    ]
    proc = subprocess.run(
        [GIT, "-C", str(subject), *env_args, "commit", "-q", "-m", COMMIT_MESSAGE],
        capture_output=True,
        text=True,
        env={
            "GIT_AUTHOR_NAME": AUTHOR_NAME,
            "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL,
            "GIT_AUTHOR_DATE": AUTHOR_DATE,
            "GIT_COMMITTER_NAME": AUTHOR_NAME,
            "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL,
            "GIT_COMMITTER_DATE": AUTHOR_DATE,
            "PATH": subprocess.os.environ.get("PATH", ""),
            "HOME": subprocess.os.environ.get("HOME", ""),
        },
    )
    if proc.returncode != 0:
        raise RuntimeError(f"initial commit failed in {subject}: {proc.stderr.strip()}")
    return _head_sha(subject)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reuse", action="store_true", help="keep an existing clean subject as is")
    mode.add_argument("--recreate", action="store_true", help="delete and recreate an existing clean subject")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--subject", type=Path, default=DEFAULT_SUBJECT)
    args = parser.parse_args(argv)

    template = args.template.resolve()
    subject = args.subject.resolve()
    if not (template / "benchmarks" / "appworld" / "agent_cli.py").is_file():
        print(f"template is missing benchmarks/appworld/agent_cli.py: {template}", file=sys.stderr)
        return 2

    if subject.exists() and any(subject.iterdir()):
        if not (subject / ".git").exists():
            print(
                f"refusing: {subject} exists and is not a git repo; inspect and remove it by hand first",
                file=sys.stderr,
            )
            return 1
        if _is_dirty(subject):
            print(
                f"refusing: {subject} has uncommitted or untracked changes; "
                "inspect them (git -C ... status) and clean or remove the directory by hand first",
                file=sys.stderr,
            )
            return 1
        if args.reuse:
            print(f"reusing clean subject at {subject} (HEAD {_head_sha(subject)[:12]})")
            return 0
        if not args.recreate:
            print(
                f"subject already exists and is clean: {subject}\n"
                "pass --reuse to keep it or --recreate to rebuild it from the template",
                file=sys.stderr,
            )
            return 2
        ignored = _ignored_paths(subject)
        if ignored:
            print(
                f"refusing: {subject} contains ignored paths that --recreate would delete: {ignored[:5]}",
                file=sys.stderr,
            )
            return 1
        if not _matches_template(subject, template):
            print(
                f"refusing: {subject} is not an exact checkout of the current small-real template; "
                "inspect and remove it by hand if replacement is intended",
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(subject)
    elif subject.exists():
        subject.rmdir()

    sha = _create(template, subject)
    print(f"created subject at {subject} (HEAD {sha[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
