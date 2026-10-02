from __future__ import annotations

import argparse
import json
from pathlib import Path

from codeflow.session.export import default_export_path, verify_export, write_portable_export
from codeflow.session.manager import SessionManager


def _contents(session) -> list[str]:
    return [str(message.get("content", "")) for message in session.messages]


def _seed(workspace: Path) -> dict:
    manager = SessionManager(workspace)
    parent = manager.get_or_create("cli:continuity-parent")
    parent.metadata["title"] = "Continuity parent"
    parent.add_message("user", "shared question")
    parent.add_message("assistant", "shared answer")
    manager.save(parent)

    child = manager.fork(parent.key, title="Continuity child")
    if child is None:
        raise RuntimeError("failed to fork seeded Session")

    export_path = write_portable_export(
        parent,
        default_export_path(workspace, parent.key),
    )
    if not verify_export(export_path):
        raise RuntimeError("portable export verification failed")
    return {
        "parent_key": parent.key,
        "child_key": child.key,
        "export_path": str(export_path),
        "shared_contents": _contents(parent),
    }


def _diverge_and_delete(
    workspace: Path,
    parent_key: str,
    child_key: str,
) -> dict:
    manager = SessionManager(workspace)
    parent = manager.peek(parent_key)
    child = manager.peek(child_key)
    if parent is None or child is None:
        raise RuntimeError("fresh process could not resume seeded Sessions")
    if _contents(parent) != _contents(child):
        raise RuntimeError("fork did not preserve the source history")

    parent.add_message("user", "parent only")
    child.add_message("user", "child only")
    manager.save(parent)
    manager.save(child)
    if not manager.delete(parent_key):
        raise RuntimeError("target Session delete failed")

    return {
        "parent_key": parent_key,
        "child_key": child_key,
        "parent_contents_before_delete": _contents(parent),
        "child_contents": _contents(child),
    }


def _verify(
    workspace: Path,
    parent_key: str,
    child_key: str,
    export_path: Path,
) -> dict:
    manager = SessionManager(workspace)
    parent = manager.peek(parent_key)
    child = manager.peek(child_key)
    if parent is not None:
        raise RuntimeError("deleted Session resumed in a fresh process")
    if child is None:
        raise RuntimeError("deleting the parent removed its child")
    if "parent only" in _contents(child) or "child only" not in _contents(child):
        raise RuntimeError("post-fork writes were not isolated")
    if not verify_export(export_path):
        raise RuntimeError("portable export did not survive Session deletion")

    return {
        "parent_not_found": True,
        "child_contents": _contents(child),
        "child_parent": child.metadata.get("parent_session_id"),
        "export_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("seed", "diverge", "verify"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--parent-key")
    parser.add_argument("--child-key")
    parser.add_argument("--export-path", type=Path)
    args = parser.parse_args()

    if args.action == "seed":
        result = _seed(args.workspace)
    elif args.action == "diverge":
        if not args.parent_key or not args.child_key:
            parser.error("diverge requires --parent-key and --child-key")
        result = _diverge_and_delete(
            args.workspace,
            args.parent_key,
            args.child_key,
        )
    else:
        if not args.parent_key or not args.child_key or args.export_path is None:
            parser.error("verify requires --parent-key, --child-key, and --export-path")
        result = _verify(
            args.workspace,
            args.parent_key,
            args.child_key,
            args.export_path,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
