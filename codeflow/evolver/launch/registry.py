"""维护 bench name -> ``module:function`` 的 BenchBundle builder registry。

registered plugin 通常位于 subject repo，而非 installed CodeFlow package。loader 会把 subject root
放在 ``sys.path`` 前部，并清理来自其他 root 的冲突 package cache，防止导入错误 checkout。
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional, Union

BENCHES: dict[str, str] = {
    "appworld": "benchmarks.appworld.evolve.entry:build",
}


def _module_belongs_to(module: ModuleType, root: Path) -> bool:
    locations = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        locations.append(Path(module_file))
    module_path = getattr(module, "__path__", None)
    if module_path:
        locations.extend(Path(path) for path in module_path)
    if not locations:
        return False
    for location in locations:
        try:
            location.resolve().relative_to(root)
        except ValueError:
            return False
    return True


def _evict_conflicting_package(package: str, root: Path) -> None:
    cached = {
        name: module
        for name, module in sys.modules.items()
        if (name == package or name.startswith(f"{package}.")) and isinstance(module, ModuleType)
    }
    if cached and not all(_module_belongs_to(module, root) for module in cached.values()):
        for name in cached:
            del sys.modules[name]


def load_bench(name: str, repo_root: Optional[Union[str, Path]] = None) -> Callable:
    """导入 ``name`` 注册的 bench plugin builder。

    plugin 位于 subject repo 的 ``benchmarks/``，不是 installed CodeFlow package，因此提供
    ``repo_root`` 时将 subject checkout 放到 ``sys.path`` 首位；省略只在 checkout 已可 import
    时有效，例如 cwd 正好是 root。若 target module file 在 subject root，却实际解析到外部
    package，抛出 ``ImportError``。unknown name 抛出列出 registry 的 ``ValueError``。
    """
    target = BENCHES.get(name)
    if target is None:
        raise ValueError(
            f"unknown bench {name!r}; registered: {sorted(BENCHES)} (add yours to codeflow.evolver.launch.registry.BENCHES)"
        )
    mod_name, _, fn_name = target.partition(":")
    if repo_root is not None:
        resolved_root = Path(repo_root).resolve()
        root = str(resolved_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        subject_entry = resolved_root.joinpath(*mod_name.split(".")).with_suffix(".py")
        if subject_entry.is_file():
            _evict_conflicting_package(target.partition(".")[0], resolved_root)
    module = import_module(mod_name)
    if repo_root is not None and subject_entry.is_file() and not _module_belongs_to(module, resolved_root):
        raise ImportError(f"bench {name!r} resolved outside repo_root {resolved_root}: {module.__file__}")
    return getattr(module, fn_name)


__all__ = ["BENCHES", "load_bench"]
