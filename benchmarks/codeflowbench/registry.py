from __future__ import annotations

from .protocol import Pack


class PackRegistry:
    def __init__(self) -> None:
        self._packs: dict[str, Pack] = {}

    def register(self, pack: Pack) -> None:
        pack_id = pack.definition().pack_id
        if pack_id in self._packs:
            raise ValueError(f"Pack already registered: {pack_id}")
        self._packs[pack_id] = pack

    def get(self, pack_id: str) -> Pack:
        try:
            return self._packs[pack_id]
        except KeyError as exc:
            raise KeyError(f"Pack is not registered: {pack_id}") from exc

    def resolve(self, pack_ids: tuple[str, ...]) -> tuple[Pack, ...]:
        return tuple(self.get(pack_id) for pack_id in pack_ids)


_DEFAULT_REGISTRY = PackRegistry()


def register_standard_packs(registry: PackRegistry) -> None:
    try:
        from .packs.swebench import SwebenchPack

        if "swebench" not in registry._packs:
            registry.register(SwebenchPack())
    except ImportError:
        pass

    try:
        from .packs.security import SecurityPack

        if "security" not in registry._packs:
            registry.register(SecurityPack())
    except ImportError:
        pass

    try:
        from .packs.trajectory import TrajectoryPack

        if "trajectory" not in registry._packs:
            registry.register(TrajectoryPack())
    except ImportError:
        pass


def default_registry() -> PackRegistry:
    register_standard_packs(_DEFAULT_REGISTRY)
    return _DEFAULT_REGISTRY
