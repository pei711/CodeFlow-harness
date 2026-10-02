from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any


class CanonicalizationError(ValueError):
    pass


_FORBIDDEN_KEYS = {
    "access_key",
    "access_token",
    "api_key",
    "authorization",
    "credential",
    "credentials",
    "created_at",
    "generated_at",
    "output_dir",
    "output_directory",
    "output_root",
    "password",
    "secret",
    "secret_key",
    "timestamp",
    "token",
}
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def to_primitive(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: to_primitive(getattr(value, field.name)) for field in dataclasses.fields(value) if field.repr
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [to_primitive(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((to_primitive(item) for item in value), key=canonical_json)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise CanonicalizationError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_primitive(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_plan_value(value: Any, *, path: tuple[str, ...] = ()) -> None:
    primitive = to_primitive(value)
    _validate_primitive(primitive, path=path)


def _validate_primitive(value: Any, *, path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _FORBIDDEN_KEYS:
                location = ".".join((*path, str(key)))
                raise CanonicalizationError(f"forbidden plan field: {location}")
            _validate_primitive(item, path=(*path, str(key)))
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _validate_primitive(item, path=(*path, str(index)))
        return
    if isinstance(value, str) and (value.startswith("/") or _WINDOWS_ABSOLUTE_PATH.match(value)):
        location = ".".join(path) or "<root>"
        raise CanonicalizationError(f"absolute path is not canonical plan input: {location}")
