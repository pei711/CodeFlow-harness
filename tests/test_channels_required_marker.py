"""Guard: channel ``required`` markers stay aligned with adapter ``start()`` guards.

Each channel's required set below is the audited truth — the credentials whose
absence makes the adapter bail on startup (see the referenced guard). The schema
markers (``Field(json_schema_extra={"required": True})``) must match this table
exactly. When an adapter changes what it enforces, update the adapter, the schema
marker, and this table in the same change.
"""

from __future__ import annotations

import pytest

from codeflow.config.update_channels import channel_field_specs

EXPECTED_REQUIRED: dict[str, set[str]] = {
    "feishu": {"app_id", "app_secret"},
    "qq": {"app_id", "secret"},
    "wecom": {"bot_id", "secret"},
}


@pytest.mark.parametrize("channel, expected", sorted(EXPECTED_REQUIRED.items()))
def test_required_markers_match_audit(channel: str, expected: set[str]) -> None:
    specs = channel_field_specs(channel)
    marked = {path for path, spec in specs.items() if spec.get("required")}
    assert marked == expected
