"""Canonical CallEfficiency Usage 与 Cost 的 Tracing Projection。

`normalize` 复用 `codeflow.call_efficiency.usage.normalize_usage` 统一 Provider Token Semantics，并只在 Usage
Complete 且有 Model 时调用同一 Pricing Source 估算 Cost。返回 Dict 同时保留 Normalized Fields、
Findings 与 Raw Usage，使 Viewer 能区分已知值、Ambiguous Evidence 与原始响应。

Tracing 不维护第二套计费规则；Cost 是 Estimate，不是 Invoice，`usage_complete=False` 时刻意不计算。
"""

from __future__ import annotations

from typing import Any


def normalize(usage: dict[str, Any] | None, model: str | None) -> dict[str, Any]:
    from codeflow.call_efficiency.pricing import estimate_cost_usd
    from codeflow.call_efficiency.usage import normalize_usage

    raw = usage or {}
    normalized, findings = normalize_usage(raw, model or "")
    total = normalized.total_tokens or (
        normalized.input_tokens
        + normalized.cache_read_tokens
        + normalized.cache_write_tokens
        + normalized.output_tokens
    )
    cost = None
    if model and normalized.complete:
        try:
            cost = estimate_cost_usd(
                model,
                normalized.input_tokens,
                normalized.output_tokens,
                normalized.cache_read_tokens,
                normalized.cache_write_tokens,
                allow_litellm_import=False,
            )
        except Exception:  # noqa: BLE001
            cost = None
    return {
        "input_tokens": normalized.input_tokens,
        "output_tokens": normalized.output_tokens,
        "cache_read_tokens": normalized.cache_read_tokens,
        "cache_write_tokens": normalized.cache_write_tokens,
        "total_tokens": total,
        "cost_usd": cost,
        "usage_complete": normalized.complete,
        "findings": findings,
        "raw": dict(raw),
    }
