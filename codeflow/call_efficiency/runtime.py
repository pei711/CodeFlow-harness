"""Request Caching、Usage Normalization 与 Call Cost 的 Runtime Owner。

`CallEfficiency` 是一次 Provider Call 计量链的协调者。调用前，`prepare` 根据 Mode 与 Provider
Capability 决定保留、清除或重新规划 ``cache_control``；调用后，`record` 统一不同 Provider 的 Usage、
选择 Accounting Model、估算 Cost，并生成带 Trace Correlation 的 `CallRecord` 交给 Ledger。

Mode 分为 `off`、`observe` 与 `optimize`：关闭时不持久化，观察时只记录而不改请求，优化时才应用
Cache Plan。Ledger Failure 会记录日志但不能把已经完成的 Provider Call 变成失败 Turn；因此调用
记录是独立 Evidence，持久化健康状态必须另行核对。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from codeflow.call_efficiency.cache import (
    CacheCapability,
    apply_anthropic_cache_plan,
    cache_capability,
    has_cache_control,
    strip_cache_control,
    valid_cache_control,
)
from codeflow.call_efficiency.ledger import CallLedger
from codeflow.call_efficiency.models import CallRecord, PreparedCall
from codeflow.call_efficiency.pricing import estimate_cost_usd
from codeflow.call_efficiency.usage import normalize_usage
from codeflow.product import get_product_home
from codeflow.providers.base import LLMProvider, LLMResponse
from codeflow.tracing import trace

_MODES = frozenset({"off", "observe", "optimize"})


class CallEfficiency:
    def __init__(
        self,
        *,
        mode: str = "observe",
        telemetry_dir: Path | None = None,
        persist: bool = True,
        max_cache_breakpoints: int = 4,
        provider: LLMProvider | None = None,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"unsupported CallEfficiency mode: {mode}")
        if not 1 <= max_cache_breakpoints <= 4:
            raise ValueError("max_cache_breakpoints must be between 1 and 4")
        self.mode = mode
        self.max_cache_breakpoints = max_cache_breakpoints
        self.provider = provider
        self.ledger = CallLedger(
            telemetry_dir or get_product_home() / "telemetry",
            persist=persist and mode != "off",
        )

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        telemetry_dir: Path,
        provider: LLMProvider | None = None,
    ) -> "CallEfficiency":
        mode = getattr(config, "effective_mode", getattr(config, "mode", "observe"))
        if not isinstance(mode, str) or mode not in _MODES:
            mode = "observe"
        usage_tracking = getattr(config, "usage_tracking", True)
        if not isinstance(usage_tracking, bool):
            usage_tracking = True
        max_cache_breakpoints = getattr(config, "max_cache_breakpoints", 4)
        if isinstance(max_cache_breakpoints, bool) or not isinstance(max_cache_breakpoints, int):
            max_cache_breakpoints = 4
        return cls(
            mode=mode,
            telemetry_dir=telemetry_dir,
            persist=usage_tracking,
            max_cache_breakpoints=max_cache_breakpoints,
            provider=provider,
        )

    @classmethod
    def disabled(cls) -> "CallEfficiency":
        return cls(mode="off", persist=False)

    @property
    def records(self) -> tuple[CallRecord, ...]:
        return tuple(self.ledger.records)

    def prepare(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
        *,
        provider: LLMProvider | None = None,
    ) -> PreparedCall:
        if self.mode != "optimize":
            policy = "disabled" if self.mode == "off" else "observe_only"
            return PreparedCall(messages, tools, model, policy)
        capability_provider = provider or self.provider
        explicit_supported = (
            capability_provider.supports_explicit_cache_control(model) if capability_provider is not None else None
        )
        capability = cache_capability(
            model,
            explicit_cache_control_supported=explicit_supported,
        )
        replanned = False
        if has_cache_control(messages, tools):
            if capability is CacheCapability.ANTHROPIC_EXPLICIT and valid_cache_control(
                messages,
                tools,
                max_breakpoints=self.max_cache_breakpoints,
            ):
                return PreparedCall(messages, tools, model, "external_cache_control")
            messages, tools = strip_cache_control(messages, tools)
            replanned = capability is CacheCapability.ANTHROPIC_EXPLICIT
        if capability is CacheCapability.ANTHROPIC_EXPLICIT:
            planned_messages, planned_tools = apply_anthropic_cache_plan(
                messages,
                tools,
                max_breakpoints=self.max_cache_breakpoints,
            )
            policy = "anthropic_explicit_replanned_v1" if replanned else "anthropic_explicit_v1"
            return PreparedCall(planned_messages, planned_tools, model, policy)
        if capability is CacheCapability.PROVIDER_AUTOMATIC:
            return PreparedCall(messages, tools, model, "provider_automatic")
        return PreparedCall(messages, tools, model, "unsupported")

    def record(
        self,
        response: LLMResponse,
        *,
        requested_model: str,
        attempted_model: str | None = None,
        session_key: str | None,
        cache_policy: str | None = None,
        trace_id: str | None = None,
        turn_span_id: str | None = None,
        outcome: str | None = None,
    ) -> CallRecord:
        attempted_model = attempted_model or requested_model
        actual_model = response.model or attempted_model
        accounting_model = _accounting_model(attempted_model, actual_model)
        usage, findings = normalize_usage(response.usage, accounting_model)
        cost = None
        all_findings = list(findings)
        if usage.complete:
            cost = estimate_cost_usd(
                accounting_model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read_tokens,
                usage.cache_write_tokens,
                allow_litellm_import=False,
            )
            if cost is None:
                all_findings.append("pricing_unavailable")
        ctx = trace.current()
        record = CallRecord(
            requested_model=requested_model,
            attempted_model=attempted_model,
            actual_model=actual_model,
            accounting_model=accounting_model,
            usage=usage,
            estimated_cost_usd=cost,
            outcome=outcome or ("error" if response.finish_reason == "error" else "success"),
            finish_reason=response.finish_reason,
            error_category=getattr(response.error_classification, "category", None),
            session_key=session_key or None,
            trace_id=trace_id if trace_id is not None else getattr(ctx, "trace_id", None),
            turn_span_id=turn_span_id if turn_span_id is not None else getattr(ctx, "parent_span_id", None),
            mode=self.mode,
            cache_policy=cache_policy or ("disabled" if self.mode == "off" else "observe_only"),
            observed_at=datetime.now(timezone.utc).isoformat(),
            findings=tuple(all_findings),
        )
        if self.mode != "off":
            try:
                self.ledger.append(record)
            except Exception:
                # 记账证据不得把已完成的 Provider 调用变成失败的 Turn。
                logger.exception("CallEfficiency could not persist a Call Record")
        return record

    def close(self) -> None:
        self.ledger.close()


def _accounting_model(requested_model: str, actual_model: str) -> str:
    if requested_model.startswith("openrouter/") and not actual_model.startswith("openrouter/"):
        return f"openrouter/{actual_model}"
    return actual_model


__all__ = ["CallEfficiency"]
