"""Shared CLI helpers used by multiple top-level command modules.

Extracted from commands.py so that per-command modules
(``agent_commands.py``, ``gateway_commands.py``, ``skill_commands.py``)
can import them directly instead of going
through lazy wrappers.

Function names drop the leading underscore: the file itself is marked
internal with the ``_helpers`` prefix, so members do not also need the
private-name convention.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import typer
from rich.console import Console

from codeflow.config.schema import Config

console = Console()


DEFAULT_PROBE_MESSAGE = "Hi! Say hello in one sentence."


def warn_about_pending_cli_reminders(cron_service, config: Config) -> None:
    """At REPL exit, list cron jobs pinned to channel="cli" that won't fire
    while the REPL is down. Hint at the config knob that forwards them to
    a durable channel at trigger time."""
    from datetime import datetime

    try:
        jobs = cron_service.list_jobs()
    except Exception:
        return
    now_ms = int(datetime.now().timestamp() * 1000)
    pending = [
        j
        for j in jobs
        if (j.payload.channel or "") == "cli" and j.state.next_run_at_ms and j.state.next_run_at_ms > now_ms
    ]
    if not pending:
        return

    console.print(f"\n[yellow]⚠  You have {len(pending)} pending CLI reminder(s):[/yellow]")
    for j in pending:
        fire = datetime.fromtimestamp(j.state.next_run_at_ms / 1000).strftime("%H:%M")
        mins = max(0, (j.state.next_run_at_ms - now_ms) // 60_000)
        console.print(f"   - '{j.name}' at {fire} (in {mins} min)")

    if not config.cron.forward_channels:
        console.print(
            "[dim]   Tip: cron.forward_channels is empty — these reminders will "
            "be dropped silently when they fire. Run "
            "`codeflow cron config set forward_channels '*'` to broadcast to "
            "all enabled channels.[/dim]"
        )


def check_provider_credentials(config: Config) -> None:
    """Fail-fast when the configured provider is missing required credentials.

    Cheap (no litellm import), so it can run at startup even when the real
    provider is built lazily. Kept in sync with the branches of make_provider.
    """
    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)

    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        return
    if provider_name == "azure_openai":
        if not p or not p.api_key or not p.api_base:
            console.print("[red]Error: Azure OpenAI requires api_key and api_base.[/red]")
            console.print("Set them in ~/.codeflow/config.json under providers.azure_openai section")
            console.print("Use the model field to specify the deployment name.")
            raise typer.Exit(1)
        return
    from codeflow.providers.registry import find_by_name

    spec = find_by_name(provider_name)
    if not model.startswith("bedrock/") and not (p and p.api_key) and not (spec and (spec.is_oauth or spec.is_local)):
        console.print("[red]Error: No API key configured.[/red]")
        console.print("Set one in ~/.codeflow/config.json under providers section")
        raise typer.Exit(1)


def make_provider(config: Config):
    """Create the appropriate LLM provider from config."""
    from codeflow.providers.azure_openai_provider import AzureOpenAIProvider
    from codeflow.providers.base import GenerationSettings
    from codeflow.providers.openai_codex_provider import OpenAICodexProvider

    check_provider_credentials(config)

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)

    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        provider = OpenAICodexProvider(default_model=model)
    elif provider_name == "azure_openai":
        provider = AzureOpenAIProvider(
            api_key=p.api_key,
            api_base=p.api_base,
            default_model=model,
        )
    else:
        from codeflow.providers.litellm_provider import LiteLLMProvider

        # OpenRouter 会把 qwen3.x-27B 路由到默认启用推理模式的提供商（如 AtlasCloud）：每次对话
        # 补全都会产生约 800 个思维链令牌并耗时约 30 秒，这会严重影响交互使用和大规模基准运行。
        # ``reasoning.enabled=false`` 是 OpenRouter 专用参数，通过 LiteLLM 的 ``extra_body`` 转发。
        extra_body = None
        if provider_name == "openrouter" and "qwen" in (model or "").lower():
            extra_body = {"reasoning": {"enabled": False}}
        provider = LiteLLMProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            provider_name=provider_name,
            extra_body=extra_body,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    return provider


def make_lazy_provider(config: Config):
    """Provider that defers the real (litellm-importing) build to the first model
    call, so AgentLoop construction stays fast. Credentials are checked now
    (fail-fast preserved) and the real provider is pre-warmed in the background."""
    from codeflow.providers.base import GenerationSettings
    from codeflow.providers.lazy import LazyProvider

    check_provider_credentials(config)
    defaults = config.agents.defaults
    provider = LazyProvider(
        factory=lambda: make_provider(config),
        default_model=defaults.model,
        generation=GenerationSettings(
            temperature=defaults.temperature,
            max_tokens=defaults.max_tokens,
            reasoning_effort=defaults.reasoning_effort,
        ),
    )
    provider.prewarm()
    return provider


def send_probe(
    *,
    message: str = DEFAULT_PROBE_MESSAGE,
    timeout_s: int = 15,
    max_tokens: int = 200,
) -> tuple[str, int | None, float]:
    """Build provider from current config and exchange one chat message.

    Shared by ``onboard`` Step 3 and ``doctor --probe``. Bypasses the full
    ``AgentLoop`` so the probe only proves the provider answers, not that
    the agent runtime is healthy.

    Returns ``(response_text, tokens_used, elapsed_s)``. Raises ``RuntimeError``
    on provider error, ``asyncio.TimeoutError`` on timeout, or whatever
    ``load_config`` / ``make_provider`` raise on config failure.
    """
    from codeflow.config.loader import load_config

    config = load_config()
    provider = make_provider(config)

    start = time.monotonic()
    response = asyncio.run(
        asyncio.wait_for(
            provider.chat_with_retry(
                messages=[{"role": "user", "content": message}],
                max_tokens=max_tokens,
                temperature=0.3,
            ),
            timeout=timeout_s,
        )
    )
    elapsed = time.monotonic() - start

    if response.finish_reason == "error":
        raise RuntimeError(response.content or "provider returned an error")

    usage = response.usage or {}
    tokens = usage.get("total_tokens") or usage.get("completion_tokens")
    return (response.content or "").strip(), tokens, elapsed


def print_probe_troubleshooting(provider: str | None) -> None:
    """Common-case hints when a probe fails.

    Shared by ``onboard`` Step 3 and ``doctor --probe`` so the diagnostic
    advice stays in one place.
    """
    console.print("\n  [dim]Troubleshooting:[/dim]")
    if provider:
        console.print(
            f"  [dim]·[/dim] [cyan]codeflow provider test {provider}[/cyan] — re-check credentials without spending tokens"
        )
        console.print(
            f"  [dim]·[/dim] [cyan]codeflow provider get {provider}[/cyan] — inspect what's actually stored on disk"
        )
    console.print(
        "  [dim]·[/dim] Check the model id in [cyan]~/.codeflow/config.json[/cyan] "
        "under [cyan]agents.defaults.model[/cyan] — it should match a model the "
        "provider serves."
    )


def load_runtime_config(config: str | None = None, workspace: str | None = None) -> Config:
    """Load config and optionally override the active workspace."""
    from codeflow.config.loader import load_config, set_config_path

    config_path = None
    if config:
        config_path = Path(config).expanduser().resolve()
        if not config_path.exists():
            console.print(f"[red]Error: Config file not found: {config_path}[/red]")
            raise typer.Exit(1)
        set_config_path(config_path)
        Console(stderr=True).print(f"[dim]Using config: {config_path}[/dim]")

    loaded = load_config(config_path)
    if workspace:
        loaded.agents.defaults.workspace = workspace
    return loaded


def print_deprecated_memory_window_notice(config: Config) -> None:
    """Warn when running with old memoryWindow-only config."""
    if config.agents.defaults.should_warn_deprecated_memory_window:
        console.print(
            "[yellow]Hint:[/yellow] Detected deprecated `memoryWindow` without "
            "`contextWindowTokens`. `memoryWindow` is ignored; run "
            "[cyan]codeflow onboard[/cyan] to refresh your config template."
        )


__all__ = [
    "DEFAULT_PROBE_MESSAGE",
    "warn_about_pending_cli_reminders",
    "make_provider",
    "send_probe",
    "print_probe_troubleshooting",
    "load_runtime_config",
    "print_deprecated_memory_window_notice",
]
