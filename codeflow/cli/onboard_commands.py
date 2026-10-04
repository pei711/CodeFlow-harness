"""Four-step onboarding wizard: provider → Memory + first Turn → sandbox → channel.

Goal: get a new user from ``pip install`` to a working agent in a few
minutes, without ever opening ``~/.codeflow/config.json`` or
the Myna repository binding.

Steps:
  0. Welcome
  1. LLM provider (required; multi-provider, credential check + model choice)
  2. Myna repository Memory (default) and first real Runtime Turn
  3. Sandbox / run location (optional, single-select)
  4. Chat channel (optional, stackable)
  5. Done

All writes go through the ``update_providers`` / ``update_channels`` /
``update`` ops libraries. This module owns the UX layer, not config-schema
knowledge.

Navigation: questionary 2.1.1 has no first-class cross-screen "back", so the
wizard is a screen state machine. Step 1 can return to language selection,
Step 2 returns to Step 1 when the first Turn needs new Provider settings, and
Step 3 exposes an explicit back choice. Step 4 is forward-only; re-run
``onboard`` to change it. Ctrl+C exits at any point, keeping durable writes.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from codeflow.cli._helpers import (
    DEFAULT_PROBE_MESSAGE,
    print_probe_troubleshooting,
)

console = Console()

_TOTAL_STEPS = 4

# 页面函数返回此哨兵，要求运行器退回上一页；选择器返回 ``None`` 表示 Ctrl+C（退出）。
_BACK = object()

# 统一提示外观（仅影响显示）：去掉开头的提问符号（弃用 questionary 默认的 "?"）。
# 单空格 qmark 会渲染为一个空白，加上 questionary 自带的前导空格，使所有提示行与
# 帮助/状态输出一样从第 2 列开始，左边缘保持对齐，不再在一格和两格缩进之间跳动。
# 指针使用比 questionary 默认 "»" 更平和的 "❯"。
_QMARK = " "
_POINTER = "❯"

# UI 语言在向导第一页选择。``_t`` 返回英文或中文版本，使后续所有提示和消息保持双语。
_LANG = "en"


def _t(en: str, zh: str) -> str:
    """Return ``zh`` when the user picked Chinese, else ``en``."""
    return zh if _LANG == "zh" else en


# ---------------------------------------------------------------------------
# 第一步选择器中展示的精选提供商目录。
# ---------------------------------------------------------------------------


_CURATED_PROVIDERS: list[dict[str, Any]] = [
    {
        "name": "openrouter",
        "label": "OpenRouter (recommended — one key, many models)",
        "label_zh": "OpenRouter(推荐 · 一个 Key 调用多家模型)",
        "is_oauth": False,
    },
    {"name": "openai", "label": "OpenAI", "label_zh": "OpenAI", "is_oauth": False},
    {"name": "anthropic", "label": "Anthropic", "label_zh": "Anthropic", "is_oauth": False},
    {"name": "gemini", "label": "Gemini", "label_zh": "Gemini", "is_oauth": False},
    {"name": "deepseek", "label": "DeepSeek", "label_zh": "DeepSeek", "is_oauth": False},
    {
        "name": "github_copilot",
        "label": "GitHub Copilot (OAuth)",
        "label_zh": "GitHub Copilot(OAuth 登录)",
        "is_oauth": True,
    },
    {
        "name": "openai_codex",
        "label": "Codex (OAuth)",
        "label_zh": "Codex(OAuth 登录)",
        "is_oauth": True,
    },
    {
        "name": "custom",
        "label": "Other (OpenAI-compatible endpoint)",
        "label_zh": "其他(OpenAI 兼容端点)",
        "is_oauth": False,
    },
]

_QUESTIONARY_INSTALL_HINT = (
    "[red]Missing dependency:[/red] [#fbe23f]questionary[/#fbe23f] is required for "
    "interactive onboarding.\n"
    "Install it with: [#fbe23f]uv add 'questionary>=2.0,<3.0'[/#fbe23f]\n"
    "Or re-run with [#fbe23f]--non-interactive[/#fbe23f] plus the relevant flags."
)


_PROMPT_THEMED = False


def _theme_questionary(questionary: Any) -> None:
    """Give every ``select`` a consistent pointer and drop questionary's own
    "(Use arrow keys)" hint — the step header already prints the controls.

    Display-only and applied once: we wrap ``questionary.select`` so callers
    that don't pass ``pointer`` / ``instruction`` inherit the unified look,
    while any explicit value still wins (``setdefault``).
    """
    global _PROMPT_THEMED
    if _PROMPT_THEMED:
        return
    import functools

    _orig_select = questionary.select

    @functools.wraps(_orig_select)
    def _themed_select(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("pointer", _POINTER)
        # instruction 为假值时，questionary 会显示 "(Use arrow keys)"；单空格为真值但视觉上
        # 为空，因此可隐藏该提示（步骤标题已显示操作方法）。
        kwargs.setdefault("instruction", " ")
        return _orig_select(*args, **kwargs)

    questionary.select = _themed_select
    _PROMPT_THEMED = True


def _require_questionary() -> Any:
    """Lazy-import :mod:`questionary` so missing-package errors stay scoped here."""
    try:
        import questionary
    except ModuleNotFoundError:
        console.print(_QUESTIONARY_INSTALL_HINT)
        raise typer.Exit(1)
    _theme_questionary(questionary)
    return questionary


def _config_language() -> str:
    """Read the saved UI language from the on-disk config ('en' / 'zh').

    A missing / empty config (fresh install) defaults to 'en'; a malformed one
    raises ConfigReadError (surfaced by the CLI entrypoint) rather than being
    silently read as empty.
    """
    data = _load_raw_config()
    lang = data.get("language")
    return lang if lang in ("en", "zh") else "en"


def _pick_language() -> None:
    """First screen: choose the wizard's language. Updates module-level ``_LANG``.

    Persistence happens later (after bootstrap created the config file), via
    ``set_language`` in :func:`_run_wizard_body`.
    """
    global _LANG
    questionary = _require_questionary()
    from codeflow.cli._styles import CODEFLOW_STYLE

    # 与其他页面采用相同框架（此时尚未选择语言，故使用双语），让它看起来是向导的
    # 第一步，而不是孤立悬浮的列表。
    console.print()
    console.print(
        Panel(
            "[bold white]Let's set up CodeFlow — first, choose your language.[/bold white]\n"
            "[dim]开始配置 CodeFlow — 请先选择语言。[/dim]",
            title="[bold #fbe23f]CodeFlow setup[/bold #fbe23f]",
            title_align="left",
            border_style="#c8a900",
            padding=(1, 2),
        )
    )
    console.print("  [dim]↑↓ select · Enter confirm · Ctrl+C quit[/dim]")
    console.print()

    picked = questionary.select(
        "Language / 语言",
        choices=[
            questionary.Choice("English", value="en"),
            questionary.Choice("中文(简体)", value="zh"),
        ],
        default=_LANG,  # 再次运行时预选已保存的语言
        style=CODEFLOW_STYLE,
        qmark=_QMARK,
    ).ask()
    if picked is None:
        raise typer.Exit(1)
    _LANG = picked


# ---------------------------------------------------------------------------
# 共用辅助方法
# ---------------------------------------------------------------------------


def _step_header(n: int, title: str) -> None:
    # 进度点：已完成和当前步骤为实心，后续步骤为空心。
    dots = " ".join("[#fbe23f]●[/#fbe23f]" if i <= n else "[grey37]○[/grey37]" for i in range(1, _TOTAL_STEPS + 1))
    console.print()
    console.print(
        Panel(
            f"[bold white]{title}[/bold white]",
            title=f"[bold #fbe23f]{_t('Step', '步骤')} {n}/{_TOTAL_STEPS}[/bold #fbe23f]",
            title_align="left",
            subtitle=dots,
            subtitle_align="right",
            border_style="#c8a900",
            padding=(0, 2),
        )
    )
    console.print()  # 在标题和步骤提示之间留出呼吸空间


def _check_tty_or_die(non_interactive: bool) -> None:
    """Bail when stdout isn't a TTY and the user didn't opt into headless mode."""
    if non_interactive:
        return
    if not sys.stdout.isatty():
        console.print(
            "[red]Non-interactive terminal detected.[/red]\n"
            "Re-run with: "
            "[#fbe23f]codeflow onboard --non-interactive --provider <name> --api-key <key>[/#fbe23f]"
        )
        raise typer.Exit(2)


def _load_raw_config() -> dict[str, Any]:
    """Return the parsed on-disk config, or ``{}`` if absent/empty.

    A present-but-unparseable config raises ConfigReadError (surfaced cleanly by
    the CLI entrypoint) instead of being silently treated as empty -- which
    would let onboard misread state and write over a config whose only fault is
    a syntax typo.
    """
    from codeflow.config.loader import get_config_path, read_raw_or_raise

    return read_raw_or_raise(get_config_path()) or {}


def _configured_providers() -> list[str]:
    """Names of providers that currently have an api_key set on disk."""
    data = _load_raw_config()
    providers = data.get("providers") or {}
    return [name for name, p in providers.items() if isinstance(p, dict) and p.get("apiKey")]


def _is_config_populated() -> bool:
    """True iff at least one provider has a key AND a default model is set.

    "Populated" for the startup gate means the required step (Step 1) is
    satisfied: a provider key plus ``agents.defaults.model``. Either alone is
    not enough to talk to a model.
    """
    data = _load_raw_config()
    providers = data.get("providers") or {}
    has_provider = any(isinstance(p, dict) and p.get("apiKey") for p in providers.values())
    model = (data.get("agents", {}) or {}).get("defaults", {}).get("model")
    return bool(has_provider and model)


def _handle_existing_config(*, reset: bool, yes: bool, non_interactive: bool) -> None:
    """Guard against silently overwriting an existing config in non-interactive
    runs.

    Interactive runs always fall through into the structured wizard: every step
    defaults to "Keep current" for already-set values, so pressing Enter all the
    way through is equivalent to skipping, and changing any value reconfigures
    just that one. No separate skip/redo/quit screen — it would drop the wizard's
    welcome banner and step framing.
    """
    if reset:
        return
    if not _is_config_populated():
        return

    if non_interactive:
        if yes:
            console.print("[dim]Existing config detected; --yes set, proceeding with overwrite.[/dim]")
            return
        console.print(
            "[red]Existing config detected.[/red] Pass [#fbe23f]--reset[/#fbe23f] (or "
            "[#fbe23f]--yes[/#fbe23f]) to overwrite, or edit in place with "
            "[#fbe23f]codeflow provider set[/#fbe23f] / [#fbe23f]codeflow channels enable[/#fbe23f]."
        )
        raise typer.Exit(2)
    # 交互模式：继续进入向导（各步骤的“保留当前值”可妥善处理现有配置）。


def _bootstrap_empty_config() -> None:
    """Make sure ``~/.codeflow/config.json`` + workspace dir exist before we patch.

    We seed the user-facing extension defaults (memory / plugins / skillForge),
    including ``memory.backend = "myna"``. Step 4 keeps that selection or
    writes ``None`` when the user explicitly disables Memory.

    Seeding runs on EVERY onboard, not just a brand-new config: the writer is
    ``setdefault``-based (non-clobbering), so it backfills these blocks into a
    pre-existing config that predates them without touching any value the user
    already set. The base ``Config()`` is only written when the file is absent —
    overwriting an existing file there would clobber it.
    """
    from codeflow.config.loader import get_config_path, load_config, save_config
    from codeflow.config.paths import get_workspace_path
    from codeflow.utils.helpers import sync_workspace_templates

    path = get_config_path()
    if not path.exists():
        save_config(load_config())  # 将默认 Config() 写入磁盘
    _init_extension_block_defaults()
    workspace = get_workspace_path()
    workspace.mkdir(parents=True, exist_ok=True)
    sync_workspace_templates(workspace)


# ---------------------------------------------------------------------------
# 步骤 1——提供商基础操作（三步向导原样复用）
# ---------------------------------------------------------------------------


def _provider_label(name: str) -> str:
    """Display label for a provider, falling back to the registry's display_name."""
    for entry in _CURATED_PROVIDERS:
        if entry["name"] == name:
            return _t(entry["label"], entry.get("label_zh", entry["label"]))
    try:
        from codeflow.providers.registry import find_by_name

        spec = find_by_name(name)
        return spec.label if spec else name
    except Exception:
        return name


def _validate_provider_name(name: str) -> str:
    """Resolve a user-supplied provider name (kebab or snake) to a registry key."""
    from codeflow.config.update_providers import provider_field_specs

    candidate = name.replace("-", "_")
    try:
        provider_field_specs(candidate)
    except KeyError as exc:
        raise typer.BadParameter(str(exc))
    return candidate


def _back_placeholder(allow_back: bool) -> Any:
    """A faint in-field placeholder telling the user an empty submit rewinds.

    Rendered greyed inside the input (via prompt_toolkit's ``placeholder``),
    it disappears the moment they type and leaves nothing behind once the
    prompt is answered. Returns ``None`` when back isn't offered.
    """
    if not allow_back:
        return None
    return [("fg:#6c6c6c italic", _t("empty ↵ to go back", "留空回车返回上一步"))]


def _field_placeholder(allow_back: bool, required: bool) -> Any:
    """In-field hint for a channel credential prompt.

    First field: empty submit rewinds to the channel picker (back). Later
    optional fields: empty submit skips them. Required later fields get no
    hint — an empty submit there silently drops a value the channel needs.
    """
    if allow_back:
        return _back_placeholder(True)
    if not required:
        return [("fg:#6c6c6c italic", _t("empty ↵ to skip", "留空回车跳过"))]
    return None


def _collect_fields(prompts: list[Callable[[], Any]]) -> Optional[list[Any]]:
    """Run text-prompt callables in order with empty-submit = back.

    Each callable prompts one field and returns its value, or ``_BACK`` (an
    empty submit) to rewind one field. Backing out of the first field returns
    ``None`` so the caller can rewind to the preceding screen. Returns the list
    of collected values on success.
    """
    values: list[Any] = []
    i = 0
    while i < len(prompts):
        value = prompts[i]()
        if value is _BACK:
            if i == 0:
                return None
            values.pop()
            i -= 1
            continue
        if i < len(values):
            values[i] = value
        else:
            values.append(value)
        i += 1
    return values


def _select_provider() -> Optional[str]:
    """Interactive provider picker built from the curated catalogue.

    Returns the provider name, ``_BACK`` if the user chose the back sentinel,
    or ``None`` on Ctrl+C.
    """
    questionary = _require_questionary()
    from codeflow.cli._styles import CODEFLOW_STYLE

    choices: list[Any] = [
        questionary.Choice(
            _t(entry["label"], entry.get("label_zh", entry["label"])),
            value=entry["name"],
        )
        for entry in _CURATED_PROVIDERS
    ]
    choices.append(questionary.Separator())
    choices.append(questionary.Choice(_t("Back", "返回"), value=_BACK))

    picked = questionary.select(
        _t("Provider:", "服务商:"),
        choices=choices,
        style=CODEFLOW_STYLE,
        qmark=_QMARK,
    ).ask()
    return picked  # Ctrl+C 时为 None


def _prompt_api_key(provider: str, *, allow_back: bool = False) -> Any:
    """Ask for an API key (hidden input). Returns ``_BACK`` on empty submit
    when ``allow_back`` is set, else the key string."""
    questionary = _require_questionary()
    from codeflow.cli._styles import CODEFLOW_STYLE

    def _validate(v: str) -> Any:
        if allow_back and v == "":
            return True  # 空值表示返回，而不是错误
        return (
            True
            if len(v) >= 8
            else _t(
                "API key looks off (empty or too short) — please re-enter (≥ 8 chars).",
                "API Key 看起来不对(过短或为空),请重新输入(至少 8 位)。",
            )
        )

    key = questionary.password(
        _t("Paste your API key:", "粘贴你的 API Key:"),
        validate=_validate,
        placeholder=_back_placeholder(allow_back),
        style=CODEFLOW_STYLE,
        qmark=_QMARK,
    ).ask()
    if key is None:
        raise typer.Exit(1)
    key = key.strip()
    if allow_back and key == "":
        return _BACK
    if not key:
        raise typer.Exit(1)
    return key


def _prompt_base_url(default: str = "https://", *, allow_back: bool = False) -> Any:
    """Ask for an OpenAI-compatible base URL (used by the 'custom' provider).
    Returns ``_BACK`` on empty submit when ``allow_back`` is set."""
    questionary = _require_questionary()
    from codeflow.cli._styles import CODEFLOW_STYLE

    # 启用返回时不要预填默认值——必须允许输入框为空，用户才能提交空值退回。
    seed = "" if allow_back else default

    def _validate(v: str) -> Any:
        if allow_back and v == "":
            return True
        return (
            True
            if v.startswith(("http://", "https://"))
            else _t("URL must start with http:// or https://", "地址需以 http:// 或 https:// 开头")
        )

    url = questionary.text(
        _t("Base URL (must include /v1):", "Base URL(需包含 /v1):"),
        default=seed,
        validate=_validate,
        placeholder=_back_placeholder(allow_back),
        style=CODEFLOW_STYLE,
        qmark=_QMARK,
    ).ask()
    if url is None:
        raise typer.Exit(1)
    url = url.strip()
    if allow_back and url == "":
        return _BACK
    if not url:
        raise typer.Exit(1)
    return url


def _prompt_custom_model(*, allow_back: bool = False) -> Any:
    """Ask for the model name when using a custom OpenAI-compatible endpoint.
    Returns ``_BACK`` on empty submit when ``allow_back`` is set."""
    questionary = _require_questionary()
    from codeflow.cli._styles import CODEFLOW_STYLE

    def _validate(v: str) -> Any:
        if allow_back and v.strip() == "":
            return True
        return True if v.strip() else _t("Model id is required for custom endpoints.", "自定义端点必须指定模型 id。")

    model = questionary.text(
        _t(
            "Default model id (e.g. 'gpt-3.5-turbo' or 'qwen-max'):",
            "默认模型 id(如 'gpt-3.5-turbo' 或 'qwen-max'):",
        ),
        validate=_validate,
        placeholder=_back_placeholder(allow_back),
        style=CODEFLOW_STYLE,
        qmark=_QMARK,
    ).ask()
    if model is None:
        raise typer.Exit(1)
    if allow_back and model.strip() == "":
        return _BACK
    if not model:
        raise typer.Exit(1)
    return model.strip()


def _run_oauth_login(provider: str) -> bool:
    """Dispatch the OAuth login handler registered by ``provider_commands``.

    Returns ``True`` on success. A login that fails (the handler raises
    ``typer.Exit`` or any error) returns ``False`` so the caller can offer a
    retry / back menu instead of tearing the whole wizard down. A genuine
    Ctrl+C (``KeyboardInterrupt``) is left to propagate as a quit.
    """
    from codeflow.cli.provider_commands import _LOGIN_HANDLERS
    from codeflow.providers.registry import find_by_name

    spec = find_by_name(provider)
    if not spec or not spec.is_oauth:
        console.print(
            _t(
                f"  [red]✗ {provider} is not an OAuth provider.[/red]",
                f"  [red]✗ {provider} 不是 OAuth 服务商。[/red]",
            )
        )
        raise typer.Exit(1)
    handler = _LOGIN_HANDLERS.get(spec.name)
    if not handler:
        console.print(
            _t(
                f"  [red]✗ No login handler registered for {provider}.[/red]",
                f"  [red]✗ 未为 {provider} 注册登录处理器。[/red]",
            )
        )
        raise typer.Exit(1)
    console.print(
        _t(
            f"  [#fbe23f]Starting OAuth login for {spec.label}…[/#fbe23f]\n",
            f"  [#fbe23f]正在为 {spec.label} 启动 OAuth 登录…[/#fbe23f]\n",
        )
    )
    console.print(
        _t(
            "  [dim]A browser window / link will open — finish the sign-in there, "
            "then come back here. This waits until you're done.[/dim]\n",
            "  [dim]会打开浏览器窗口 / 链接 — 在那里完成登录后回到这里;这里会一直等到你完成。[/dim]\n",
        )
    )
    try:
        handler()
    except typer.Exit as exc:
        # 处理器以 Exit(1) 表示登录失败；Exit(0)（若有）表示成功。
        if exc.exit_code:
            return False
    except Exception as exc:  # 网络、浏览器或令牌错误，可恢复
        console.print(
            _t(
                f"  [yellow]✗ Login didn't complete: {exc}[/yellow]",
                f"  [yellow]✗ 登录未完成:{exc}[/yellow]",
            )
        )
        return False
    return True


def _verify_provider(provider: str, *, skip_test: bool = False) -> tuple[bool, str, Optional[list[str]]]:
    """Hit ``GET /v1/models`` to verify the credentials we just stored.

    Returns ``(ok, status, model_ids)``. ``status`` is one of the ops-library
    failure codes (``invalid_key`` / ``no_credits`` / ``rate_limited`` /
    ``network_error`` / …) and drives the failure submenu's wording.
    """
    from codeflow.config.update_providers import test_provider as probe

    console.print(_t("  [dim]⏳ Verifying your API key…[/dim]", "  [dim]⏳ 正在验证 API Key…[/dim]"))
    result = probe(provider)
    if result["ok"]:
        models = result.get("models_count")
        suffix = _t(f" ({models} models available)", f"(共 {models} 个可用模型)") if models else ""
        console.print(_t(f"  [green]✓ Connected!{suffix}[/green]", f"  [green]✓ 连接成功!{suffix}[/green]"))
        return True, "valid", result.get("model_ids")

    status = result.get("status", "unknown")
    # 一些直连提供商（openai / anthropic / deepseek / gemini）不提供 base URL，
    # 而依赖 SDK 内置端点，因此无法通过 GET /v1/models 预检；api_base 为空会让探针
    # 报告 "not_configured"。这不是真实的认证失败：应跳过预检（稍后发送的测试消息会
    # 通过 litellm 验证真实连接），而不是把用户带入失败子菜单。
    if status == "not_configured" and "api_base" in (result.get("error") or ""):
        if skip_test:
            console.print(
                _t(
                    "  [dim]Skipping the model-list pre-check (this provider has no public /models endpoint); connectivity is not tested (--skip-test).[/dim]",
                    "  [dim]跳过模型列表预检(该服务商无公开 /models 端点);未做连通测试(--skip-test)。[/dim]",
                )
            )
        else:
            console.print(
                _t(
                    "  [dim]Skipping the model-list pre-check (this provider has no public /models endpoint); the test message below will confirm connectivity.[/dim]",
                    "  [dim]跳过模型列表预检(该服务商无公开 /models 端点);稍后的测试消息会验证连通。[/dim]",
                )
            )
        return True, "skipped", None
    hint_map = {
        "invalid_key": _t(
            "Auth failed: the API key is invalid — check for typos / stray spaces.",
            "鉴权失败:API Key 无效 — 检查有无拼写错误或多余空格。",
        ),
        "no_credits": _t(
            "Account out of credits or not provisioned — top up and retry.",
            "账户余额不足或未开通 — 充值后重试。",
        ),
        "rate_limited": _t(
            "Rate limited — wait a bit and retry, or switch provider.",
            "触发限流 — 稍等后重试,或更换服务商。",
        ),
        "network_error": _t(
            "Network error reaching the provider — check network / proxy / VPN.",
            "连接服务商时网络出错 — 检查网络 / 代理 / VPN。",
        ),
        "oauth_token_missing": _t(
            f"Run: codeflow provider login {provider.replace('_', '-')}",
            f"请运行:codeflow provider login {provider.replace('_', '-')}",
        ),
    }
    msg = hint_map.get(status, _t(f"Verification failed: {status}", f"验证失败:{status}"))
    console.print(f"  [yellow]✗ {msg}[/yellow]" + (f"  [dim]{result['error']}[/dim]" if result.get("error") else ""))
    return False, status, None


def _load_current_default_model() -> Optional[str]:
    """Read ``agents.defaults.model`` from the on-disk config, if it exists."""
    data = _load_raw_config()
    return data.get("agents", {}).get("defaults", {}).get("model") or None


def _model_routes_to_provider(model: str, spec: Any) -> bool:
    """True if ``model`` would auto-route to ``spec`` under ``provider='auto'``."""
    if not model or not spec:
        return False
    model_lower = model.lower()
    model_normalized = model_lower.replace("-", "_")
    if "/" in model_lower:
        prefix = model_lower.split("/", 1)[0].replace("-", "_")
        return prefix == spec.name
    return any(
        kw.lower() in model_lower or kw.lower().replace("-", "_") in model_normalized
        for kw in (getattr(spec, "keywords", None) or ())
    )


def _format_model_for_provider(spec: Any, model_id: str) -> str:
    """Apply ``spec.litellm_prefix`` to a raw ``/v1/models`` id when needed."""
    if not model_id:
        return model_id
    prefix = getattr(spec, "litellm_prefix", "") or ""
    if not prefix:
        return model_id
    if model_id.startswith(f"{prefix}/"):
        return model_id
    for skip in getattr(spec, "skip_prefixes", ()) or ():
        if model_id.startswith(skip):
            return model_id
    return f"{prefix}/{model_id}"


def _pick_model(
    spec: Any,
    *,
    current_model: Optional[str],
    model_ids: Optional[list[str]],
    user_provided_model: Optional[str],
    non_interactive: bool,
) -> str:
    """Decide the model string to write into ``agents.defaults.model``."""
    if user_provided_model:
        return user_provided_model

    if non_interactive:
        if not spec.default_model:
            raise typer.BadParameter(
                f"--model is required for provider '{spec.name}' (no built-in default model in registry)."
            )
        return spec.default_model

    questionary = _require_questionary()
    from codeflow.cli._styles import CODEFLOW_STYLE

    if current_model and _model_routes_to_provider(current_model, spec):
        default_value = current_model
    else:
        default_value = spec.default_model or ""

    if model_ids:
        choices = [_format_model_for_provider(spec, mid) for mid in model_ids]
        if default_value and default_value not in choices:
            choices.insert(0, default_value)
        prompt_label = _t(
            f"Default model ({len(choices)} available — type to filter, Tab to complete):",
            f"默认模型(共 {len(choices)} 个 — 输入可筛选,Tab 补全):",
        )
        chosen = questionary.autocomplete(
            prompt_label,
            choices=choices,
            default=default_value,
            style=CODEFLOW_STYLE,
            qmark=_QMARK,
            ignore_case=True,
            match_middle=True,
        ).ask()
    else:
        console.print(
            _t(
                "  [dim]Couldn't fetch the model list — enter the model id by hand.[/dim]",
                "  [dim]未能拉取模型列表,请手动输入模型 id。[/dim]",
            )
        )
        if default_value:
            chosen = questionary.text(
                _t(
                    f"Default model (press Enter for [{default_value}]):",
                    f"默认模型(回车使用 [{default_value}]):",
                ),
                default=default_value,
                style=CODEFLOW_STYLE,
                qmark=_QMARK,
            ).ask()
        else:
            chosen = questionary.text(
                _t(f"Default model id for {spec.name}:", f"{spec.name} 的默认模型 id:"),
                validate=lambda v: True if v.strip() else _t("Model id is required.", "必须指定模型 id。"),
                style=CODEFLOW_STYLE,
                qmark=_QMARK,
            ).ask()

    if chosen is None:
        raise typer.Exit(1)  # 用户按下 Ctrl+C
    chosen = chosen.strip()
    if not chosen:
        # 提交空值（例如清除了预填默认值）时回退到默认值，而不是终止向导。无默认值的
        # 分支会校验非空，因此只有存在默认值时，空值才会到达这里。
        if default_value:
            return default_value
        raise typer.Exit(1)
    return chosen


def _write_provider_fields(provider: str, fields: dict[str, Any]) -> None:
    """Thin wrapper that surfaces ops-library errors with friendly hints."""
    from pydantic import ValidationError

    from codeflow.config.update_providers import set_provider_fields

    try:
        set_provider_fields(provider, fields)
    except KeyError as exc:
        console.print(f"  [red]✗[/red] {exc}")
        raise typer.Exit(1)
    except RuntimeError as exc:
        console.print(f"  [red]✗[/red] {exc}")
        raise typer.Exit(1)
    except ValidationError as exc:
        console.print(_t(f"  [red]✗ Validation failed:[/red]\n{exc}", f"  [red]✗ 校验失败:[/red]\n{exc}"))
        raise typer.Exit(1)


def _persist_default_model(model: Optional[str]) -> None:
    """Patch ``agents.defaults.model`` if we picked one."""
    if not model:
        return
    from codeflow.config.update import set_default_model

    set_default_model(model)


# ---------------------------------------------------------------------------
# 步骤 1——连接失败子菜单与测试探针
# ---------------------------------------------------------------------------


def _failure_choice(options: list[tuple[str, str]], *, non_interactive: bool) -> str:
    """Render a numbered failure submenu, return the chosen value.

    ``options`` is a list of ``(label, value)``. In non-interactive mode the
    last option (always "continue anyway") is auto-chosen so headless runs
    never block.
    """
    if non_interactive:
        return options[-1][1]
    questionary = _require_questionary()
    from codeflow.cli._styles import CODEFLOW_STYLE

    chosen = questionary.select(
        _t("What would you like to do?", "想做什么?"),
        choices=[questionary.Choice(label, value=value) for label, value in options],
        style=CODEFLOW_STYLE,
        qmark=_QMARK,
    ).ask()
    if chosen is None:
        raise typer.Exit(1)
    return chosen


def run_first_turn(
    *,
    message: str = DEFAULT_PROBE_MESSAGE,
    timeout_s: int = 120,
) -> tuple[str, int | None, float]:
    """Execute the onboarding message through the public Runtime path."""
    from codeflow.config.loader import get_config_path

    command = [
        sys.executable,
        "-m",
        "codeflow.cli.commands",
        "run",
        "-m",
        message,
        "--no-markdown",
        "--no-logs",
        "--config",
        str(get_config_path()),
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(detail[-2000:] or f"Runtime Turn exited {completed.returncode}")
    return completed.stdout.strip() or "Runtime Turn completed.", None, elapsed


def _run_test_probe(provider: str, *, non_interactive: bool, warnings: list[str], allow_repick: bool = True) -> str:
    """Run the first Runtime Turn; on failure offer recovery options.

    Returns one of ``"ok"`` / ``"continue"`` / ``"repick"`` / ``"rekey"`` /
    ``"switch"``. A test-message failure can be a wrong model, a bad key, or an
    account/balance issue, so the menu offers all the matching exits (aligning
    with the connectivity-failure menu in ``_resolve_model_with_test``);
    ``allow_repick=False`` drops the model option for custom providers whose
    model was fixed with the base_url upfront (Switch re-enters both).
    """
    console.print(
        _t(
            f'  [dim]Sending test message: "{DEFAULT_PROBE_MESSAGE}"[/dim]',
            f'  [dim]正在发送测试消息:"{DEFAULT_PROBE_MESSAGE}"[/dim]',
        )
    )
    try:
        text, tokens, elapsed = run_first_turn()
    except Exception as exc:
        console.print(_t(f"  [red]✗ Test failed:[/red] {exc}", f"  [red]✗ 测试失败:[/red] {exc}"))
        console.print(
            _t(
                "  [dim]Run 'codeflow provider test' to re-check, or confirm the model is served by this provider.[/dim]",
                "  [dim]可运行 'codeflow provider test' 复查,或确认该模型确由此服务商提供。[/dim]",
            )
        )
        print_probe_troubleshooting(provider)
        options = [(_t("Retry", "重试"), "retry")]
        if allow_repick:
            options.append((_t("Re-pick model", "重新选模型"), "repick"))
        options += [
            (_t("Re-enter key", "重新填 Key"), "rekey"),
            (_t("Switch provider", "更换服务商"), "switch"),
            (_t("Continue anyway", "仍然继续"), "continue"),
        ]
        choice = _failure_choice(options, non_interactive=non_interactive)
        if choice == "retry":
            return _run_test_probe(
                provider, non_interactive=non_interactive, warnings=warnings, allow_repick=allow_repick
            )
        if choice in ("repick", "rekey", "switch"):
            return choice
        warnings.append("first Runtime Turn")
        return "continue"

    console.print(f"  [bold]▶ Agent:[/bold] {text}")
    extras: list[str] = []
    if tokens:
        extras.append(f"{tokens} tokens")
    extras.append(f"{elapsed:.1f}s")
    console.print(f"  [green]✓ {', '.join(extras)}[/green]")
    return "ok"


# ---------------------------------------------------------------------------
# 步骤 1——添加一个提供商（首次运行和“添加”入口共用）
# ---------------------------------------------------------------------------


def _configure_one_provider(
    *,
    provider: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    non_interactive: bool,
    warnings: list[str],
    skip_test: bool = False,
) -> Optional[dict[str, Any]]:
    """Drive one provider through pick → credentials → verify → model → test.

    Returns ``{"provider", "model"}`` on success, or ``None`` if the user
    chose to go back from the interactive provider picker.
    """
    from codeflow.providers.registry import find_by_name

    # 循环处理连接失败时的“切换提供商”，使其退回选择器而非终止整个向导
    # （保留步骤 2/3/4）。通过参数传入的提供商只使用一次；之后切换必须使用
    # 交互选择器（非交互模式下则无法切换）。
    flag_provider = provider
    configured_before = set(_configured_providers())
    while True:
        if flag_provider:
            provider = _validate_provider_name(flag_provider)
        else:
            if non_interactive:
                raise typer.BadParameter("--provider is required in non-interactive mode")
            picked = _select_provider()
            if picked is None:
                raise typer.Exit(1)
            if picked is _BACK:
                return None
            provider = picked

        spec = find_by_name(provider)
        is_oauth = bool(spec and spec.is_oauth)
        is_custom = provider == "custom"
        # 交互选择器已回显所选提供商；只有来自 --provider 时才显式确认，因为此时没有回显。
        if flag_provider:
            console.print(
                _t(
                    f"  [dim]Provider:[/dim] [#fbe23f]{_provider_label(provider)}[/#fbe23f]",
                    f"  [dim]服务商:[/dim] [#fbe23f]{_provider_label(provider)}[/#fbe23f]",
                )
            )

        # 在 _collect_credentials 覆盖前保存现有密钥，使重新配置已有提供商失败时能恢复
        # 原先可用的密钥，而不是留下刚输入的错误密钥。
        _prev = (_load_raw_config().get("providers") or {}).get(provider) or {}
        old_key = _prev.get("apiKey")
        old_base = _prev.get("apiBase")

        custom_model = _collect_credentials(
            provider,
            is_oauth=is_oauth,
            is_custom=is_custom,
            api_key=api_key,
            base_url=base_url,
            model=model,
            non_interactive=non_interactive,
        )
        if custom_model is _BACK:
            # 用户从第一个凭据字段返回——退回提供商选择器（移除参数，确保选择器会显示）。
            flag_provider = None
            continue

        chosen_model = _resolve_model_with_test(
            spec,
            is_custom=is_custom,
            custom_model=custom_model,
            user_model_flag=model,
            non_interactive=non_interactive,
            warnings=warnings,
            skip_test=skip_test,
            defer_first_turn=True,
        )
        if chosen_model is None:
            # “切换提供商”——重新运行选择器（移除参数，让第二轮提示选择，而不是复用失败的
            # 参数值）。回滚刚写入的密钥：若本轮新添加该提供商则清空；若重新配置已有
            # 提供商则恢复旧密钥，避免失败的编辑覆盖可用配置。
            if provider not in configured_before:
                _write_provider_fields(provider, {"api_key": ""})
            elif old_key:
                _write_provider_fields(provider, {"api_key": old_key, "api_base": old_base})
            flag_provider = None
            continue
        _persist_default_model(chosen_model)
        return {"provider": provider, "model": chosen_model}


def _collect_credentials(
    provider: str,
    *,
    is_oauth: bool,
    is_custom: bool,
    api_key: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    non_interactive: bool,
) -> Any:
    """Auth setup: OAuth browser flow or api_key write. Returns the custom
    model id when the provider is ``custom`` (locked in here), ``None`` for a
    non-custom provider, or ``_BACK`` if the user backed out of the first
    interactive credential field (caller should rewind to the picker)."""
    if is_oauth:
        if non_interactive:
            console.print(
                "[red]OAuth providers require an interactive browser flow.[/red]\n"
                "Run [#fbe23f]codeflow provider login "
                f"{provider.replace('_', '-')}[/#fbe23f] separately, then re-run "
                "onboard."
            )
            raise typer.Exit(2)
        # 循环处理失败登录，以提供重试/返回选项，而不是直接退出。
        while True:
            if _run_oauth_login(provider):
                return None
            choice = _failure_choice(
                [
                    (_t("Retry", "重试"), "retry"),
                    (_t("Back (pick another provider)", "返回(改选服务商)"), "back"),
                ],
                non_interactive=non_interactive,
            )
            if choice == "retry":
                continue
            return _BACK

            # 纯交互路径（参数未提供凭据）：逐字段提示，提交空值表示返回；从第一个字段
            # 返回会退回提供商选择器。
    pure_interactive = not non_interactive and not api_key and (not is_custom or (not base_url and not model))
    if pure_interactive:
        prompts: list[Callable[[], Any]] = [lambda: _prompt_api_key(provider, allow_back=True)]
        if is_custom:
            prompts.append(lambda: _prompt_base_url(allow_back=True))
            prompts.append(lambda: _prompt_custom_model(allow_back=True))
        collected = _collect_fields(prompts)
        if collected is None:
            return _BACK
        api_key = collected[0]
        if is_custom:
            base_url = collected[1]
            model = collected[2]
    else:
        if not api_key:
            if non_interactive:
                raise typer.BadParameter("--api-key is required in non-interactive mode")
            api_key = _prompt_api_key(provider)
        if is_custom:
            if not base_url:
                if non_interactive:
                    raise typer.BadParameter("--base-url is required when --provider=custom in non-interactive mode")
                base_url = _prompt_base_url()
            if not model:
                if non_interactive:
                    raise typer.BadParameter("--model is required when --provider=custom in non-interactive mode")
                model = _prompt_custom_model()

    fields: dict[str, Any] = {"api_key": api_key}
    custom_model: Optional[str] = None
    if is_custom:
        fields["api_base"] = base_url
        custom_model = model
    elif base_url:
        fields["api_base"] = base_url

    _write_provider_fields(provider, fields)
    return custom_model


def _resolve_model_with_test(
    spec: Any,
    *,
    is_custom: bool,
    custom_model: Optional[str],
    user_model_flag: Optional[str],
    non_interactive: bool,
    warnings: list[str],
    skip_test: bool = False,
    defer_first_turn: bool = False,
) -> Optional[str]:
    """Verify credentials, pick the default model, and optionally send a test Turn.

    On a verify or test-message failure, offers a recovery submenu (retry /
    re-pick model / re-enter key / switch / continue). Custom providers are
    probed too (model was fixed upfront). Only failures stop; success
    auto-advances. Returns the chosen model, or ``None`` to signal "switch
    provider" (the caller rewinds to the picker).
    """
    while True:
        ok, status, model_ids = _verify_provider(spec.name, skip_test=skip_test)
        if not ok:
            options = (
                [(_t("Retry", "重试"), "retry"), (_t("Continue anyway", "仍然继续"), "continue")]
                if status == "network_error"
                else [
                    (_t("Re-enter key", "重新填 Key"), "rekey"),
                    (_t("Switch provider", "更换服务商"), "switch"),
                    (_t("Continue anyway", "仍然继续"), "continue"),
                ]
            )
            choice = _failure_choice(options, non_interactive=non_interactive)
            if choice == "retry":
                continue
            if choice == "rekey" and not non_interactive:
                _write_provider_fields(spec.name, {"api_key": _prompt_api_key(spec.name)})
                continue
            if choice == "switch":
                return None
            warnings.append("provider connectivity")
            model_ids = None
        break

    if is_custom:
        assert custom_model is not None, "custom provider must have model set earlier"
        # 自定义端点过去未发送测试消息便直接信任，这是最容易输错的场景。发送真实探针
        # （它根据已保存配置构建，因此错误的 base_url 或模型 ID 会在此处失败，而非首次对话时）。
        _persist_default_model(custom_model)
        if skip_test or defer_first_turn:
            return custom_model
        while True:
            result = _run_test_probe(spec.name, non_interactive=non_interactive, warnings=warnings, allow_repick=False)
            if result == "switch":
                return None
            if result == "rekey":
                _write_provider_fields(spec.name, {"api_key": _prompt_api_key(spec.name)})
                continue
            return custom_model  # 成功并继续

    current = _load_current_default_model()
    while True:
        chosen = _pick_model(
            spec,
            current_model=current,
            model_ids=model_ids,
            user_provided_model=user_model_flag,
            non_interactive=non_interactive,
        )
        _persist_default_model(chosen)
        if skip_test or defer_first_turn:
            return chosen
        result = _run_test_probe(spec.name, non_interactive=non_interactive, warnings=warnings)
        if result == "switch":
            return None
        if result == "rekey":
            _write_provider_fields(spec.name, {"api_key": _prompt_api_key(spec.name)})
            # 使用新密钥重新测试同一模型（选择器默认选中它）。
            current = chosen
            user_model_flag = None
            continue
        if result == "repick":
            current = chosen
            user_model_flag = None
            continue
        return chosen  # 成功并继续


# ---------------------------------------------------------------------------
# 步骤 1——多提供商入口（已有配置分支：完成/添加/编辑）
# ---------------------------------------------------------------------------


def _manage_existing_providers(*, non_interactive: bool) -> None:
    """Edit/remove submenu for already-configured providers (interactive only)."""
    questionary = _require_questionary()
    from codeflow.cli._styles import CODEFLOW_STYLE

    while True:
        configured = _configured_providers()
        if not configured:
            return
        choices = [questionary.Choice(_provider_label(n), value=n) for n in configured]
        choices.append(questionary.Choice(_t("Back", "返回"), value=_BACK))
        target = questionary.select(
            _t("Pick a provider to manage:", "选择要管理的服务商:"),
            choices=choices,
            style=CODEFLOW_STYLE,
            qmark=_QMARK,
        ).ask()
        if target is None or target is _BACK:
            return

        action = questionary.select(
            _t(
                f"What would you like to do with {_provider_label(target)}?",
                f"对 {_provider_label(target)} 想做什么?",
            ),
            choices=[
                questionary.Choice(_t("Update API key", "更新 API Key"), value="update"),
                questionary.Choice(
                    _t("Remove (clear this provider's key)", "移除(清除该服务商的 Key)"),
                    value="remove",
                ),
                questionary.Choice(_t("Back", "返回"), value=_BACK),
            ],
            style=CODEFLOW_STYLE,
            qmark=_QMARK,
        ).ask()
        if action is None or action is _BACK:
            continue
        if action == "update":
            _write_provider_fields(target, {"api_key": _prompt_api_key(target)})
            console.print(
                _t(
                    f"  [green]✓ Updated {_provider_label(target)}.[/green]",
                    f"  [green]✓ 已更新 {_provider_label(target)}。[/green]",
                )
            )
        elif action == "remove":
            current = _load_current_default_model()
            from codeflow.providers.registry import find_by_name

            spec = find_by_name(target)
            was_default_source = bool(current and spec and _model_routes_to_provider(current, spec))
            if was_default_source:
                confirm = questionary.confirm(
                    _t(
                        f"The current default model comes from {_provider_label(target)}; "
                        "removing it means you'll need to pick a new default. Remove anyway?",
                        f"当前默认模型来自 {_provider_label(target)};移除后需要重新选择默认模型。仍要移除吗?",
                    ),
                    default=False,
                    style=CODEFLOW_STYLE,
                    qmark=_QMARK,
                ).ask()
                if not confirm:
                    continue
            _write_provider_fields(target, {"api_key": ""})
            if was_default_source:
                # 清除失效的默认值，让步骤 1 的防护强制重新选择，避免留下提供商无密钥的模型。
                from codeflow.config.update import set_default_model

                set_default_model("")
            console.print(
                _t(
                    f"  [green]✓ Removed {_provider_label(target)}'s configuration.[/green]",
                    f"  [green]✓ 已移除 {_provider_label(target)} 的配置。[/green]",
                )
            )


def _step1_provider(
    *,
    provider: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    non_interactive: bool,
    warnings: list[str],
    skip_test: bool = False,
) -> object:
    """Step 1 screen. Returns ``_BACK`` only when the user backs out of the
    first-run picker on the welcome screen (handled by the runner)."""
    _step_header(1, _t("Choose your LLM provider", "选择 LLM 服务商"))
    console.print(
        _t(
            "  [dim]CodeFlow's chat and reasoning are all driven by it.[/dim]",
            "  [dim]CodeFlow 的对话与思考都由它驱动。[/dim]",
        )
    )

    configured = _configured_providers()
    if non_interactive or not configured:
        result = _configure_one_provider(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            non_interactive=non_interactive,
            warnings=warnings,
            skip_test=skip_test,
        )
        if result is None:
            return _BACK
        return None

    questionary = _require_questionary()
    from codeflow.cli._styles import CODEFLOW_STYLE

    while True:
        names = ", ".join(_provider_label(n).split(" (")[0] for n in _configured_providers())
        action = questionary.select(
            _t(
                f"LLM provider already configured: {names}. What would you like to do?",
                f"LLM 服务商已配置:{names}。想做什么?",
            ),
            choices=[
                questionary.Choice(_t("Done, continue", "完成,继续"), value="done"),
                questionary.Choice(_t("Add another provider", "新增一个服务商"), value="add"),
                questionary.Choice(_t("Edit / remove a provider", "编辑 / 移除服务商"), value="edit"),
            ],
            style=CODEFLOW_STYLE,
            qmark=_QMARK,
        ).ask()
        if action is None:
            raise typer.Exit(1)  # Ctrl+C 表示退出，绝不能视为“完成”
        if action == "done":
            # 步骤 1 必填：至少有一个提供商且已设默认模型才能前进，避免删除所有提供商后漏过校验。
            if not (_configured_providers() and _load_current_default_model()):
                console.print(
                    _t(
                        "  [yellow]At least one provider with a default model is required — add or re-pick one.[/yellow]",
                        "  [yellow]至少需要一个带默认模型的服务商 — 请新增或重新选择一个。[/yellow]",
                    )
                )
                continue
            return None
        if action == "add":
            _configure_one_provider(
                provider=None,
                api_key=None,
                base_url=None,
                model=None,
                non_interactive=False,
                warnings=warnings,
                skip_test=skip_test,
            )
        elif action == "edit":
            _manage_existing_providers(non_interactive=non_interactive)


# ---------------------------------------------------------------------------
# 步骤 3——沙箱/运行位置
# ---------------------------------------------------------------------------


def _current_sandbox_backend() -> str:
    """Read ``tools.sandbox.backend`` from disk; defaults to ``none``."""
    data = _load_raw_config()
    return ((data.get("tools") or {}).get("sandbox") or {}).get("backend") or "none"


def _persist_sandbox_backend(backend: str) -> None:
    """Patch ``sandbox.backend`` on the on-disk config via the ops layer."""
    from codeflow.config.update import set_sandbox_backend

    set_sandbox_backend(backend)


def _probe_boxlite() -> tuple[bool, str]:
    """Probe boxlite availability. Returns ``(ok, reason)``.

    ``reason`` ∈ ``"ok"`` / ``"missing"`` / ``"error"``. The runtime import is
    the same availability gate ``build_executor`` uses for the boxlite backend.
    """
    console.print(_t("  [dim]⏳ Checking sandbox availability…[/dim]", "  [dim]⏳ 正在检测沙箱可用性…[/dim]"))
    try:
        import boxlite  # noqa: F401
    except ImportError:
        return False, "missing"
    except Exception:
        return False, "error"
    return True, "ok"


def _step3_sandbox(*, skip: bool, non_interactive: bool) -> object:
    """Step 3 — choose run location (host / boxlite sandbox)."""
    _step_header(3, _t("Choose where CodeFlow runs code / commands", "选择 CodeFlow 运行代码 / 命令的位置"))

    if skip or non_interactive:
        console.print(
            _t(
                "  [dim]Keeping run location: host (direct).[/dim]",
                "  [dim]保持运行位置:本机直接运行。[/dim]",
            )
        )
        return None

    questionary = _require_questionary()
    from codeflow.cli._styles import CODEFLOW_STYLE

    current = _current_sandbox_backend()
    choices: list[Any] = []
    if current != "none":
        choices.append(
            questionary.Choice(
                _t(f"Keep current: sandbox ({current})", f"沿用当前:沙箱({current})"), value="keep"
            )
        )
    choices.extend(
        [
            questionary.Choice(
                _t(
                    "Host (direct) — simplest, runs right on your machine",
                    "本机直接运行 — 最简单,直接在你的电脑上执行",
                ),
                value="none",
            ),
            questionary.Choice(
                _t(
                    "Sandbox isolation (boxlite) — isolated in a lightweight VM, safer (needs platform support)",
                    "沙箱隔离(boxlite)— 用轻量虚拟机隔离,更安全,需环境支持",
                ),
                value="boxlite",
            ),
            questionary.Choice(
                _t(
                    "SRT shell sandbox — OS-level isolation for commands (needs Node.js and platform setup)",
                    "SRT Shell 沙箱 — 按操作系统隔离命令(需要 Node.js 和平台初始化)",
                ),
                value="srt",
            ),
            questionary.Choice(_t("Back", "返回"), value=_BACK),
        ]
    )

    picked = questionary.select(_t("Run location:", "运行位置:"), choices=choices, style=CODEFLOW_STYLE, qmark=_QMARK).ask()
    if picked is None:
        raise typer.Exit(1)
    if picked is _BACK:
        return _BACK
    if picked == "keep":
        return None
    if picked == "none":
        _persist_sandbox_backend("none")
        console.print(
            _t(
                "  [green]✓ Running directly on the host.[/green]",
                "  [green]✓ 将在本机直接运行。[/green]",
            )
        )
        return None

    if picked == "srt":
        _persist_sandbox_backend("srt")
        console.print(
            _t(
                "  [green]✓ SRT will sandbox Shell commands. Startup fails if the SRT CLI or OS sandbox is unavailable.[/green]",
                "  [green]✓ Shell 命令将通过 SRT 沙箱执行。若 SRT CLI 或操作系统隔离不可用，Agent 会拒绝启动。[/green]",
            )
        )
        return None

    # boxlite——提交配置前先探测。
    while True:
        ok, reason = _probe_boxlite()
        if ok:
            _persist_sandbox_backend("boxlite")
            console.print(
                _t(
                    "  [green]✓ Sandbox available. Using default resources "
                    "(2 CPU / 2 GB / network); tune in the config file if needed.[/green]",
                    "  [green]✓ 沙箱可用。将使用默认资源(2 CPU / 2 GB / 联网);如需调整可改配置文件。[/green]",
                )
            )
            return None
        if reason == "missing":
            console.print(
                _t(
                    "  [yellow]✗ Sandbox runtime (boxlite) isn't installed.[/yellow]\n"
                    "  [dim]Install it, then choose “Retry after install”. Source checkout: "
                    "uv sync --extra sandbox. Tool install: "
                    "uv tool install --force 'codeflow-harness\\[channels,sandbox]'[/dim]",
                    "  [yellow]✗ 未安装沙箱运行时(boxlite)。[/yellow]\n"
                    "  [dim]先安装,再选「安装后重试」。源码 checkout: "
                    "uv sync --extra sandbox。工具安装: "
                    "uv tool install --force 'codeflow-harness\\[channels,sandbox]'[/dim]",
                )
            )
        else:  # reason 为 error：可导入但初始化失败
            console.print(
                _t(
                    "  [yellow]✗ Sandbox runtime (boxlite) is installed but failed to "
                    "start.[/yellow]\n"
                    "  [dim]Your machine may lack the required virtualization support. "
                    "Fall back to host, or check the boxlite setup docs.[/dim]",
                    "  [yellow]✗ 沙箱运行时(boxlite)已安装,但启动失败。[/yellow]\n"
                    "  [dim]可能本机缺少所需的虚拟化支持。可退回本机运行,或查阅 boxlite 安装文档。[/dim]",
                )
            )
        choice = _failure_choice(
            [
                (_t("Fall back to host", "退回本机运行"), "host"),
                (_t("Retry after install", "安装后重试"), "retry"),
                (_t("Skip", "跳过"), "skip"),
            ],
            non_interactive=non_interactive,
        )
        if choice == "retry":
            continue
        if choice == "host":
            _persist_sandbox_backend("none")
            console.print(
                _t(
                    "  [green]✓ Running directly on the host.[/green]",
                    "  [green]✓ 将在本机直接运行。[/green]",
                )
            )
        return None


# ---------------------------------------------------------------------------
# 步骤 4——聊天渠道（可叠加）
# ---------------------------------------------------------------------------


def _enabled_channels() -> list[str]:
    """Names of channels currently enabled on disk."""
    data = _load_raw_config()
    channels = data.get("channels") or {}
    return [name for name, c in channels.items() if isinstance(c, dict) and c.get("enabled")]


# CodeFlow 保留的国内渠道按精选顺序排列。
_CHANNEL_ORDER = (
    "feishu",
    "qq",
    "wecom",
)


# 各渠道凭据的获取位置——在字段提示前以淡色显示，让用户知道从哪里获取令牌或密钥。
_CHANNEL_CRED_HELP: dict[str, tuple[str, str]] = {
    "feishu": (
        "Feishu / Lark Open Platform → your app → Credentials for App ID & App Secret.",
        "飞书开放平台 → 你的应用 → 凭证与基础信息 拿 App ID / App Secret。",
    ),
    "wecom": (
        "WeCom admin console → your bot / app for its ID and secret.",
        "企业微信管理后台 → 机器人 / 应用 拿 ID 和 secret。",
    ),
    "qq": (
        "QQ Open Platform → your bot for App ID & secret.",
        "QQ 开放平台 → 你的机器人 拿 App ID 和 secret。",
    ),
}


def _ordered_channel_names() -> list[str]:
    from codeflow.channels.registry import discover_channel_names

    rank = {name: i for i, name in enumerate(_CHANNEL_ORDER)}
    return sorted(discover_channel_names(), key=lambda n: (rank.get(n, len(rank)), n))


def _select_channel() -> Optional[str]:
    """List available channels via the registry and let the user pick one."""
    questionary = _require_questionary()
    from codeflow.cli._styles import CODEFLOW_STYLE

    names = _ordered_channel_names()
    choices = [questionary.Choice(n, value=n) for n in names]
    choices.append(questionary.Choice(_t("Back", "返回"), value=_BACK))
    picked = questionary.select(_t("Channel:", "渠道:"), choices=choices, style=CODEFLOW_STYLE, qmark=_QMARK).ask()
    return picked


def _prompt_channel_fields(channel: str) -> Any:
    """Reflect a channel's Pydantic schema and prompt for credential-like fields."""
    questionary = _require_questionary()
    from codeflow.cli._styles import CODEFLOW_STYLE
    from codeflow.config.update_channels import channel_field_specs

    try:
        specs = channel_field_specs(channel)
    except KeyError as exc:
        console.print(f"  [red]✗[/red] {exc}")
        raise typer.Exit(1)

    # 预先扫描将要询问的凭据字段，以便提前告知用户正在配置什么，并处理零字段场景。
    promptable = [
        (path, spec)
        for path, spec in specs.items()
        if path != "enabled" and spec.get("type", "") == "str" and spec.get("default") in ("", None)
    ]
    if promptable:
        names = ", ".join(path for path, _ in promptable)
        console.print(
            _t(
                f"  [dim]Configuring {channel} — fill in:[/dim] {names}",
                f"  [dim]正在配置 {channel} — 请填写:[/dim] {names}",
            )
        )
        help_text = _CHANNEL_CRED_HELP.get(channel)
        if help_text:
            console.print(
                _t(
                    f"  [dim]Where to get it: {help_text[0]}[/dim]",
                    f"  [dim]去哪拿:{help_text[1]}[/dim]",
                )
            )
    else:
        console.print(
            _t(
                f"  [dim]{channel} needs no credentials; enabling.[/dim]",
                f"  [dim]{channel} 无需填写凭证,正在启用。[/dim]",
            )
        )

    fields: dict[str, Any] = {}
    for idx, (path, spec) in enumerate(promptable):
        required = bool(spec.get("required"))
        description = spec.get("description", "")
        opt_tag = "" if required else _t(" (optional)", " (可选)")
        prompt_label = f"{path}{opt_tag}" + (f" — {description}" if description else "") + ":"
        # 第一个字段提交空值会退回渠道选择器；后续可选字段提交空值会跳过；后续必填字段
        # 则重新提示（过去会静默接受空值，导致启用仅配置一半的渠道——写入层只把
        # "required" 当作 UX 标记）。
        allow_back = idx == 0
        placeholder = _field_placeholder(allow_back, required)
        while True:
            if spec.get("is_secret"):
                value = questionary.password(
                    prompt_label, placeholder=placeholder, style=CODEFLOW_STYLE, qmark=_QMARK
                ).ask()
            else:
                value = questionary.text(prompt_label, placeholder=placeholder, style=CODEFLOW_STYLE, qmark=_QMARK).ask()
            if value is None:
                raise typer.Exit(1)
            value = value.strip()
            if value:
                fields[path] = value
                break
            if allow_back:
                return _BACK  # 第一个字段为空则返回渠道选择器
            if required:
                console.print(_t(f"  [yellow]{path} is required.[/yellow]", f"  [yellow]{path} 为必填项。[/yellow]"))
                continue  # 重新提示，避免启用缺少凭据的渠道
            break  # 可选字段提交空值时跳过
    return fields


def _enable_channel(channel: str, fields: dict[str, Any]) -> None:
    """Thin wrapper for ``enable_channel`` that surfaces ops errors with hints."""
    from pydantic import ValidationError

    from codeflow.config.update_channels import enable_channel

    try:
        enable_channel(channel, fields)
    except KeyError as exc:
        console.print(f"  [red]✗[/red] {exc}")
        raise typer.Exit(1)
    except ValidationError as exc:
        console.print(_t(f"  [red]✗ Validation failed:[/red]\n{exc}", f"  [red]✗ 校验失败:[/red]\n{exc}"))
        raise typer.Exit(1)


def _channel_uses_interactive_login(channel: str) -> bool:
    """True for scancode/QR channels that pair via a live
    login flow rather than reflected credential fields."""
    try:
        from codeflow.channels.registry import discover_specs

        spec = discover_specs().get(channel)
        return bool(spec and spec.capabilities.interactive_login)
    except Exception:
        return False


def _scancode_login(channel: str, *, non_interactive: bool = False) -> None:
    """Run a scancode channel's real QR login (reuses ``channel.login``).

    Mirrors ``codeflow channels login``: enable the channel so its config section
    persists, build the adapter via its spec factory, then drive
    ``await channel.login()``, which displays the QR and waits. A failed or
    timed-out login drops into a numbered submenu (retry / skip).
    """
    import asyncio

    from codeflow.channels.registry import discover_specs
    from codeflow.config.update_channels import disable_channel

    # 先启用，使登录尝试期间工厂能读取对应配置段。任何未完成登录的路径都会回滚为禁用，
    # 因此取消或跳过扫码绝不会显示为“已连接”。
    _enable_channel(channel, {})

    specs = discover_specs()
    spec = specs.get(channel)
    if spec is None:
        disable_channel(channel)
        console.print(_t(f"  [red]✗ Unknown channel: {channel}[/red]", f"  [red]✗ 未知渠道:{channel}[/red]"))
        return

    # 上方已启用，便于工厂在登录期间读取配置段。任何未完成登录的路径都必须撤销启用，
    # 包括子菜单中按 Ctrl+C（抛出 typer.Exit）或扫码中断（KeyboardInterrupt）；两者都不是
    # ``Exception`` 子类，因此需包住整个流程，并在未真正登录时于 ``finally`` 中禁用。
    logged_in = False
    try:
        while True:
            from codeflow.config.loader import load_config

            channel_cfg = getattr(load_config().channels, channel, None)
            if channel_cfg is None:
                console.print(
                    _t(
                        f"  [red]✗ No config section for channel: {channel}[/red]",
                        f"  [red]✗ 渠道 {channel} 没有配置段。[/red]",
                    )
                )
                return
            adapter = spec.factory(channel_cfg)
            console.print(
                _t(
                    f"  [dim]Starting {spec.display_name} QR login…[/dim]",
                    f"  [dim]正在启动 {spec.display_name} 扫码登录…[/dim]",
                )
            )
            console.print(
                _t(
                    f"  [dim]A login link / QR code will appear below — scan it with "
                    f"{spec.display_name} (or open the link on a phone signed in to "
                    f"{spec.display_name}) to connect. This waits until you finish.[/dim]",
                    f"  [dim]下方会出现登录链接 / 二维码 — 用 {spec.display_name} 扫码"
                    f"(或在已登录 {spec.display_name} 的手机上打开该链接)即可接入;"
                    f"这里会一直等到你完成。[/dim]",
                )
            )
            from loguru import logger as _wiz_logger

            # 向导会静音 CodeFlow 日志以保持 UI 整洁，但扫码登录通过 loguru 输出二维码、链接、进度和
            # 失败原因。登录尝试期间只重新启用当前渠道的适配器子树（而非整个 CodeFlow，以免输出
            # 无关噪声），随后恢复静音。
            _login_log_scope = f"codeflow.channels.adapters.{channel}"
            try:
                _wiz_logger.enable(_login_log_scope)
                ok = asyncio.run(adapter.login(force=True))
            except Exception as exc:
                console.print(
                    _t(
                        f"  [yellow]✗ Login failed: {exc}[/yellow]",
                        f"  [yellow]✗ 登录失败:{exc}[/yellow]",
                    )
                )
                ok = False
            finally:
                _wiz_logger.disable(_login_log_scope)
            if ok:
                console.print(
                    _t(
                        f"  [green]✓ Logged in; {channel} connected.[/green]",
                        f"  [green]✓ 已登录;{channel} 已接入。[/green]",
                    )
                )
                logged_in = True
                return
            choice = _failure_choice(
                [
                    (_t("Retry", "重试"), "retry"),
                    (_t("Skip this channel", "跳过此渠道"), "skip"),
                ],
                non_interactive=non_interactive,
            )
            if choice == "retry":
                continue
            console.print(
                _t(
                    f"  [dim]{channel} not connected — finish later with codeflow channels login {channel}.[/dim]",
                    f"  [dim]{channel} 未接入 — 之后用 codeflow channels login {channel} 完成。[/dim]",
                )
            )
            return
    finally:
        if not logged_in:
            # 任何未登录的退出（跳过、无配置、子菜单 Ctrl+C、扫码中断）都会撤销启用，确保取消的
            # 扫码不会持久化为“已连接”。配置段仍保留，供 `codeflow channels login` 使用。
            disable_channel(channel)


def _channel_maturity(channel: str) -> str:
    """Evidence level declared by the channel's spec (``ChannelSpec.maturity``)."""
    try:
        from codeflow.channels.registry import discover_specs

        spec = discover_specs().get(channel)
    except Exception:
        return "unknown"
    return spec.maturity if spec else "unknown"


def _print_maturity_note(channel: str) -> None:
    """State a Beta channel's evidence level before credentials are entered."""
    if _channel_maturity(channel) != "beta":
        return
    console.print(
        _t(
            f"  [dim]{channel} is Beta: deterministic contract and security checks only, "
            f"no live send/receive evidence yet.[/dim]",
            f"  [dim]{channel} 处于 Beta:仅通过确定性契约与安全检查,尚无真实收发证据。[/dim]",
        )
    )


def _add_one_channel(*, non_interactive: bool = False) -> None:
    """Pick + (scancode login | reflect-prompt) + enable one channel."""
    while True:
        channel = _select_channel()
        if channel is None or channel is _BACK:
            return
        _print_maturity_note(channel)
        if _channel_uses_interactive_login(channel):
            _scancode_login(channel, non_interactive=non_interactive)
            return
        fields = _prompt_channel_fields(channel)
        if fields is _BACK:
            continue  # 从第一个字段返回，重新选择渠道
        _enable_channel(channel, fields)
        console.print(_t(f"  [green]✓ {channel} enabled.[/green]", f"  [green]✓ {channel} 已启用。[/green]"))
        return


def _manage_existing_channels() -> None:
    """Edit/disable submenu for already-enabled channels."""
    questionary = _require_questionary()
    from codeflow.cli._styles import CODEFLOW_STYLE
    from codeflow.config.update_channels import disable_channel, set_channel_fields

    while True:
        enabled = _enabled_channels()
        if not enabled:
            return
        choices = [questionary.Choice(n, value=n) for n in enabled]
        choices.append(questionary.Choice(_t("Back", "返回"), value=_BACK))
        target = questionary.select(
            _t("Pick a channel to manage:", "选择要管理的渠道:"),
            choices=choices,
            style=CODEFLOW_STYLE,
            qmark=_QMARK,
        ).ask()
        if target is None or target is _BACK:
            return
        action = questionary.select(
            _t(f"What would you like to do with {target}?", f"对 {target} 想做什么?"),
            choices=[
                questionary.Choice(_t("Edit config (re-enter fields)", "编辑配置(重填字段)"), value="edit"),
                questionary.Choice(_t("Disable (keep credentials)", "停用(保留凭证)"), value="disable"),
                questionary.Choice(_t("Back", "返回"), value=_BACK),
            ],
            style=CODEFLOW_STYLE,
            qmark=_QMARK,
        ).ask()
        if action is None or action is _BACK:
            continue
        if action == "edit":
            fields = _prompt_channel_fields(target)
            if fields is _BACK:
                continue  # 返回管理菜单
            if fields:
                set_channel_fields(target, fields)
            console.print(
                _t(
                    f"  [green]✓ {target} config updated.[/green]",
                    f"  [green]✓ {target} 配置已更新。[/green]",
                )
            )
        elif action == "disable":
            disable_channel(target)
            console.print(
                _t(
                    f"  [green]✓ Disabled {target} (credentials kept; re-enable later "
                    f"with codeflow channels enable {target}).[/green]",
                    f"  [green]✓ 已停用 {target}(凭证保留;之后用 codeflow channels enable {target} 重新启用)。[/green]",
                )
            )


def _step4_channel(*, channel: Optional[str], skip: bool, non_interactive: bool) -> object:
    """Step 4 — optionally enable chat channel(s)."""
    _step_header(
        4,
        _t(
            "(Optional) Connect a messaging app so you can chat with CodeFlow there",
            "(可选)接入即时通讯软件,直接在里面和 CodeFlow 聊天",
        ),
    )

    if skip:
        console.print(
            _t(
                "  [dim]Skipped via --skip-channel.[/dim]",
                "  [dim]已通过 --skip-channel 跳过。[/dim]",
            )
        )
        return None

    if non_interactive:
        if channel:
            console.print(
                f"[red]--channel {channel} given but non-interactive mode can't "
                "prompt for credential fields.[/red]\n"
                f"Run [#fbe23f]codeflow channels enable {channel} --<field> <value> ...[/#fbe23f] "
                "after onboard finishes."
            )
            raise typer.Exit(2)
        console.print(
            _t(
                "  [dim]Skipped (non-interactive, --channel not given).[/dim]",
                "  [dim]已跳过(非交互且未提供 --channel)。[/dim]",
            )
        )
        return None

    questionary = _require_questionary()
    from codeflow.cli._styles import CODEFLOW_STYLE

    if channel:
        if _channel_uses_interactive_login(channel):
            _scancode_login(channel, non_interactive=non_interactive)
        else:
            fields = _prompt_channel_fields(channel)
            if fields is _BACK:
                console.print(_t("  [dim]Skipped.[/dim]", "  [dim]已跳过。[/dim]"))
                return None
            _enable_channel(channel, fields)
            console.print(
                _t(
                    f"  [green]✓ {channel} enabled.[/green]",
                    f"  [green]✓ {channel} 已启用。[/green]",
                )
            )
        return None

    while True:
        enabled = _enabled_channels()
        if not enabled:
            action = questionary.select(
                _t("Connect a chat channel?", "接入一个聊天渠道吗?"),
                choices=[
                    questionary.Choice(_t("Add a channel", "新增一个渠道"), value="add"),
                    questionary.Choice(
                        _t(
                            "Skip (add later with codeflow channels enable)",
                            "跳过(之后用 codeflow channels enable 添加)",
                        ),
                        value="skip",
                    ),
                ],
                style=CODEFLOW_STYLE,
                qmark=_QMARK,
            ).ask()
            if action is None:
                raise typer.Exit(1)
            if action == "skip":
                console.print(_t("  [dim]Skipped.[/dim]", "  [dim]已跳过。[/dim]"))
                return None
            _add_one_channel(non_interactive=non_interactive)
            continue

        action = questionary.select(
            _t(
                f"Chat channel already connected: {', '.join(enabled)}. What would you like to do?",
                f"聊天渠道已接入:{', '.join(enabled)}。想做什么?",
            ),
            choices=[
                questionary.Choice(_t("Done, next step", "完成,下一步"), value="done"),
                questionary.Choice(_t("Add a channel", "新增一个渠道"), value="add"),
                questionary.Choice(_t("Edit / remove a channel", "编辑 / 移除渠道"), value="edit"),
            ],
            style=CODEFLOW_STYLE,
            qmark=_QMARK,
        ).ask()
        if action is None:
            raise typer.Exit(1)
        if action == "done":
            return None
        if action == "add":
            _add_one_channel(non_interactive=non_interactive)
        elif action == "edit":
            _manage_existing_channels()


# ---------------------------------------------------------------------------
# 步骤 2——长期仓库记忆与首次运行时轮次
# ---------------------------------------------------------------------------


def _set_memory_backend(backend: Optional[str]) -> None:
    """Set ``memory.backend`` through the config operations layer."""
    from codeflow.config.update import set_memory_backend

    set_memory_backend(backend)


def _init_extension_block_defaults() -> None:
    """Seed the memory / plugins / skillForge extension defaults via the ops layer."""
    from codeflow.config.update import init_extension_block_defaults

    init_extension_block_defaults()


def _memory_enabled() -> bool:
    """Return whether the configured Runtime selects Myna Memory."""
    data = _load_raw_config()
    return (data.get("memory") or {}).get("backend") == "myna"


def _default_provider_name() -> str:
    from codeflow.config.loader import load_config

    return load_config().get_provider_name() or "auto"


def _disable_memory_after_setup_problem(message: str, *, non_interactive: bool) -> bool:
    console.print(_t(f"  [red]✗ Myna is not ready:[/red] {message}", f"  [red]✗ Myna 尚未就绪:[/red] {message}"))
    if non_interactive:
        console.print(
            _t(
                "  [dim]Install a compatible myna-memory distribution, or re-run with --skip-memory.[/dim]",
                "  [dim]请安装兼容的 myna-memory distribution,或使用 --skip-memory 重新运行。[/dim]",
            )
        )
        raise typer.Exit(2)

    questionary = _require_questionary()
    from codeflow.cli._styles import CODEFLOW_STYLE

    action = questionary.select(
        _t("Memory cannot start. What next?", "记忆无法启动,接下来怎么做?"),
        choices=[
            questionary.Choice(_t("Continue without Memory", "关闭记忆并继续"), value="disable"),
            questionary.Choice(_t("Exit and install/fix Myna", "退出并安装或修复 Myna"), value="exit"),
        ],
        style=CODEFLOW_STYLE,
        qmark=_QMARK,
    ).ask()
    if action != "disable":
        raise typer.Exit(2)
    _set_memory_backend(None)
    console.print(
        _t(
            "  [yellow]○ Memory disabled explicitly; Local Skills still work.[/yellow]",
            "  [yellow]○ 已明确关闭记忆;本地技能仍可使用。[/yellow]",
        )
    )
    return False


def _prepare_myna(*, non_interactive: bool, yes: bool) -> bool:
    """Validate the installed Myna plugin and apply its consent-bound setup."""
    _set_memory_backend("myna")
    from codeflow.cli._plugin_stack import inspect_memory_backend
    from codeflow.config.codeflow import load_codeflow_config

    status = inspect_memory_backend(load_codeflow_config())
    if status.state != "available":
        return _disable_memory_after_setup_problem(
            status.error or "plugin unavailable", non_interactive=non_interactive
        )

    try:
        from myna.integrations.codeflow import descriptor

        integration = descriptor()
        if getattr(integration, "protocol", None) != "myna.codeflow-app-integration.v1":
            raise RuntimeError("unsupported Myna setup protocol")
        preview = integration.preview_setup(Path.cwd())
    except Exception as exc:
        return _disable_memory_after_setup_problem(str(exc), non_interactive=non_interactive)

    if not isinstance(preview, dict):
        return _disable_memory_after_setup_problem("invalid setup preview", non_interactive=non_interactive)
    state = preview.get("state")
    if state == "ready":
        console.print(
            _t(
                f"  [green]✓ Myna ready[/green] [dim]({preview.get('repo_key', 'repository')})[/dim]",
                f"  [green]✓ Myna 已就绪[/green] [dim]({preview.get('repo_key', '当前仓库')})[/dim]",
            )
        )
        return True
    if state != "setup_required":
        return _disable_memory_after_setup_problem("invalid setup preview", non_interactive=non_interactive)

    raw_creates = preview.get("creates") or []
    creates = (
        raw_creates if isinstance(raw_creates, list) and all(isinstance(path, str) for path in raw_creates) else []
    )
    raw_retrieval = preview.get("retrieval") or {}
    if not isinstance(raw_retrieval, dict):
        return _disable_memory_after_setup_problem("invalid setup preview", non_interactive=non_interactive)
    retrieval = raw_retrieval.get("profile", "unknown")
    console.print(
        _t(
            "  [dim]Myna will bind only this Git repository. It will not import history or install hooks.[/dim]",
            "  [dim]Myna 只会绑定当前 Git 仓库,不会导入历史或安装 Hook。[/dim]",
        )
    )
    console.print(f"  [dim]Workspace:[/dim] {preview.get('workspace')}")
    console.print(f"  [dim]Retrieval:[/dim] {retrieval}")
    for path in creates:
        console.print(f"  [dim]Create:[/dim] {path}")

    approved = yes
    if not non_interactive and not yes:
        questionary = _require_questionary()
        from codeflow.cli._styles import CODEFLOW_STYLE

        approved = bool(
            questionary.confirm(
                _t("Initialize Myna for this repository?", "为当前仓库初始化 Myna?"),
                default=True,
                style=CODEFLOW_STYLE,
                qmark=_QMARK,
            ).ask()
        )
    if not approved:
        if non_interactive:
            console.print(
                _t(
                    "  [red]Myna setup needs --yes, or use --skip-memory.[/red]",
                    "  [red]Myna 初始化需要 --yes,或使用 --skip-memory。[/red]",
                )
            )
            raise typer.Exit(2)
        _set_memory_backend(None)
        console.print(_t("  [dim]Memory disabled.[/dim]", "  [dim]已关闭记忆。[/dim]"))
        return False

    token = preview.get("consent_token")
    if not isinstance(token, str) or not token:
        return _disable_memory_after_setup_problem("setup consent is missing", non_interactive=non_interactive)
    try:
        receipt = integration.apply_setup(token)
    except Exception as exc:
        return _disable_memory_after_setup_problem(str(exc), non_interactive=non_interactive)
    if not isinstance(receipt, dict) or receipt.get("state") != "initialized":
        return _disable_memory_after_setup_problem("initialization did not complete", non_interactive=non_interactive)
    console.print(
        _t(
            f"  [green]✓ Myna initialized[/green] [dim]({receipt.get('repo_key', 'repository')})[/dim]",
            f"  [green]✓ Myna 初始化完成[/green] [dim]({receipt.get('repo_key', '当前仓库')})[/dim]",
        )
    )
    return True


def _step2_memory(
    *,
    skip: bool,
    non_interactive: bool,
    main_model: Optional[str],
    warnings: list[str],
    skip_test: bool = False,
    yes: bool = False,
) -> object:
    """Prepare Memory, then prove the configured Runtime with one real Turn."""
    del main_model
    _step_header(2, _t("Prepare Memory and meet your agent", "准备记忆并完成首次对话"))
    if skip:
        _set_memory_backend(None)
        console.print(
            _t(
                "  [dim]Memory disabled explicitly; Local Skills remain available.[/dim]",
                "  [dim]已明确关闭记忆；本地技能仍然可用。[/dim]",
            )
        )
    else:
        _prepare_myna(non_interactive=non_interactive, yes=yes)

    if skip_test:
        console.print(
            _t(
                "  [dim]First Turn skipped via --skip-test; run codeflow doctor --probe later.[/dim]",
                "  [dim]已通过 --skip-test 跳过首次 Turn;稍后可运行 codeflow doctor --probe。[/dim]",
            )
        )
        return None

    result = _run_test_probe(_default_provider_name(), non_interactive=non_interactive, warnings=warnings)
    if result in {"repick", "rekey", "switch"}:
        return _BACK
    return None


# ---------------------------------------------------------------------------
# 最终总结
# ---------------------------------------------------------------------------


def _print_next_steps(*, warnings: list[str]) -> None:
    from rich.table import Table

    console.print()
    if warnings:
        console.print(
            Panel(
                _t(
                    "[bold yellow]⚠ Setup finished with warnings[/bold yellow]",
                    "[bold yellow]⚠ 配置完成,但有警告[/bold yellow]",
                )
                + "\n\n"
                + _t(
                    "[dim]These items didn't pass a connectivity test:[/dim] ",
                    "[dim]以下项目未通过连通测试:[/dim] ",
                )
                + f"{', '.join(warnings)}\n"
                + _t(
                    "[dim]Fix them before relying on the related features "
                    "(re-run [/dim][#fbe23f]codeflow onboard[/#fbe23f][dim] to reconfigure).[/dim]",
                    "[dim]在依赖相关功能前请先修复(重新运行 [/dim][#fbe23f]codeflow onboard[/#fbe23f][dim] 重新配置)。[/dim]",
                ),
                border_style="yellow",
                padding=(1, 2),
            )
        )
    else:
        console.print(
            Panel(
                _t(
                    "[bold green]🎉 Setup complete![/bold green]",
                    "[bold green]🎉 配置完成![/bold green]",
                ),
                border_style="green",
                padding=(0, 2),
            )
        )

    # 从磁盘读取并回顾已完成的配置，让用户获得明确的收尾。
    provs = ", ".join(_provider_label(n).split(" (")[0] for n in _configured_providers()) or "—"
    run_loc = (
        _t("Host (direct)", "本机直接运行")
        if _current_sandbox_backend() == "none"
        else _t("Sandbox (boxlite)", "沙箱(boxlite)")
    )
    chans = ", ".join(_enabled_channels()) or _t("none", "无")
    mem = _t("Myna", "Myna") if _memory_enabled() else _t("disabled", "已关闭")
    recap = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    recap.add_column(style="dim", no_wrap=True)
    recap.add_column()
    recap.add_row(_t("Provider", "服务商"), provs)
    recap.add_row(_t("Default model", "默认模型"), _load_current_default_model() or "—")
    recap.add_row(_t("Run location", "运行位置"), run_loc)
    recap.add_row(_t("Channels", "聊天渠道"), chans)
    recap.add_row(_t("Memory", "长期记忆"), mem)
    console.print(
        Panel(
            recap,
            title=f"[bold]{_t('Your setup', '你的配置')}[/bold]",
            title_align="left",
            border_style="#8a6d00",
            padding=(1, 2),
        )
    )

    table = Table(show_header=False, box=None, padding=(0, 3, 0, 0))
    table.add_column(style="#fbe23f", no_wrap=True)
    table.add_column(style="dim")
    table.add_row("codeflow", _t("launch the native TUI (default)", "启动原生 TUI(默认)"))
    table.add_row("codeflow gateway", _t("run the gateway (serve channels)", "运行网关(对接渠道)"))
    table.add_row('codeflow run -m "hello, world"', _t("ask a one-shot question", "一次性提问"))
    table.add_row("codeflow channels list", _t("see connected chat channels", "查看已接入的渠道"))
    table.add_row("codeflow provider list", _t("check your provider config", "检查当前服务商配置"))
    console.print(
        Panel(
            table,
            title=f"[bold]{_t('Get started', '开始使用')}[/bold]",
            title_align="left",
            border_style="#c8a900",
            padding=(1, 2),
        )
    )


# ---------------------------------------------------------------------------
# 向导运行器（页面状态机）与可复用入口
# ---------------------------------------------------------------------------


def run_wizard(
    *,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    channel: Optional[str] = None,
    skip_sandbox: bool = False,
    skip_channel: bool = False,
    skip_memory: bool = False,
    non_interactive: bool = False,
    yes: bool = False,
    reset: bool = False,
    skip_test: bool = False,
) -> None:
    """Run the 4-step onboarding wizard end-to-end.

    The reusable entry point: the ``onboard`` CLI command and the startup gate
    both call this. Screens form a state machine so a ``0) Back`` choice can
    rewind one step; Ctrl+C exits keeping whatever was already written.

    Internal INFO logs (config writes, etc.) are hushed for the wizard's
    duration so they don't clutter the UI, then restored in ``finally`` —
    display-only; logging elsewhere is unaffected.
    """
    from loguru import logger as _logger

    _logger.disable("codeflow")
    try:
        _run_wizard_body(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            channel=channel,
            skip_sandbox=skip_sandbox,
            skip_channel=skip_channel,
            skip_memory=skip_memory,
            non_interactive=non_interactive,
            yes=yes,
            reset=reset,
            skip_test=skip_test,
        )
    finally:
        _logger.enable("codeflow")


def _run_wizard_body(
    *,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    channel: Optional[str] = None,
    skip_sandbox: bool = False,
    skip_channel: bool = False,
    skip_memory: bool = False,
    non_interactive: bool = False,
    yes: bool = False,
    reset: bool = False,
    skip_test: bool = False,
) -> None:
    global _LANG
    _check_tty_or_die(non_interactive)
    _LANG = _config_language()  # 从已保存语言开始（默认为 "en"）
    if not non_interactive:
        _pick_language()  # 可能改变 _LANG（在下方引导完成后持久化）
    _handle_existing_config(reset=reset, yes=yes, non_interactive=non_interactive)
    _bootstrap_empty_config()
    if not non_interactive:
        from codeflow.config.update import set_language

        set_language(_LANG)  # config.json 已存在，现在持久化语言

    console.print()
    console.print(
        Panel(
            _t(
                "[bold #fbe23f]✨ Welcome to the CodeFlow setup wizard[/bold #fbe23f]\n\n"
                "[dim]We'll configure, in order:[/dim]\n"
                "  [#fbe23f]①[/#fbe23f] LLM      [#fbe23f]②[/#fbe23f] Memory + first reply      "
                "[#fbe23f]③[/#fbe23f] Run location      [#fbe23f]④[/#fbe23f] Chat channel\n\n"
                "[dim]↑↓ select · Enter confirm · Ctrl+C quit anytime — anything already written is kept.[/dim]",
                "[bold #fbe23f]✨ 欢迎使用 CodeFlow 配置向导[/bold #fbe23f]\n\n"
                "[dim]我们将依次配置:[/dim]\n"
                "  [#fbe23f]①[/#fbe23f] LLM      [#fbe23f]②[/#fbe23f] 记忆 + 首次回复      "
                "[#fbe23f]③[/#fbe23f] 运行位置      [#fbe23f]④[/#fbe23f] 聊天渠道\n\n"
                "[dim]↑↓ 选择 · Enter 确认 · 随时 Ctrl+C 退出 — 已写入的配置会保留。[/dim]",
            ),
            border_style="#c8a900",
            padding=(1, 2),
        )
    )

    warnings: list[str] = []

    # 页面状态机。每个页面返回 ``_BACK`` 表示后退，返回其他值表示前进。步骤 1 必填；
    # 在首个页面从中后退不会产生效果，因为没有更早的页面。
    screens: list[Callable[[], object]] = [
        lambda: _step1_provider(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            non_interactive=non_interactive,
            warnings=warnings,
            skip_test=skip_test,
        ),
        lambda: _step2_memory(
            skip=skip_memory,
            non_interactive=non_interactive,
            main_model=_load_current_default_model(),
            warnings=warnings,
            skip_test=skip_test,
            yes=yes,
        ),
        lambda: _step3_sandbox(skip=skip_sandbox, non_interactive=non_interactive),
        lambda: _step4_channel(channel=channel, skip=skip_channel, non_interactive=non_interactive),
    ]

    index = 0
    while index < len(screens):
        result = screens[index]()
        if result is _BACK:
            if index == 0:
                # 语言选择器在状态机前运行，因此步骤 1 是首个带编号页面，却不是用户看到的
                # 第一个页面。从中后退会回到语言选择器：重新选择并持久化后，再以所选语言
                # 显示步骤 1。步骤 1 始终必填，绝不能跳过；否则提供商/模型不会写入，启动门
                # 会再次触发并陷入无限循环。
                _pick_language()
                from codeflow.config.update import set_language

                set_language(_LANG)
            else:
                index -= 1
        else:
            index += 1

    _print_next_steps(warnings=warnings)


# ---------------------------------------------------------------------------
# 启动门——由裸 `codeflow`、`codeflow run` 和 TUI 入口调用
# ---------------------------------------------------------------------------


def ensure_configured_or_onboard(*, non_interactive: bool = False) -> bool:
    """Run the wizard when the required config (provider + model) is missing.

    Returns ``True`` if config was already complete (caller proceeds straight
    to the session), ``False`` if the wizard ran (config is now populated). In
    a non-interactive context with missing config, the wizard's TTY check
    will raise — callers on non-TTY paths must guard before invoking.
    """
    if _is_config_populated():
        return True
    run_wizard(non_interactive=non_interactive)
    return False


# ---------------------------------------------------------------------------
# Typer 入口
# ---------------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    """Attach the ``onboard`` command to ``app``."""

    @app.command()
    def onboard(
        provider: Optional[str] = typer.Option(None, "--provider", help="LLM provider name (skips Step 1's prompt)"),
        api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for the chosen provider"),
        base_url: Optional[str] = typer.Option(None, "--base-url", help="Custom OpenAI-compatible base URL"),
        model: Optional[str] = typer.Option(None, "--model", help="Default model id (e.g. 'openai/gpt-4o-mini')"),
        channel: Optional[str] = typer.Option(None, "--channel", help="Channel to enable in Step 4"),
        skip_sandbox: bool = typer.Option(False, "--skip-sandbox", help="Skip Step 3 (run location)"),
        skip_channel: bool = typer.Option(False, "--skip-channel", help="Skip Step 4 (channel setup)"),
        skip_memory: bool = typer.Option(False, "--skip-memory", help="Disable Memory in Step 2"),
        non_interactive: bool = typer.Option(
            False,
            "--non-interactive",
            help="Run without prompts (requires flags for any missing field)",
        ),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip all confirm prompts"),
        reset: bool = typer.Option(
            False,
            "--reset",
            help="Re-run the wizard over an existing config (does not erase it; each step keeps current values as defaults)",
        ),
        skip_test: bool = typer.Option(
            False,
            "--skip-test",
            help="Skip the one-shot test message (avoids a billed call; connectivity is still checked)",
        ),
    ) -> None:
        """Four-step setup: provider → Memory + first Turn → sandbox → channel."""
        run_wizard(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            channel=channel,
            skip_sandbox=skip_sandbox,
            skip_channel=skip_channel,
            skip_memory=skip_memory,
            non_interactive=non_interactive,
            yes=yes,
            reset=reset,
            skip_test=skip_test,
        )


__all__ = ["register", "run_wizard", "ensure_configured_or_onboard"]
