"""Local slash-command handling for the interactive ``codeflow run`` REPL.

Commands here are parsed and executed in-process, BEFORE the message is
submitted as a turn to the spine, so they never reach the LLM. They reuse the existing
``codeflow cron`` CLI command functions directly
(the functions Typer decorates stay plain callables), so behaviour and
output match the shell CLI with no duplicated logic.

Scope:

- ``/cron``     — full job management: list / get / add / enable / disable /
                  delete, plus read-only ``config get``.
- ``/help``     — list the available slash commands.

What is exposed follows one rule: the REPL may write *operational state*
(cron jobs in ``jobs.json`` — add/enable/disable/delete) but NOT the global
``~/.codeflow/config.json``. Every config write is shell-only, including
``cron config set/reset``.

Two further REPL-specific constraints:

- The prompt_toolkit REPL owns the TTY, so a nested ``click`` confirm prompt
  would corrupt terminal state. Destructive cron ops (``delete`` / ``disable``)
  therefore require an inline ``-y`` instead of an interactive confirm.
- ``cron run`` calls ``asyncio.run`` and is shell-only.

Anything that is not a recognised ``/cron`` / ``/help`` command
returns ``False`` so the caller forwards it unchanged — that keeps control
commands like ``/stop`` and ``/restart`` working.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from rich.console import Console

_CRON_SHELL_ONLY = {
    "run": "test-fire mutates job state and uses asyncio.run, which can't "
    "run inside the REPL event loop. Use `codeflow cron run` in a shell.",
}


def handle_repl_slash(command: str, *, console: "Console") -> bool:
    """Execute a local slash command. Return True iff it was handled here
    (caller must then NOT forward the input to the LLM)."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False  # 引号不成对，不由本层处理
    if not tokens:
        return False

    head, args = tokens[0].lower(), tokens[1:]
    if head == "/cron":
        return _handle_cron(args, console)
    if head in ("/help", "/?"):
        _print_help(console)
        return True
    return False


# ── 小型参数辅助方法 ─────────────────────────────────────────────────


def _has_flag(tokens: list[str], *names: str) -> bool:
    return any(t in names for t in tokens)


def _opt(tokens: list[str], *names: str) -> str | None:
    """Value of ``--name value`` or ``--name=value``; None if absent."""
    for i, t in enumerate(tokens):
        if t in names and i + 1 < len(tokens):
            return tokens[i + 1]
        for nm in names:
            if t.startswith(nm + "="):
                return t.split("=", 1)[1]
    return None


def _first_positional(tokens: list[str]) -> str | None:
    return next((t for t in tokens if not t.startswith("-")), None)


def _invoke(console: "Console", fn: Callable[..., Any], **kwargs: Any) -> None:
    """Call a CLI command function, swallowing the control-flow exceptions
    Typer/Click would normally turn into a process exit. The command has
    already printed its own success/error output by the time these raise.

    Constraint: ``fn`` is a Typer-decorated function whose parameter defaults
    are ``OptionInfo`` sentinels, not real values. Callers MUST pass every
    parameter explicitly in ``kwargs`` — an omitted one binds the sentinel,
    not the intended default. The per-command tests guard against drift."""
    import click
    import typer

    try:
        fn(**kwargs)
    except (click.BadParameter, click.UsageError) as exc:
        console.print(f"[red]{exc.format_message()}[/red]")
    except (typer.Exit, click.exceptions.Exit, click.exceptions.Abort, SystemExit):
        pass


# ── /cron 命令 ─────────────────────────────────────────────────────────


def _handle_cron(args: list[str], console: "Console") -> bool:
    from codeflow.cli import cron_commands as cc

    if not args or args[0] in ("help", "-h", "--help"):
        _print_cron_help(console)
        return True

    sub, rest = args[0], args[1:]

    if sub in _CRON_SHELL_ONLY:
        console.print(f"[yellow]/cron {sub} is shell-only: {_CRON_SHELL_ONLY[sub]}[/yellow]")
        return True

    if sub == "list":
        _invoke(console, cc.cron_list, all_=_has_flag(rest, "--all", "-a"))
        return True

    if sub == "get":
        ident = _first_positional(rest)
        if not ident:
            console.print("[red]usage: /cron get <id>[/red]")
            return True
        _invoke(console, cc.cron_get, id_prefix=ident)
        return True

    if sub == "enable":
        ident = _first_positional(rest)
        if not ident:
            console.print("[red]usage: /cron enable <id>[/red]")
            return True
        _invoke(console, cc.cron_enable, id_prefix=ident)
        return True

    if sub in ("delete", "disable"):
        return _handle_cron_destructive(sub, rest, console)

    if sub == "add":
        return _handle_cron_add(rest, console)

    if sub == "config":
        if rest and rest[0] in ("set", "reset"):
            console.print(
                "[yellow]/cron config set|reset is shell-only (writes global "
                "config). Use `codeflow cron config ...`.[/yellow]"
            )
            return True
        _invoke(
            console,
            cc.cron_config_get,
            forward_channels=_has_flag(rest, "--forward-channels"),
            default_timezone=_has_flag(rest, "--default-timezone"),
        )
        return True

    console.print(f"[red]Unknown /cron subcommand: {sub!r}[/red]")
    _print_cron_help(console)
    return True


def _handle_cron_destructive(sub: str, rest: list[str], console: "Console") -> bool:
    from codeflow.cli import cron_commands as cc

    ident = _first_positional(rest)
    if not ident:
        console.print(f"[red]usage: /cron {sub} <id> -y[/red]")
        return True
    if not _has_flag(rest, "-y", "--yes"):
        # prompt_toolkit 下不做交互确认——展示将发生的变更，并要求重新运行时显式传入 -y。
        job = _resolve_quiet(ident)
        if job is not None:
            console.print(
                f"[yellow]Would {sub} job {job.id} "
                f"({cc._format_schedule(job.schedule)}). "
                f"Re-run with -y to confirm: /cron {sub} {ident} -y[/yellow]"
            )
        return True
    fn = cc.cron_delete if sub == "delete" else cc.cron_disable
    _invoke(console, fn, id_prefix=ident, yes=True)
    return True


def _handle_cron_add(rest: list[str], console: "Console") -> bool:
    from codeflow.cli import cron_commands as cc

    name = _opt(rest, "--name")
    message = _opt(rest, "--message")
    if not name or not message:
        console.print(
            "[red]usage: /cron add --name <name> --message <text> "
            "(--cron <expr> | --at <iso> | --every <dur>) "
            "[--tz <zone>] [--channel <ch>] [--to <id>][/red]"
        )
        return True
    # 调度校验（互斥选择、语法）委托给 cron_add；它会输出友好错误并在输入无效时抛出 typer.Exit。
    _invoke(
        console,
        cc.cron_add,
        name=name,
        message=message,
        cron=_opt(rest, "--cron"),
        at_iso=_opt(rest, "--at"),
        every=_opt(rest, "--every"),
        tz=_opt(rest, "--tz"),
        channel=_opt(rest, "--channel"),
        to=_opt(rest, "--to"),
        yes=True,
    )
    return True


def _resolve_quiet(ident: str):
    """Resolve a job id/prefix for a preview message; None on no/ambiguous
    match (the resolver has already printed why)."""
    import click
    import typer

    from codeflow.cli import cron_commands as cc

    try:
        return cc._resolve_id(cc._open_service(), ident)
    except (typer.Exit, click.exceptions.Exit, SystemExit):
        return None


# ── 帮助 ─────────────────────────────────────────────────────────────────


def _print_help(console: "Console") -> None:
    console.print(
        "[bold]Local slash commands[/bold] (run in-process, not sent to the agent):\n"
        "  [cyan]/cron[/cyan] …      manage scheduled jobs — type [cyan]/cron help[/cyan]\n"
        "  [cyan]/help[/cyan]        this message"
    )


def _print_cron_help(console: "Console") -> None:
    console.print(
        "[bold]/cron[/bold] — scheduled jobs (~/.codeflow/cron/jobs.json)\n"
        "  [cyan]/cron list[/cyan] [--all]            list jobs\n"
        "  [cyan]/cron get[/cyan] <id>                full detail of one job\n"
        "  [cyan]/cron add[/cyan] --name N --message M (--cron E | --at ISO | --every DUR)\n"
        "  [cyan]/cron enable[/cyan] <id>             re-enable a paused job\n"
        "  [cyan]/cron disable[/cyan] <id> -y         pause a job\n"
        "  [cyan]/cron delete[/cyan] <id> -y          remove a job\n"
        "  [cyan]/cron config[/cyan] [get]            show cron routing config\n"
        "[dim]delete/disable need -y (no interactive prompt in the REPL); "
        "run / config set|reset are shell-only.[/dim]"
    )


__all__ = ["handle_repl_slash"]
