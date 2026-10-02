"""Regression guard: TUI must not let third-party named
loggers write to the terminal stderr inherited by Ink.

LiteLLM (and similar libs) install their own ``<StreamHandler <stderr>>`` on
named loggers (``LiteLLM`` / ``LiteLLM Router`` / ``LiteLLM Proxy``) at import
time. The InterceptHandler installed at the root logger by
``redirect_loguru_to_file`` does NOT catch records that the named-logger's own
handler emits — the handler fires before the propagation step. So per-chunk
``verbose_logger.debug("model_response.choices[0].delta: ...")`` (litellm
``streaming_handler.py``) writes to ``sys.stderr`` per token, which under the
TUI's inherited-TTY model overlays the Ink alt-screen render.

``redirect_loguru_to_file`` must therefore explicitly strip stderr/stdout
StreamHandlers from named loggers when it sets up TUI logging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from codeflow.cli._log_file import redirect_loguru_to_file  # noqa: E402

_KNOWN_TTY_LEAKING_LOGGERS = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")


def _tty_stream_handlers(logger_name: str) -> list[logging.Handler]:
    lg = logging.getLogger(logger_name)
    return [
        h
        for h in lg.handlers
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) in (sys.stderr, sys.stdout)
    ]


@pytest.fixture
def _isolate_logging(tmp_path, monkeypatch):
    """Snapshot + restore global logging state and reroute log file to tmp_path.

    Without this, ``redirect_loguru_to_file`` would: (a) leave the root
    InterceptHandler installed for subsequent tests, masking their logging;
    (b) write to the real ``~/.codeflow/logs/tui.log``;
    (c) leave LiteLLM loggers stripped for the rest of the session, breaking
    the precondition of later parametrized cases.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CODEFLOW_CLI_DEBUG", raising=False)

    monkeypatch.setattr("codeflow.config.loader._current_config_path", None)

    root = logging.getLogger()
    saved_root_handlers = list(root.handlers)
    saved_root_filters = list(root.filters)
    saved_root_level = root.level
    saved_root_disabled = root.disabled
    saved_named_handlers = {name: list(logging.getLogger(name).handlers) for name in _KNOWN_TTY_LEAKING_LOGGERS}
    saved_disable = logging.root.manager.disable
    for name in _KNOWN_TTY_LEAKING_LOGGERS:
        logging.getLogger(name).addHandler(logging.StreamHandler(sys.stderr))

    try:
        yield
    finally:
        root.handlers = saved_root_handlers
        root.filters = saved_root_filters
        root.level = saved_root_level
        root.disabled = saved_root_disabled
        for name, handlers in saved_named_handlers.items():
            logging.getLogger(name).handlers = handlers
        logging.disable(saved_disable)


@pytest.mark.parametrize("logger_name", _KNOWN_TTY_LEAKING_LOGGERS)
def test_third_party_tty_handlers_stripped_after_redirect(logger_name, _isolate_logging):

    assert _tty_stream_handlers(logger_name), (
        f"Test premise invalid: no stderr StreamHandler on {logger_name!r} at "
        "test start. Either litellm's logging setup changed upstream, or a "
        "prior test stripped handlers without restoration."
    )

    redirect_loguru_to_file("tui.log", retention=3)

    leaked = _tty_stream_handlers(logger_name)
    assert not leaked, (
        f"Logger {logger_name!r} still has a stderr/stdout StreamHandler after "
        f"redirect_loguru_to_file(): {leaked!r}. During TUI chat streaming, "
        "LiteLLM emits one DEBUG record per SSE chunk via this handler, which "
        "writes to the terminal inherited by Ink and overlays the alt-screen "
        "render."
    )


def test_redirect_returns_log_path_in_tmp_home(_isolate_logging):
    """Sanity: the function still returns the log file path it set up."""
    log_path = redirect_loguru_to_file("tui.log", retention=3)
    assert isinstance(log_path, Path)
    assert log_path.name == "tui.log"
    assert log_path.parent.name == "logs"


def test_propagation_to_root_intact_after_redirect(_isolate_logging):
    """LiteLLM records must still reach the file sink via root propagation.

    Stripping the named-logger's stderr handler must not silence the record —
    it must still propagate to root, be caught by the InterceptHandler, and
    land in the file sink. We verify the wiring (`propagate=True`) survives.

    This guards a FUTURE regression: a maintainer might "optimize" the fix by
    flipping ``propagate=False`` on these loggers to short-circuit root, which
    would silently drop the records from `~/.codeflow/logs/tui.log`. The fix
    as shipped does not modify propagate, so this test is green; if it ever
    goes red, the proposed change must be rejected.
    """
    redirect_loguru_to_file("tui.log", retention=3)
    for name in _KNOWN_TTY_LEAKING_LOGGERS:
        assert logging.getLogger(name).propagate is True, (
            f"Logger {name!r} no longer propagates to root after "
            "redirect_loguru_to_file — records would be silently dropped."
        )
