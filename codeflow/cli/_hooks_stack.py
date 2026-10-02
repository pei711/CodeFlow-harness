"""CLI assembly helper for AgentLoop's hook chain.

Composes a :class:`CompositeHook` from optional sub-stacks:

- Eval Engine hooks from :func:`build_eval_stack`.
- Future caller-supplied hooks.

The helper is intentionally thin — most callers just hand
``EvalEngine.hooks()`` to AgentLoop's ``hooks=...`` constructor
parameter. ``build_hooks_stack`` is for callers that want to assemble
a chain across multiple sub-engines before the AgentLoop is
constructed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable

from codeflow.agent.hook import AgentHook, CompositeHook

if TYPE_CHECKING:
    from codeflow.eval_engine import EvalEngine

logger = logging.getLogger(__name__)


def build_hooks_stack(
    *,
    eval_engine: "EvalEngine | None" = None,
    extra_hooks: Iterable[AgentHook] | None = None,
) -> CompositeHook:
    """Build a :class:`CompositeHook` from optional contributing engines.

    Order (matches the documented priority in agent/hook/__init__):
      1. Eval Engine's three hooks (before_iteration → tool_audit →
         after_iteration). All three are no-ops in default config.
      2. Caller-supplied ``extra_hooks``.

    """
    chain = CompositeHook()
    if eval_engine is not None:
        hooks = eval_engine.hooks()
        chain.extend(hooks)
        logger.debug("Hooks stack: added %d Eval Engine hooks", len(hooks))
    if extra_hooks is not None:
        for hook in extra_hooks:
            chain.append(hook)
            logger.debug("Hooks stack: appended %s", hook.name)
    return chain


__all__ = ["build_hooks_stack"]
