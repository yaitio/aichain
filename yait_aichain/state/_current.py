"""
state._current — which run's context is in flight, reachable without threading it through.

Ambient rather than a ``context=`` on every Skill, Pool and Tool, so the
environment does not leak into the scenario; a ``ContextVar`` rather than a
thread-local, so a Pool's workers inherit it through a copied context and a
reused thread does not. Nothing secret goes in here: the context is written
into the run document. See docs/production.md.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager

_CURRENT: contextvars.ContextVar = contextvars.ContextVar(
    "yait_aichain_run_context", default=None)


def current():
    """The `RunContext` of the run in flight, or None outside one."""
    return _CURRENT.get()


@contextmanager
def using(context):
    """Make *context* the current one for the duration of a block.

    ::

        with using(RunContext(tenant="acme")):
            chain.run()

    Parameters
    ----------
    context : RunContext | None
        The context every call inside the block sees. ``None`` changes
        nothing: the context already in force stays, so a primitive given no
        context of its own never takes away its caller's tenant.
    """
    if context is None:
        yield current()
        return
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        # Reset by token rather than to None: runs nest — an Agent inside a
        # Chain step — and clearing on the way out of the inner one would
        # leave the outer run without its own tenant for the rest of its life.
        _CURRENT.reset(token)
