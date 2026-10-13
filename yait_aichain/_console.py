"""
``verbose=`` output, through a logger of its own rather than ``print``.

A library that prints writes into a Lambda's log, a notebook cell or a test's
output whether or not anybody asked; a logger can be routed, silenced or kept.
The lines go to ``yait_aichain.console``, which has its own handler and does
not propagate: one ``Agent(verbose=1)`` used to add a handler to the package
logger and lower its level for good, after which every agent in the process
printed its tool arguments, and lines appeared twice where the caller had a
handler of their own.
"""

from __future__ import annotations

import logging

_CONSOLE = logging.getLogger("yait_aichain.console")


def console() -> logging.Logger:
    """The logger ``verbose`` output goes to — a plain stream handler, once."""
    if not any(getattr(h, "_aichain_console", False) for h in _CONSOLE.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._aichain_console = True                # type: ignore[attr-defined]
        _CONSOLE.addHandler(handler)
        _CONSOLE.setLevel(logging.DEBUG)
        _CONSOLE.propagate = False
    return _CONSOLE
