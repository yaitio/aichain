"""
``verbose=`` output, through a logger of its own rather than ``print``.

The lines go to ``yait_aichain.console``, which has its own handler and does
not propagate, so ``verbose`` never changes the package logger or doubles a
caller's own handler. See docs/control/observability.md.
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
