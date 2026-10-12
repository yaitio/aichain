"""
``verbose=`` output, through the package logger rather than ``print``.

A library that prints writes into a Lambda's log, a notebook cell or a test's
output whether or not anybody asked; a logger can be routed, silenced or kept.
``verbose`` installs one plain handler on ``yait_aichain``, once, so the lines
still reach a terminal by default.
"""

from __future__ import annotations

import logging


def install_console_handler(verbose: int) -> None:
    """Route ``verbose`` output through the package logger, once."""
    pkg   = logging.getLogger("yait_aichain")
    level = logging.INFO if verbose == 1 else logging.DEBUG
    if not any(getattr(h, "_aichain_console", False) for h in pkg.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._aichain_console = True                # type: ignore[attr-defined]
        pkg.addHandler(handler)
    if pkg.level == logging.NOTSET or pkg.level > level:
        pkg.setLevel(level)
