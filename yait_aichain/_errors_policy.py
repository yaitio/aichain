"""
_errors_policy — one vocabulary for "what happens when a step fails".

Four words, one meaning each, on both `Chain` (``on_step_error``) and `Pool`
(``on_error``): ``raise`` propagates, ``stop`` ends and keeps what completed,
``skip`` carries on and warns, ``collect`` carries on silently and records the
error. See docs/pipelines/chain.md and docs/pipelines/pool.md.
"""

from __future__ import annotations

import traceback

RAISE   = "raise"
STOP    = "stop"
SKIP    = "skip"
COLLECT = "collect"

POLICIES = frozenset({RAISE, STOP, SKIP, COLLECT})

def check(value: str, *, parameter: str) -> str:
    """Validate a policy, naming the whole vocabulary when it is wrong."""
    if value not in POLICIES:
        raise ValueError(
            f"{parameter} must be one of {sorted(POLICIES)}; got {value!r}. "
            "raise = propagate; stop = end and keep what completed; "
            "skip = carry on and warn; collect = carry on silently.")
    return value


def describe(exc: BaseException) -> dict:
    """What a failure was, in a form a reader can act on.

    `str(exc)` was what both primitives stored, and it is the least useful
    part: `KeyError('tenant')` renders as `'tenant'`, which in a run record
    reads as a value rather than a fault, and a `TimeoutError` with an empty
    message renders as nothing at all. The type and the traceback are what
    tell a reader whether to retry, fix a caller, or call the provider.
    """
    return {
        "type":      type(exc).__name__,
        "message":   str(exc),
        "traceback": "".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__)),
    }
