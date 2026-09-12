"""
_errors_policy — one vocabulary for "what happens when a step fails".

`Chain` took `on_step_error` in {raise, stop, skip}; `Pool` took `on_error` in
{raise, collect, skip}. Two names for one decision, three values each, and
only `raise` meant the same thing in both — a caller who learned one primitive
guessed wrong at the other, and `skip` was the trap: in a chain it meant
"carry on to the next step, loudly", in a pool "record it and say nothing"
was `collect` while `skip` added the warning.

Four words, one meaning each, on both:

    raise     stop everything and propagate. The default for Chain, because
              a pipeline that continues past a broken step produces a result
              nobody can trust.
    stop      end here and keep what completed. No exception, no further work.
    skip      carry on, and warn. What you want when a failure is expected
              but should still be visible.
    collect   carry on, silently, recording the error. The default for Pool,
              because a fan-out over a thousand items usually expects a few
              to fail and a thousand warnings help nobody.

Both primitives accept all four. What each does with `stop` differs in the
obvious way — a chain stops running steps, a pool stops starting items — and
that is the difference between a sequence and a fan-out, not between two
vocabularies.
"""

from __future__ import annotations

import traceback

RAISE   = "raise"
STOP    = "stop"
SKIP    = "skip"
COLLECT = "collect"

POLICIES = frozenset({RAISE, STOP, SKIP, COLLECT})

#: The old spellings, kept working. Renaming a parameter breaks code that has
#: no other way to say the same thing, and the library's own rule is that a
#: caller is told what changed rather than left to find out.
ALIASES: dict = {}


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
