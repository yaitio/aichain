"""
models._adaptation
==================

What the library changed about a request, said out loud.

Providers disagree, so a universal option cannot always travel unchanged:
``top_k`` has no field on the OpenAI wire, a reasoner refuses ``temperature``,
and asking DeepSeek to think routes to a different model altogether. Adapting
is right. Adapting **in silence** is not: an option the caller set and did not
get, with nothing said, is how a comparison between two models is quietly
invalidated — the caller believes both arms ran with the same settings.

Every option therefore meets one of five fates, and each of the last four is
reported:

    translated  same value, the provider's spelling (top_k → topK)
    adapted     same intent, a different shape (reasoning → thinking budget)
    declined    the provider has no such control; sent without it
    swapped     the MODEL changed, so price and answer both change
    refused     rejected before the wire, with the allowed values named

Two outlets, and the second is the one that matters. A warning is emitted once
per (model, option) so a loop does not drown in them — and warnings are lost
in a log. The record is what survives: ``Skill.last_adaptations``,
``AgentResult.adaptations``, so a measurement run can write down what its arms
actually sent rather than what they were asked to send.

Detection is deliberately not a list of rules. Sites that translate, adapt or
swap say so by calling :func:`record`; everything else is caught by comparing
the options the caller set against the request that came out. That way a new
conversion which forgets to record shows up as a wrong "declined" — visible
and reported, rather than silent. The failure mode of the mechanism is the
thing the mechanism exists to prevent.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field

TRANSLATED = "translated"
ADAPTED    = "adapted"
DECLINED   = "declined"
SWAPPED    = "swapped"
REFUSED    = "refused"


@dataclass(frozen=True)
class Adaptation:
    """
    One change the library made to a request, or refused to make.

    Attributes
    ----------
    kind    : one of the five above.
    option  : the universal option's name, as the caller wrote it.
    asked   : the value the caller set.
    sent    : what actually went out — a field name, a shape, a model name,
              or ``None`` when nothing did.
    why     : one sentence a reader can act on.
    model   : the model the request was built for.
    """

    kind:   str
    option: str
    asked:  object = None
    sent:   object = None
    why:    str = ""
    model:  str = ""

    def __str__(self) -> str:
        head = f"{self.model}: {self.option}={self.asked!r}"
        tail = f" → {self.sent!r}" if self.sent is not None else ""
        return f"{head}{tail} ({self.kind}) — {self.why}" if self.why else \
               f"{head}{tail} ({self.kind})"


# ── Collection ───────────────────────────────────────────────────────────────
#
# Request building is synchronous and nested (a Chain step builds inside a
# Skill inside an Agent), so the collector is a thread-local stack: each
# `collect()` gathers only what its own build produced.

_local = threading.local()


def _stack() -> list:
    if not hasattr(_local, "stack"):
        _local.stack = []
    return _local.stack


def record(adaptation: Adaptation) -> None:
    """Note one change. A no-op outside a :func:`collect` scope, so a builder
    called directly in a test does not need ceremony."""
    stack = _stack()
    if stack:
        stack[-1].append(adaptation)


class collect:
    """Context manager gathering the adaptations made inside it.

    ::

        with collect() as made:
            path, body = client.build_request(...)
        # made is a list of Adaptation
    """

    def __init__(self) -> None:
        self._made: list = []

    def __enter__(self) -> list:
        _stack().append(self._made)
        return self._made

    def __exit__(self, *exc) -> bool:
        _stack().pop()
        return False


# ── Reporting ────────────────────────────────────────────────────────────────

#: One warning per (model, option, kind). Enough to be seen once, few enough
#: that a thousand-step agent loop stays readable — and warning fatigue is the
#: way a channel like this stops being read at all.
_WARNED: set = set()


def announce(made: list) -> None:
    """Warn about the adaptations in *made*, once each per process."""
    for a in made:
        if a.kind == TRANSLATED:
            continue                    # a rename changes nothing for a caller
        mark = (a.model, a.option, a.kind)
        if mark in _WARNED:
            continue
        _WARNED.add(mark)
        warnings.warn(str(a), RuntimeWarning, stacklevel=4)


def reset_warnings() -> None:
    """Forget what has been warned about. For tests and for the parameter
    matrix, which asks "does this call say so?" and must not depend on whether
    an earlier call in the same process already did."""
    _WARNED.clear()


# ── Detecting what simply never arrived ──────────────────────────────────────

def _leaves(obj, path=()) -> dict:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_leaves(v, path + (str(k),)))
        return out
    # Tuples are walked too: a multipart body is a list of (name, value)
    # pairs, and treating a pair as one opaque leaf hid the value inside it —
    # three image models were told their `quality` had been declined while it
    # was travelling in the request.
    if isinstance(obj, (list, tuple)):
        out = {}
        for i, v in enumerate(obj):
            out.update(_leaves(v, path + (str(i),)))
        return out
    return {path: obj}


def note_absent(made: list, asked: dict, body, model_name: str,
                why: str = "", provider: str = "") -> None:
    """
    Append a ``declined`` for every option in *asked* that left no trace.

    An option counts as arrived when a site recorded it explicitly — a
    translation or a conversion — or when its value appears somewhere in the
    built body. Nothing else needs a rule: the options a provider never reads
    are precisely the ones that leave no mark, and they are the majority of
    what used to disappear quietly.
    """
    spoken = {a.option for a in made}
    # Compared with the type, not just the value: `True == 1 == 1.0` in
    # Python, so a plain `in` found `cache_control=True` inside
    # `temperature=1.0` and reported five providers as honouring an option
    # none of them implements.
    def _mark(v):
        return (type(v).__name__, v)

    # A multipart body carries every field as a string, so 0.4 arrives as
    # "0.4" and a type-exact match reports a delivered option as declined.
    # The looser comparison is confined to that shape: on a JSON body an
    # unrelated "1" must not stand in for an asked 1.
    multipart = isinstance(body, dict) and bool(body.get("_multipart"))

    values = set()
    for v in _leaves(body).values():
        try:
            values.add(_mark(v))
            if multipart and isinstance(v, str):
                values.add(("as-text", v))
        except TypeError:               # unhashable leaf; cannot have been ours
            pass

    for option, value in asked.items():
        if option in spoken or value is None:
            continue
        if isinstance(value, bool):
            # A flag cannot be traced by its value: every `True` in a body
            # looks alike, and `cache_control=True` was matching an unrelated
            # `enable_thinking=True`. A boolean counts as arrived only when
            # the site that consumed it said so.
            arrived = False
        else:
            try:
                arrived = _mark(value) in values or (
                    multipart and ("as-text", str(value)) in values)
            except TypeError:
                arrived = False
        if not arrived:
            from ._options import is_universal, why_absent
            reason = why or (why_absent(option, provider) if provider else
                             "this provider has no such control; the request "
                             "was sent without it")
            # A name the library does not know is a different thing from an
            # option this provider lacks, and the reader reacts differently:
            # one is a typo to fix, the other a capability to work around.
            kind = DECLINED if (not provider or is_universal(option)) else REFUSED
            made.append(Adaptation(
                kind=kind, option=option, asked=value, sent=None,
                model=model_name, why=reason))
