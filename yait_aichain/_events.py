"""
yait_aichain._events — observability events & lifecycle hooks
============================================================

The machine-readable side of the step boundary (1.4.4). While ``logging`` is the
human/ops diagnostic channel, **events** are a structured stream a program can
consume: route intermediate statuses into a chat UI, write traces to a database,
forward to OpenTelemetry, or assert on them in tests.

A *hook* is any callable ``hook(event) -> None`` registered on a ``Skill``,
``Chain`` or ``Agent`` (``hooks=[...]``). The engine emits an :class:`Event` at
every boundary (LLM call, tool call, step, run lifecycle). Hooks are
**observe-only** — a hook that raises is logged at DEBUG and never crashes the
run, and a hook cannot change the engine's behavior (approval/denial is the
permission layer's job, not a hook's).

Convenience bases:

* :class:`Hook`         — dispatches an event to a same-named method
  (``"tool_call.started"`` → ``tool_call_started(event)``), so you implement
  only the boundaries you care about.
* :class:`Tracer`       — records every event into ``.events``.
* :class:`LoggingTracer`— logs every event to a ``logging`` logger.

The library ships no heavy tracing dependency; OpenTelemetry/file/DB sinks live
in the application (or the product).
"""

from __future__ import annotations

import logging
import warnings
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

_log = logging.getLogger("yait_aichain.events")


# ── Event ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Event:
    """
    One operational event at a step boundary. Carries only operational fields —
    never hidden chain-of-thought.

    Attributes
    ----------
    type : str
        Dotted event name, e.g. ``"llm_call.started"``, ``"tool_call.ended"``,
        ``"tool_call.started"``, ``"step.started"`` (a **Chain** step;
        the agent's tool events were spelled this way before 2.7.0).
    run_id : str | None
        Identifier of the run this event belongs to.
    step : int | None
        Zero-based step index within the run, when applicable.
    name : str | None
        Tool or model name, when applicable.
    payload : dict
        Event-specific extra fields. For ``tool_call.*`` this carries what a
        program needs to **reconstruct** the turn rather than describe it:
        ``id`` (so concurrent calls in one turn can be told apart),
        ``arguments`` on started, and the tool's **raw** result on ended.

        The result travels by value, and that is a deliberate choice about
        what an ``Event`` is. A rendering of it cannot be recovered
        downstream — units, column metadata, the difference between *no rows*
        and *the tool declined* are all gone once it is prose — and a handle
        instead would need a lifetime and a place to live, which is hostile to
        the serverless niche this library targets: the process that issued the
        handle may be gone. Events are therefore no longer uniformly small;
        ``Event.__repr__`` omits the payload so a log stays readable.
    usage : int | None
        Token delta attributable to this event, when known.
    cost : float | None
        Estimated USD cost delta, when known.
    duration : float | None
        Wall-clock seconds the operation took (set on ``*.ended`` events).
    error : str | None
        Error message when the operation failed.
    ts : float
        Unix timestamp when the event was created (filled automatically).
    """

    type:     str
    run_id:   "str | None"  = None
    step:     "int | None"  = None
    name:     "str | None"  = None
    payload:  dict          = field(default_factory=dict)
    usage:    "int | None"  = None
    cost:     "float | None" = None
    duration: "float | None" = None
    error:    "str | None"  = None
    ts:       float         = field(default_factory=time.time)

    def __repr__(self) -> str:                       # compact, log-friendly
        bits = [f"type={self.type!r}"]
        if self.name is not None:     bits.append(f"name={self.name!r}")
        if self.step is not None:     bits.append(f"step={self.step}")
        if self.usage is not None:    bits.append(f"usage={self.usage}")
        if self.duration is not None: bits.append(f"dur={self.duration:.3f}s")
        if self.error is not None:    bits.append(f"error={self.error!r}")
        return "Event(" + ", ".join(bits) + ")"


# ── Hook bases ─────────────────────────────────────────────────────────────────

class Hook:
    """
    Optional convenience base for a hook.

    Subclass and implement any subset of boundary methods; an event of type
    ``"tool_call.started"`` is dispatched to ``tool_call_started(event)``. The
    raw callable form (a plain ``def hook(event): ...``) works too — ``Hook`` is
    only ergonomics.
    """

    #: The agent emitted a tool call as ``step.*`` while this module documented
    #: it as ``tool_call.*`` — and `Chain` emits `step.*` too, for a chain
    #: step, so one name meant two different things depending on which
    #: primitive a hook was attached to. Renamed 2026-09-11; a hook written
    #: against the old spelling keeps firing, once, with a warning.
    _RENAMED = {"tool_call_started": "step_started",
                "tool_call_ended":   "step_ended"}

    def __call__(self, event: "Event") -> None:
        wanted = event.type.replace(".", "_")
        method = getattr(self, wanted, None)
        if callable(method):
            method(event)
            return
        legacy = self._RENAMED.get(wanted)
        if legacy and callable(getattr(self, legacy, None)):
            warnings.warn(
                f"{type(self).__name__}.{legacy}() is the old name for "
                f"{wanted}(): the agent's tool events were called {legacy
                .replace('_', '.')} until 2.7.0, which collided with Chain's "
                "own step events. Rename the method.",
                DeprecationWarning, stacklevel=2)
            getattr(self, legacy)(event)


class Tracer(Hook):
    """A hook that records every event into ``.events`` for later inspection."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def __call__(self, event: "Event") -> None:
        self.events.append(event)


class LoggingTracer(Hook):
    """A hook that logs every event to a ``logging`` logger (default INFO)."""

    def __init__(self, logger: "logging.Logger | None" = None,
                 level: int = logging.INFO) -> None:
        self._logger = logger or logging.getLogger("yait_aichain.trace")
        self._level  = level

    def __call__(self, event: "Event") -> None:
        self._logger.log(self._level, "%r", event)


# ── Dispatch ───────────────────────────────────────────────────────────────────

def emit(hooks: "Iterable[Callable[[Event], None]] | None", event: "Event") -> None:
    """
    Dispatch *event* to every hook, swallowing hook errors.

    A buggy or slow hook must never crash the run, so each hook is called inside
    a try/except; failures are logged at DEBUG and otherwise ignored.
    """
    if not hooks:
        return
    for hook in hooks:
        try:
            hook(event)
        except Exception as exc:                     # observe-only: never propagate
            _log.debug("hook %r failed on %s: %s", hook, event.type, exc)


__all__ = ["Event", "Hook", "Tracer", "LoggingTracer", "emit"]
