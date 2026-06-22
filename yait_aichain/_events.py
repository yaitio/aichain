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
        ``"step.started"``, ``"run.suspended"``.
    run_id : str | None
        Identifier of the run this event belongs to.
    step : int | None
        Zero-based step index within the run, when applicable.
    name : str | None
        Tool or model name, when applicable.
    payload : dict
        Event-specific extra fields (tool kwargs, decision, reason, …).
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

    def __call__(self, event: "Event") -> None:
        method = getattr(self, event.type.replace(".", "_"), None)
        if callable(method):
            method(event)


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
