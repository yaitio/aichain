"""
yait_aichain._events — observability events and lifecycle hooks.

The machine-readable side of the step boundary: an :class:`Event` at every LLM
call, tool call, step and run, sent to observe-only hooks, with the
:class:`Hook`, :class:`Tracer` and :class:`LoggingTracer` bases. Heavy sinks
(OpenTelemetry, files, databases) live in the application.
See docs/control/observability.md.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from collections.abc import Callable, Iterable

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
    run_id:   str | None  = None
    step:     int | None  = None
    name:     str | None  = None
    payload:  dict          = field(default_factory=dict)
    usage:    int | None  = None
    cost:     float | None = None
    duration: float | None = None
    error:    str | None  = None
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

    def __call__(self, event: Event) -> None:
        wanted = event.type.replace(".", "_")
        method = getattr(self, wanted, None)
        if callable(method):
            method(event)
            return


class Tracer(Hook):
    """A hook that records every event into ``.events`` for later inspection."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def __call__(self, event: Event) -> None:
        self.events.append(event)


class LoggingTracer(Hook):
    """A hook that logs every event to a ``logging`` logger (default INFO).

    Parameters
    ----------
    logger : logging.Logger | None, optional
        Where to log; ``yait_aichain.trace`` by default.
    level : int, optional
        The level each event is logged at; ``INFO`` by default.
    """

    def __init__(self, logger: logging.Logger | None = None,
                 level: int = logging.INFO) -> None:
        self._logger = logger or logging.getLogger("yait_aichain.trace")
        self._level  = level

    def __call__(self, event: Event) -> None:
        self._logger.log(self._level, "%r", event)


# ── Dispatch ───────────────────────────────────────────────────────────────────

import contextlib as _contextlib
import contextvars as _contextvars

#: The run in flight — the innermost Chain, Pool or Agent — so a Skill's model
#: calls inside it carry its id. Contextvars cross into a Pool's workers
#: because the pool copies the context into each.
_RUN_ID: "_contextvars.ContextVar[str | None]" = _contextvars.ContextVar(
    "yait_aichain_run_id", default=None)


@_contextlib.contextmanager
def running(run_id):
    """Make *run_id* — or a run object, whose ``id`` is read when an event is
    sent — the id events without one are stamped with."""
    token = _RUN_ID.set(run_id)
    try:
        yield
    finally:
        _RUN_ID.reset(token)


#: Hooks that hear every event sent inside a block, beside the sender's own:
#: a lead's, while a chain it delegated to runs.
_LISTENERS: "_contextvars.ContextVar[tuple]" = _contextvars.ContextVar(
    "yait_aichain_listeners", default=())


@_contextlib.contextmanager
def listening(hooks):
    """Let *hooks* hear every event sent inside the block."""
    token = _LISTENERS.set(_LISTENERS.get() + tuple(h for h in (hooks or ()) if h is not None))
    try:
        yield
    finally:
        _LISTENERS.reset(token)


def emit(hooks: Iterable[Callable[[Event], None]] | None, event: Event) -> None:
    """
    Dispatch *event* to every hook, swallowing hook errors.

    A buggy or slow hook must never crash the run, so each hook is called inside
    a try/except; failures are logged at DEBUG and otherwise ignored.
    """
    own = list(hooks or ())
    hooks = own + [h for h in _LISTENERS.get() if all(h is not o for o in own)]
    if not hooks:
        return
    ambient = _RUN_ID.get()
    ambient = getattr(ambient, "id", ambient)     # an agent's run fills its id late
    if event.run_id is None and ambient is not None:
        import dataclasses as _dataclasses
        event = _dataclasses.replace(event, run_id=ambient)
    for hook in hooks:
        try:
            hook(event)
        except Exception as exc:                     # observe-only: never propagate
            # Said once per hook, as a warning with the traceback: a DEBUG
            # line would let tracing stop with nobody knowing.
            key = id(getattr(hook, "__func__", hook))
            if key not in _FAILED:
                _FAILED.add(key)
                _log.warning("hook %r failed on %s and is skipped for this event; "
                             "later failures of it are not repeated", hook, event.type,
                             exc_info=exc)


#: Hooks whose failure was already reported.
_FAILED: set = set()


__all__ = ["Event", "Hook", "Tracer", "LoggingTracer", "emit"]
