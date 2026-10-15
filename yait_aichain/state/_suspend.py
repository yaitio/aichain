"""
state._suspend — Suspend and SuspendedResult, the pause primitive.

A tool raises :class:`Suspend` to park the run until a signal arrives; the
engine catches it at any depth, persists the document and returns a
:class:`SuspendedResult`. See docs/pipelines/pause-and-resume.md.
"""

from __future__ import annotations

from .._picklable import Picklable

from dataclasses import dataclass, field

from .. import _hints


class Suspend(Picklable, Exception):
    """
    Raised by a suspend tool to pause the current run.

    Parameters
    ----------
    reason : str
        Human-readable reason for the pause (shown to whoever must act).
    resume_with : dict | None
        Schema/description of the signal expected on resume, e.g.
        ``{"approved": "bool"}``. Informational — the engine does not enforce
        it; the tool validates the signal on resume.
    hint : dict | None
        Optional hint for an external scheduler/trigger about how/when to
        resume (e.g. ``{"wake_at": "<ISO time>"}`` or
        ``{"on": "payment.confirmed"}``). The engine only parks the run and
        records the hint; it never schedules anything itself.
    """

    def __init__(self, reason: str = "", resume_with: dict | None = None,
                 *, hint: dict | None = None) -> None:
        super().__init__(reason)
        self.reason      = reason
        self.resume_with = resume_with or {}
        self.hint        = hint


@dataclass
class SuspendedResult:
    """
    Returned by ``run()`` / ``resume()`` when the run paused instead of
    finishing. Falsy, so ``if not result:`` distinguishes it from a completed
    run, mirroring ``AgentResult``.

    Attributes
    ----------
    run_id : str
        Identifier to resume with: ``chain.resume(run_id, signal=...)``.
    awaiting : dict
        ``{"reason": ..., "resume_with": ...}`` describing what is awaited.
    document : dict
        The full self-contained run document (also saved in the store). Exposed
        so callers can persist/transport it themselves if they prefer.
    """

    run_id:   str
    awaiting: dict  = field(default_factory=dict)
    document: dict  = field(default_factory=dict)

    #: The fields every result has, so code written against `AgentResult` or
    #: `ChainResult` reads a pause without a special case: not a success, and
    #: nothing failed — ``suspended`` says which of those falsy results this
    #: is. ``output`` is left out on purpose: reading it says to resume, where
    #: None would read as an answer of nothing.
    success   = False
    suspended = True
    usage     = None
    cost      = None
    tokens_used = 0

    @property
    def error(self) -> str:
        """Not an error — what the run waits for, for code that prints `.error`."""
        return f"waiting: {self.awaiting.get('reason') or 'a signal'}"

    def __getattr__(self, name):
        raise _hints.missing(self, name)

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return (f"SuspendedResult(run_id={self.run_id!r}, "
                f"awaiting={self.awaiting.get('reason', '')!r})")
