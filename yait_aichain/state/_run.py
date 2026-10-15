"""
state._run — one invocation's own data, apart from what ran it.

An agent is configuration: a model, tools, a policy, a store. A run is what
one invocation of it accumulates — an id, the document it is parked under, the
approval it is waiting on, whether somebody is watching it stream, the hooks it
reports to. The second used to live on the first, and one agent serves
concurrent runs (a module-level agent in a warm Lambda, a Pool's items, a
`delegate` worker), so approvals crossed between runs, one tenant's stream
carried another's events, and a run read another's id. `Agent._fork()` copied
the agent per run to keep them apart. This holds the run instead, and the
agent stays what it was configured as.

The current run is ambient, in a `ContextVar`: a run opened inside another —
a worker inside its lead — is the current one until it ends, and the lead's
is current again after. A generator advanced in a context of its own
(`_bound.bound`) keeps its run across steps.

The working counters a `stop_when` condition reads (`state`) stay a plain
dict: it is a public seam — a driver of `Agent.step()` keeps it between
invocations as JSON, and a caller's `check(fn)` receives it.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Run:
    """What one invocation owns. ``owner`` is the object that runs it."""

    owner: Any = None
    #: The id every event of this run carries; None outside a run.
    id: str | None = None
    #: The run that started this one — a worker's lead.
    parent_id: str | None = None
    #: The call of that run this one runs inside — the `delegate` call, the
    #: tool that started a pipeline — so a reader can put the two in one tree.
    parent_call_id: str | None = None
    depth: int = 0
    max_depth: int = 5
    #: The hooks this run reports to: the owner's, plus anything the run adds
    #: for itself (a stream's collector), which must not outlive it.
    hooks: list = field(default_factory=list)
    #: The approval request between going out and being answered.
    pending: Any = None
    #: How the last tool call failed, described, for the journal.
    failure: dict | None = None


@dataclass
class AgentRun(Run):
    #: The store document while this run is parked, so a second pause
    #: overwrites the first instead of leaking a second run id.
    doc_id: str | None = None
    #: Whether a consumer walks this run; set by `stream()`, never by `run()`.
    streaming: bool = False
    #: The working counters, once the loop has them.
    state: dict | None = None
    #: The run's journal, once the loop has it — where an interceptor's
    #: change to a call is written down.
    journal: Any = None

    #: The counters that survive a pause, and only those. `adaptations` are
    #: per-request notices about what went on the wire and do not describe
    #: the conversation, which is what resumes; `truncated` describes one
    #: reply. `unverified` travels: dropped, an unchecked change made before
    #: a pause was answered after it without the check `verified()` holds for.
    PORTABLE = ("steps", "tokens", "cost", "plan", "plan_calls", "unverified")

    @classmethod
    def portable(cls, state: dict) -> dict:
        """*state* as a parked document keeps it: JSON, and only what resumes."""
        out = {k: state.get(k) for k in cls.PORTABLE}
        # The nudge latch travels too, as a list: dropped, a resumed run said
        # its nudge again.
        out["nudged"] = sorted(state.get("nudged") or ())
        return out

    def parked(self, messages: list, state: dict, journal) -> dict:
        """What a parked document carries for this run — the one place that
        decides what survives a pause."""
        return {"messages": messages, "state": self.portable(state),
                "journal": journal.to_list(), "run_id": self.id}


_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("yait_aichain_run", default=None)


def active(owner) -> Run | None:
    """The current run if *owner* is the one running it, else None."""
    run = _ACTIVE.get()
    return run if run is not None and run.owner is owner else None


@contextlib.contextmanager
def opened(run: Run):
    """Make *run* the current one for the duration."""
    from .._events import _RUN_ID
    token, stamp = _ACTIVE.set(run), _RUN_ID.set(run)
    try:
        yield run
    finally:
        _RUN_ID.reset(stamp)
        _ACTIVE.reset(token)


__all__ = ["Run", "AgentRun", "active", "opened"]
