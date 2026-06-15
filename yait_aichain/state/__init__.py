"""
state
=====

Durable, resumable run state for the orchestration primitives (Chain, then
Agent). The "Store + context" mechanism (M2):

* ``RunDocument``   — a self-contained, serialisable snapshot of one run
  (step definition + key-value ``variables`` + per-step status).
* ``StateStore``    — persistence of suspended runs (``InMemoryStore`` default,
  ``FileStore`` built-in, any KV/document store by subclassing).
* ``RunContext``    — non-secret per-request context (tenant, metadata).
* ``Suspend`` / ``SuspendedResult`` — pause a run until an external signal;
  raised by a suspend tool, caught by the engine.

All of this is opt-in: a run that never suspends behaves exactly as before.
"""

from ._context import RunContext
from ._suspend import Suspend, SuspendedResult
from ._store import StateStore, InMemoryStore, FileStore
from ._run_document import RunDocument, StepStatus

__all__ = [
    "RunContext",
    "Suspend",
    "SuspendedResult",
    "StateStore",
    "InMemoryStore",
    "FileStore",
    "RunDocument",
    "StepStatus",
]
