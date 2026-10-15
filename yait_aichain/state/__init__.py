"""
state — durable, resumable run state for Chain and Agent.

``RunDocument`` is a serialisable snapshot of one run, ``StateStore`` keeps the
parked ones (``InMemoryStore``, ``FileStore``, or your own), ``RunContext`` the
non-secret per-request context, and ``Suspend`` / ``SuspendedResult`` the
pause. All opt-in: a run that never pauses needs none of it.
See docs/pipelines/pause-and-resume.md.
"""

from ._context import RunContext
from ._current import current, using
from ._suspend import Suspend, SuspendedResult
from ._store import StateStore, InMemoryStore, FileStore
from ._run_document import RunDocument, StepStatus
from ._secret import Secret

__all__ = [
    "RunContext", "current", "using",
    "Suspend",
    "SuspendedResult",
    "StateStore",
    "InMemoryStore",
    "FileStore",
    "RunDocument",
    "StepStatus",
    "Secret",
]
