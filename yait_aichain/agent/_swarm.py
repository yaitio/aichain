"""
agent._swarm — what a finished result rests on, and how a child speaks up.

Two mechanisms from the plan's *swarm coordination* row, and both are "a
field and a comparison, not an architecture" — which is the reason they were
worth waiting for `state/` and `Pool` to settle first.

**Acceptance that evidence can invalidate.** An agent closed its run on its
own word. `_journal.py` has always separated `CHECK` — a programmatic fact —
from `MODEL_CLAIM` — the model says so and nothing verified it — and nothing
read the field, so the distinction was decoration. A result now carries an
`Acceptance`: *what kind* of support it has, and a fingerprint of the checked
evidence. When the evidence changes, or the caller intervenes, the acceptance
is superseded and has to be re-earned; it is not silently still true.

**Beacons that interrupt a parent's wait.** `Pool` fanned out and had no
channel back: a child that hit a blocker could only fail, or finish and be
read afterwards. Coordination artifacts (`contract`, `decision`, `fact`) are
now kept apart from beacons that demand attention (`blocker`, `question`,
`contract_change`), and a `Pool` waiting on its items stops starting more the
moment one of those arrives — instead of polling, timing out, or spending the
rest of the fan-out on work the blocker already made pointless.

Both are *reported*, not enforced. An `unsupported` acceptance does not stop a
run; making the evidence kind **change** what the loop does is `nudge`, which
is the next row of the plan. Reporting first means the number exists before
anyone builds behaviour on it.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import threading
import time
import warnings
from dataclasses import dataclass, field, replace

# ── Beacons ──────────────────────────────────────────────────────────────────

#: Beacons that should make a waiting parent leave its wait.
ATTENTION = frozenset({"blocker", "question", "contract_change"})

#: Coordination artifacts — recorded and passed up, never interrupting.
COORDINATION = frozenset({"contract", "decision", "fact"})

KINDS = ATTENTION | COORDINATION


@dataclass(frozen=True)
class Beacon:
    """One signal from a child to whoever is waiting on it."""

    kind:    str
    message: str
    source:  str   = ""
    ts:      float = field(default_factory=time.time)

    @property
    def demands_attention(self) -> bool:
        return self.kind in ATTENTION

    def to_dict(self) -> dict:
        return {"kind": self.kind, "message": self.message,
                "source": self.source, "attention": self.demands_attention,
                "ts": self.ts}


class _Board:
    """Beacons posted during one run, forwarded to the run that contains it.

    Forwarding is what makes a grandchild's blocker reach the `Pool` two
    levels up: a delegated worker's board forwards to its parent agent's,
    which forwards to the pool item's, which the pool is watching. A board
    that kept its beacons to itself would be a record, not a channel.
    """

    def __init__(self, parent: "_Board | None" = None) -> None:
        self._items: list = []
        self._lock = threading.Lock()
        self._parent = parent

    def post(self, item: Beacon) -> None:
        with self._lock:
            self._items.append(item)
        if self._parent is not None:
            self._parent.post(item)

    def all(self) -> list:
        with self._lock:
            return list(self._items)

    def attention(self) -> list:
        return [b for b in self.all() if b.demands_attention]


_BOARD: contextvars.ContextVar = contextvars.ContextVar(
    "yait_aichain_beacon_board", default=None)


class collecting_beacons:
    """Open a board for the duration of a run, nested under any outer one."""

    def __enter__(self) -> _Board:
        self._board = _Board(parent=_BOARD.get())
        self._token = _BOARD.set(self._board)
        return self._board

    def __exit__(self, *exc) -> bool:
        try:
            _BOARD.reset(self._token)
        except ValueError:
            # A streaming run is a generator, and a generator may be resumed
            # in a different context from the one it started in. The board is
            # already detached from the run by then; failing the run over it
            # would turn bookkeeping into a crash.
            pass
        return False


def beacon(kind: str, message: str, *, source: str = "") -> Beacon:
    """Signal whoever is waiting on this run.

    Callable from anywhere inside a run — a tool, a nested agent, a pool item
    — without a parameter, because the board is ambient for the duration of
    the run, exactly as `RunContext` is.
    """
    if kind not in KINDS:
        raise ValueError(
            f"beacon kind must be one of {sorted(KINDS)}; got {kind!r}. "
            f"These interrupt a waiting parent: {sorted(ATTENTION)}.")
    item = Beacon(kind=kind, message=message, source=source)
    board = _BOARD.get()
    if board is None:
        # Not a raise — a tool may legitimately be run outside any agent —
        # but not silence either. A beacon nobody receives is the exact
        # failure this exists to prevent.
        warnings.warn(
            f"beacon({kind!r}) raised outside any run: nobody is listening "
            "and it was not delivered.", RuntimeWarning, stacklevel=2)
    else:
        board.post(item)
    return item


# ── Acceptance ───────────────────────────────────────────────────────────────

#: A result some programmatic fact supports.
CHECKED = "checked"
#: A result resting only on what a model — often a delegated worker — said.
CLAIMED = "claimed"
#: A result nothing supports at all: an answer with no action behind it.
UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class Acceptance:
    """What a finished result rests on, and whether that still holds."""

    kind:        str
    fingerprint: str
    checks:      tuple = ()
    claims:      tuple = ()
    superseded:  str   = ""

    @property
    def holds(self) -> bool:
        return not self.superseded

    def still_holds(self, journal) -> bool:
        """Recompute over fresh evidence and compare.

        *journal* is evidence obtained **again** — the same actions re-run,
        the same files re-read. If what the checks observed has changed, the
        result was accepted on facts that are no longer true.
        """
        if self.superseded:
            return False
        return acceptance_from(journal).fingerprint == self.fingerprint

    def supersede(self, reason: str) -> "Acceptance":
        """The caller intervened; the acceptance must be re-earned."""
        return replace(self, superseded=reason or "superseded")


def acceptance_from(journal) -> Acceptance:
    """Derive an `Acceptance` from a run's journal.

    Only completed attempts count, and only their evidence *kind* decides the
    category. The fingerprint covers the **checked** evidence alone: a claim
    changing does not change what was verified, and folding claims in would
    let a model re-wording its report look like the facts moving.
    """
    checks: list = []
    claims: list = []
    for entry in journal or []:
        record = entry if isinstance(entry, dict) else entry.to_dict()
        if record.get("outcome") != "done":
            continue
        kind = (record.get("evidence") or {}).get("kind")
        fact = {"action": record.get("action"),
                "observation": record.get("observation", "")}
        if kind == "check":
            checks.append(fact)
        elif kind == "model_claim":
            claims.append(fact)

    digest = hashlib.sha256(
        json.dumps(checks, sort_keys=True, default=str).encode()).hexdigest()
    kind = CHECKED if checks else CLAIMED if claims else UNSUPPORTED
    return Acceptance(kind=kind, fingerprint=digest,
                      checks=tuple(checks), claims=tuple(claims))
