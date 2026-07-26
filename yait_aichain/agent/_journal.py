"""
agent._journal — the attempt journal
====================================

An append-only, typed record of **what the agent did** — as opposed to
``AgentMemory``, which holds the *data* the agent works with.

One log, three views:

* **progress**  — entries that finished, with the evidence that proves it;
* **refuted**   — approaches already established not to work, with the reason.
  These are fed back into the next action prompt as a "do not redo" block, so a
  long run stops re-attempting what it already ruled out;
* **recent**    — the tail, for the reflection prompt.

The journal is also what makes a goal-loop *stoppable*: ``has_progress(k)``
answers "did anything actually move in the last k entries", which is the only
way to detect an agent spinning in place while still inside its budget.

Entries are keyed by a monotonic ``seq`` — ``step``/``attempt`` are optional and
only meaningful for the plan-driven modes (``waterfall`` / ``agile``); a
goal-loop iteration has no step index.

Evidence is deliberately typed. Today the honest split is:

* ``check``       — a programmatic fact (a tool raised, a permission denied);
* ``model_claim`` — the orchestrator asserts it worked. Nothing verified it.

Success is normally a *claim*; failure is usually a *check*. Surfacing that
asymmetry is the point: a run whose "done" entries are all ``model_claim`` has
proven nothing, and the journal says so instead of hiding it.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

# ── Outcomes ───────────────────────────────────────────────────────────────────

DONE    = "done"       # finished; see ``evidence``
FAILED  = "failed"     # errored — may be retried
REFUTED = "refuted"    # established as not viable → never attempt again
SKIPPED = "skipped"    # not run (denied, gated, superseded)

OUTCOMES = frozenset({DONE, FAILED, REFUTED, SKIPPED})

# ── Evidence kinds ─────────────────────────────────────────────────────────────

CHECK       = "check"        # a programmatic fact
MODEL_CLAIM = "model_claim"  # the model says so; nothing verified it

EVIDENCE_KINDS = frozenset({CHECK, MODEL_CLAIM})


def evidence(kind: str, detail: str = "") -> dict:
    """Build an evidence record, validating *kind*."""
    if kind not in EVIDENCE_KINDS:
        raise ValueError(
            f"evidence kind must be one of {sorted(EVIDENCE_KINDS)}; got {kind!r}"
        )
    return {"kind": kind, "detail": detail}


# ── Entry ──────────────────────────────────────────────────────────────────────

@dataclass
class JournalEntry:
    """
    One recorded attempt.

    Attributes
    ----------
    seq : monotonic sequence number — the primary key.
    intent : what this attempt was trying to achieve (a plan step's goal in
        waterfall/agile, the iteration's intent in a goal loop).
    action : the action that was executed (tool call / skill call).
    outcome : one of ``done`` / ``failed`` / ``refuted`` / ``skipped``.
    evidence : ``{"kind": "check"|"model_claim", "detail": str}`` or ``None``.
    reason : why it failed or was refuted — the text fed back as "do not redo".
    artifact : reference (a memory key / path) to a bulky result; the result
        itself is NOT copied into the journal.
    step, attempt : plan coordinates; ``None`` in a goal loop.
    tokens : tokens spent on this attempt.
    ts : unix timestamp.
    """

    seq:      int
    intent:   str            = ""
    action:   dict           = field(default_factory=dict)
    outcome:  str            = DONE
    evidence: "dict | None"  = None
    reason:   str            = ""
    artifact: "str | None"   = None
    step:     "int | None"   = None
    attempt:  "int | None"   = None
    tokens:   int            = 0
    ts:       float          = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "JournalEntry":
        known = {f: d.get(f) for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


# ── Journal ────────────────────────────────────────────────────────────────────

class Journal:
    """Append-only log of attempts, with the three views the loop needs."""

    def __init__(self, entries: "list[JournalEntry] | None" = None) -> None:
        self._entries: list[JournalEntry] = list(entries or [])

    # ── write ────────────────────────────────────────────────────────
    def append(
        self,
        intent:   str = "",
        *,
        action:   "dict | None" = None,
        outcome:  str  = DONE,
        evidence: "dict | None" = None,
        reason:   str  = "",
        artifact: "str | None" = None,
        step:     "int | None" = None,
        attempt:  "int | None" = None,
        tokens:   int  = 0,
    ) -> JournalEntry:
        """Record one attempt and return the stored entry."""
        if outcome not in OUTCOMES:
            raise ValueError(
                f"outcome must be one of {sorted(OUTCOMES)}; got {outcome!r}"
            )
        entry = JournalEntry(
            seq=len(self._entries), intent=intent, action=action or {},
            outcome=outcome, evidence=evidence, reason=reason,
            artifact=artifact, step=step, attempt=attempt, tokens=tokens,
        )
        self._entries.append(entry)
        return entry

    # ── read ─────────────────────────────────────────────────────────
    @property
    def entries(self) -> list[JournalEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def done(self) -> list[JournalEntry]:
        """Entries that finished — the progress record."""
        return [e for e in self._entries if e.outcome == DONE]

    def refuted(self) -> list[JournalEntry]:
        """Approaches already ruled out — the "do not redo" set."""
        return [e for e in self._entries if e.outcome == REFUTED]

    def recent(self, n: int = 8) -> list[JournalEntry]:
        return self._entries[-n:]

    def has_progress(self, last_k: int = 5) -> bool:
        """
        True when anything actually moved in the last *last_k* entries.

        Movement means a ``done`` (work landed) or a ``refuted`` (the search
        space shrank). A tail of only ``failed`` / ``skipped`` means the agent is
        spinning — the stop rule a goal loop needs, since a budget alone would
        let it spin until the tokens run out.
        """
        tail = self._entries[-last_k:]
        if not tail:
            return True                    # nothing recorded yet → not stuck
        return any(e.outcome in (DONE, REFUTED) for e in tail)

    # ── prompt view ──────────────────────────────────────────────────
    def do_not_redo(self, limit: int = 10) -> str:
        """
        Render the refuted set for the action prompt. Empty string when nothing
        has been ruled out, so the block is simply omitted.
        """
        items = self.refuted()[-limit:]
        if not items:
            return ""
        lines = [f"- {e.intent or '(no intent)'}: {e.reason or 'ruled out'}"
                 for e in items]
        return "\n".join(lines)

    # ── serialisation ────────────────────────────────────────────────
    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self._entries]

    @classmethod
    def from_list(cls, data: "list[dict] | None") -> "Journal":
        return cls([JournalEntry.from_dict(d) for d in (data or [])])


__all__ = [
    "Journal", "JournalEntry", "evidence",
    "DONE", "FAILED", "REFUTED", "SKIPPED", "OUTCOMES",
    "CHECK", "MODEL_CLAIM", "EVIDENCE_KINDS",
]
