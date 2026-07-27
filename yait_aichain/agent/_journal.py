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

import json
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


#: How much of an observation is kept inline on an entry.
OBSERVATION_CHARS = 240

#: How much of the attempted action is rendered into the prompt trail.
ACTION_CHARS = 320


def _normalise_intent(text: str) -> str:
    """Fold an intent to its content, so punctuation and case are not novelty."""
    return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " "
                            for ch in str(text).lower()).split())


def _action_signature(action: "dict | None") -> str:
    """Stable identity of an action, for spotting a call re-issued verbatim."""
    if not action:
        return ""
    try:
        return json.dumps(action, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(action)


def _action_excerpt(action: "dict | None") -> str:
    """
    One-line, bounded rendering of what an action actually did.

    For a tool call the arguments *are* the attempt, so a trail that omits them
    records that something was tried without recording what.
    """
    if not action:
        return ""
    kind = action.get("type")
    if kind == "tool":
        args = ", ".join(f"{k}={_excerpt(v, ACTION_CHARS // 2)}"
                         for k, v in (action.get("kwargs") or {}).items())
        return _excerpt(f"{action.get('tool_name', '?')}({args})", ACTION_CHARS)
    if kind == "skill":
        return _excerpt(action.get("user_prompt", ""), ACTION_CHARS)
    return ""


def _excerpt(value, limit: int = OBSERVATION_CHARS) -> str:
    """One-line, bounded excerpt of a value — the journal stays small."""
    if value is None or value == "":
        return ""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


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
    observation : a short excerpt of what came back. An attempt record that
        says only what was *tried* is not enough to decide the next move — the
        loop must be able to read its own feedback. Truncated to
        ``OBSERVATION_CHARS``; the full value lives under *artifact*.
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
    observation: str        = ""
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
        observation: str = "",
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
            artifact=artifact, observation=_excerpt(observation),
            step=step, attempt=attempt, tokens=tokens,
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

        The verdict needs a **full window**: with fewer than *last_k* entries
        there is not yet enough evidence to call a run stuck, and reporting one
        would let a single early failure end the run before it had a chance to
        recover.
        """
        if len(self._entries) < last_k:
            return True                    # not enough evidence to judge
        return any(e.outcome in (DONE, REFUTED)
                   for e in self._entries[-last_k:])

    def is_repeating(self, last_k: int = 5) -> bool:
        """
        True when the last *last_k* attempts stopped carrying new information.

        :meth:`has_progress` catches a run that is *failing*; this catches one
        that is *succeeding pointlessly*. An agent can keep issuing calls that
        return cleanly — bisecting past the resolution of its own instrument,
        re-asking a question already answered — and every entry is an honest
        ``done``. Nothing in the outcomes distinguishes that from real work.

        Two signals, both requiring a full window and both deliberately strict,
        because a false positive would interrupt an agent that is genuinely
        working:

        * every attempt in the window states the same intent, or
        * every attempt in the window issues the identical action.

        This reports; it does not stop. The judgement of whether a repeated
        approach is still worth pursuing belongs to the caller.

        **What this deliberately does not do.** A fuzzier version — flagging
        intents that merely *resemble* each other — was measured against three
        real runs and rejected. An agent bisecting past its instrument's
        resolution scored a median word-overlap of 0.73 between consecutive
        intents; a textbook binary search that finished correctly scored 0.64.
        The two are not separable, because restating "probe the midpoint of
        X–Y" every turn is what a healthy search looks like. Any threshold in
        that gap is fitted to one example and would interrupt working runs, so
        only exact repetition is reported.
        """
        if len(self._entries) < last_k:
            return False                   # not enough evidence to judge
        tail = self._entries[-last_k:]
        intents = {_normalise_intent(e.intent) for e in tail}
        if len(intents) == 1 and intents != {""}:
            return True
        actions = {_action_signature(e.action) for e in tail}
        return len(actions) == 1 and actions != {""}

    # ── prompt view ──────────────────────────────────────────────────
    def progress_summary(self, limit: int = 12) -> str:
        """
        Render the observation trail for the next action prompt.

        Every recent attempt appears, whatever its outcome, with **what was
        tried** and **what came back**. Both halves are needed. A summary of
        intents alone is useless to a loop that has to decide its next move from
        feedback; and an observation without the candidate that produced it is
        just as bad — "4 problems remain" cannot be acted on when the draft that
        had 4 problems is nowhere to be seen. Failures earn their place here
        too: a failed probe still returned information.

        Both halves are bounded, so a genuinely large artifact will still be
        clipped. That is what ``artifact`` is for: put it in memory, which the
        action prompt renders separately and does not roll off after
        *limit* entries.

        Successes are additionally marked by evidence kind, so the model can see
        which of its "achievements" nothing ever verified. What has been ruled
        out is rendered separately by :meth:`do_not_redo`.
        """
        items = self._entries[-limit:]
        if not items:
            return ""
        lines = []
        for e in items:
            if e.outcome == DONE:
                kind = (e.evidence or {}).get("kind", MODEL_CLAIM)
                mark = "verified" if kind == CHECK else "claimed"
            else:
                mark = e.outcome
            stored = f" → memory['{e.artifact}']" if e.artifact else ""
            lines.append(f"- [{mark}] {e.intent or '(no intent)'}{stored}")
            tried = _action_excerpt(e.action)
            if tried:
                lines.append(f"    tried:  {tried}")
            if e.observation:
                lines.append(f"    result: {e.observation}")
        return "\n".join(lines)

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
