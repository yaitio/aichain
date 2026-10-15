"""
agent._journal — the attempt journal: an append-only, typed record of what the
agent did, beside the conversation that holds what was said.

Each entry has an outcome and evidence — ``check`` for a fact code established,
``model_claim`` for the model's word. Refuted entries feed a "do not redo"
block, and ``has_progress(k)`` tells a run spinning in place inside its budget.
See docs/control/observability.md.
"""

from __future__ import annotations

import hashlib

import json
import time
from dataclasses import asdict, dataclass, field

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
    """Build an evidence record, validating *kind*.

    Parameters
    ----------
    kind : str
        ``check`` for a fact code verified, or ``model_claim`` for the
        model's word that nothing verified.
    detail : str, optional
        What the evidence is.
    """
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


def _action_signature(action: dict | None) -> str:
    """Stable identity of an action, for spotting a call re-issued verbatim."""
    if not action:
        return ""
    try:
        return json.dumps(action, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(action)


def _action_excerpt(action: dict | None) -> str:
    """One bounded line of what an action did: for a tool call the arguments are the
    attempt."""
    if not action:
        return ""
    if isinstance(action, str):
        # The agent loop stores the action already rendered to text (a JSON
        # string, via `_safe`).
        return _excerpt(action, ACTION_CHARS)
    kind = action.get("type")
    if kind == "tool":
        args = ", ".join(f"{k}={_excerpt(v, ACTION_CHARS // 2)}"
                         for k, v in (action.get("kwargs") or {}).items())
        return _excerpt(f"{action.get('tool_name', '?')}({args})", ACTION_CHARS)
    if kind == "skill":
        return _excerpt(action.get("user_prompt", ""), ACTION_CHARS)
    return ""


def _label(entry) -> str:
    """What an entry was: its intent, or, since the agent loop records none, what it
    did."""
    return entry.intent or _action_excerpt(entry.action) or "(no intent)"


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
    observation_sha : a hash of the whole observation, which is what an
        acceptance's fingerprint compares — not the excerpt.
    step, attempt : plan coordinates; ``None`` in a goal loop.
    tokens : tokens spent on this attempt.
    ts : unix timestamp.
    """

    seq:      int
    intent:   str            = ""
    action:   dict           = field(default_factory=dict)
    outcome:  str            = DONE
    evidence: dict | None  = None
    reason:   str            = ""
    artifact: str | None   = None
    observation: str        = ""
    #: A hash of the whole observation: the excerpt is cut at 240 characters,
    #: and a result that changed past the cut ("… 7 failed") must not "hold".
    observation_sha: str     = ""
    step:     int | None   = None
    attempt:  int | None   = None
    tokens:   int            = 0
    ts:       float          = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> JournalEntry:
        known = {f: d.get(f) for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


# ── Journal ────────────────────────────────────────────────────────────────────

class Journal:
    """Append-only log of attempts, with the three views the loop needs."""

    def __init__(self, entries: list[JournalEntry] | None = None) -> None:
        self._entries: list[JournalEntry] = list(entries or [])

    # ── write ────────────────────────────────────────────────────────
    def append(
        self,
        intent:   str = "",
        *,
        action:   dict | None = None,
        outcome:  str  = DONE,
        evidence: dict | None = None,
        reason:   str  = "",
        artifact: str | None = None,
        observation: str = "",
        step:     int | None = None,
        attempt:  int | None = None,
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
            observation_sha=(hashlib.sha256(str(observation).encode()).hexdigest()[:16]
                             if observation else ""),
            step=step, attempt=attempt, tokens=tokens,
        )
        self._entries.append(entry)
        return entry

    # ── read ─────────────────────────────────────────────────────────
    @property
    def entries(self) -> list[JournalEntry]:
        return list(self._entries)

    def __iter__(self):
        """The entries, in order — `still_holds(journal)` iterated a Journal
        and raised, since it was not iterable."""
        return iter(list(self._entries))

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

        It fires when every attempt in a full window states the same intent or issues
        the identical action — exact repetition only, so a working search is not
        interrupted. It reports and does not stop; :meth:`has_progress` catches failing.

        Parameters
        ----------
        last_k : int, optional
            The window, in entries; fewer entries than this is never repeating.
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

        Each recent attempt shows what was tried and what came back, failures
        included; successes are marked ``verified`` or ``claimed`` by evidence kind.
        Both halves are clipped — keep a large result in memory via ``artifact``.

        Parameters
        ----------
        limit : int, optional
            How many of the latest entries to render.
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
            lines.append(f"- [{mark}] {_label(e)}{stored}")
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
        lines = [f"- {_label(e)}: {e.reason or 'ruled out'}" for e in items]
        return "\n".join(lines)

    # ── serialisation ────────────────────────────────────────────────
    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self._entries]

    @classmethod
    def from_list(cls, data: list[dict] | None) -> Journal:
        return cls([JournalEntry.from_dict(d) for d in (data or [])])


__all__ = [
    "Journal", "JournalEntry", "evidence",
    "DONE", "FAILED", "REFUTED", "SKIPPED", "OUTCOMES",
    "CHECK", "MODEL_CLAIM", "EVIDENCE_KINDS",
]
