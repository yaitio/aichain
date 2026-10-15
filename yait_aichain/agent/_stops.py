"""
agent._stops — how a run ends: ceilings, checks, nudges, the verify gate.
"""

from __future__ import annotations

import logging

from .._dispatch import MAX_ARGUMENT_DEPTH  # noqa: F401  (named in the 4.0.1 notes)


_logger = logging.getLogger("yait_aichain.agent")


def step_count(n: int):
    """Stop after *n* decisions. A ceiling: reaching it is not success.

    Parameters
    ----------
    n : int
        Model turns allowed.
    """
    def _c(state):
        return "step_count" if state["steps"] >= n else None
    _c.kind, _c.name = "ceiling", "step_count"
    # `spec` is what lets a ceiling survive `Chain.save()`: the condition
    # itself is a closure and a file cannot hold one. A `check` has no
    # spec on purpose — its predicate is the caller's function, and
    # pretending it round-trips would be worse than saying it does not.
    _c.spec = {"kind": "step_count", "value": n}
    return _c


def token_budget(n: int):
    """Stop once *n* tokens have been spent. A ceiling.

    Parameters
    ----------
    n : int
        Tokens allowed, input and output together.
    """
    def _c(state):
        return "token_budget" if state["tokens"] >= n else None
    _c.kind, _c.name = "ceiling", "token_budget"
    _c.spec = {"kind": "token_budget", "value": n}
    return _c


def cost_budget(usd: float):
    """
    Stop once *usd* has been spent. A ceiling.

    Bounds "do not begin another step", not "never exceed by a cent": the
    length of a reply is not known before it is paid for.

    Parameters
    ----------
    usd : float
        Dollars allowed.
    """
    def _c(state):
        return "cost_budget" if (state["cost"] or 0.0) >= usd else None
    _c.kind, _c.name = "ceiling", "cost_budget"
    _c.spec = {"kind": "cost_budget", "value": usd}
    return _c


def nudge(predicate, message, name: str = "nudge"):
    """
    A condition that **speaks** instead of ending the run.

    `stop_when` knew two kinds, and both ended the run: a ceiling ("hit it,
    fail") and a check ("met it, succeed"). A stall needs neither. It needs
    "change approach", and the loop had no way to say that — so a stuck run
    either spun until a ceiling stopped it or was stopped early by a ceiling
    set too low to let a real run finish.

    When *predicate(state)* is true a user turn carrying *message* is
    appended and the loop carries on. It enters the conversation as an
    ordinary, logged message — not an invisible edit to the system prompt —
    so a reader of the transcript can see exactly what the model was told
    and when. *message* may be a string or a callable of `state`.

    It fires **once per streak**: while the predicate stays true it stays
    quiet, and it re-arms when the predicate clears. A reminder repeated
    every turn is how a reminder stops being read.

    Parameters
    ----------
    predicate : callable
        Given the run's state; true when the model should be told something.
    message : str | callable
        What it is told — a string, or a function of the state.
    name : str, optional
        How the nudge is named in events.
    """
    def _c(state):
        return f"nudge:{name}" if predicate(state) else None
    _c.kind, _c.name, _c.message = "nudge", name, message
    return _c


#: The risk classes that change something. A call of one of these is an
#: effect; any later call that succeeds — a read, a test run — is a look at
#: what it did.
_EFFECTS = frozenset({"write", "external", "financial", "destructive"})


def verified(message: str | None = None, name: str = "verified"):
    """
    Before the run answers, ask once for a check of what it changed.

    The commonest agent failure there is: "done" with nothing run. When the
    model answers after a call that changed something — a write, an external
    call, a delete — and no call has succeeded since to look at the result,
    its answer is held, *message* is put to it as a user turn, and the loop
    goes on. Once per run: a model that answers again is believed, and the
    record says it was asked.

    Only a call clears it — a read, a test run, anything that succeeds after
    the change. The model saying it checked does not, because that is the
    claim the gate exists to test.

    Parameters
    ----------
    message : str | None, optional
        What the model is told; a default names the change.
    name : str, optional
        How the nudge is reported in its event.
    """
    def _c(state):
        return state.get("unverified")
    _c.kind, _c.name = "verify", name
    _c.message = message or (lambda state: (
        f"You changed something with {state.get('unverified')!r} and have not looked at the "
        "result since. Check it — read it back, or run the check that shows it works — "
        "before you answer."))
    return _c


def stalled(last_k: int = 5):
    """Nudge when the last *last_k* attempts moved nothing.

    `Journal.has_progress` has existed, been documented, and been called
    from no library code at all — available only to someone writing their
    own loop. It reads the journal, not the task metric, so it has nothing to
    overfit to and applies to every task, verifiable or not.

    Parameters
    ----------
    last_k : int, optional
        Attempts that must all have moved nothing.
    """
    def _stalled(state):
        journal = state.get("journal")
        return journal is not None and not journal.has_progress(last_k)

    def _say(state):
        text = (f"The last {last_k} attempts did not move the task forward. "
                "Do not repeat them; change the approach, or conclude from "
                "what you already have.")
        ruled_out = state["journal"].do_not_redo()
        return f"{text}\n\nAlready ruled out:\n{ruled_out}" if ruled_out else text

    return nudge(_stalled, _say, name="stalled")


def repeating(last_k: int = 5):
    """Nudge when the last *last_k* attempts were the same move.

    The other half of a stall: `has_progress` catches a run that is failing,
    this catches one that is succeeding pointlessly — clean results that
    carry no new information. Like `stalled`, it existed in the journal and
    nothing called it.

    Parameters
    ----------
    last_k : int, optional
        Attempts that must all have been the same move.
    """
    def _repeating(state):
        journal = state.get("journal")
        return journal is not None and journal.is_repeating(last_k)

    return nudge(_repeating,
                 f"The last {last_k} actions were the same. Their results "
                 "will not change. Either conclude from what they returned "
                 "or try something different.",
                 name="repeating")


def check(fn, name: str = "check"):
    """
    Stop when *fn* says the objective is met — and call that a **success**.

    The only completion the harness can verify itself. Everything else the
    agent reports about its own progress is the model's word for it.

    Parameters
    ----------
    fn : callable
        Given the run's state; returns true when the objective is met.
    name : str, optional
        How the stop is reported: ``stopped_by="check:<name>"``.
    """
    def _c(state):
        # A raise is the loop's to report — as `condition_error`, the way a
        # ceiling's or a nudge's is — where it was a DEBUG line and "not yet".
        return f"check:{name}" if fn(state) else None
    _c.kind, _c.name = "check", name
    return _c


# ── Synthetic tool schemas ───────────────────────────────────────────────────

