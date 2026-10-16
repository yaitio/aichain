"""
agent._stops — how a run ends: ceilings, checks, nudges, the verify gate.
"""

from __future__ import annotations

import logging

from .._dispatch import MAX_ARGUMENT_DEPTH  # noqa: F401  (named in the release notes)


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

    When *predicate(state)* is true, *message* is appended as a user turn and the
    loop carries on — once per streak, re-armed when the predicate clears.

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

    An answer after a ``write``/``external``/``financial``/``destructive`` call is
    held and *message* is put to the model, unless a later call succeeded. Once per
    run; only a call clears it, never the model saying it checked.

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


def check(fn, name: str = "check", *, output=None, answer: bool = False):
    """Stop when *fn* says the objective is met — a **success**, the only one the
    harness verifies itself. The output is what *output* returns, else with *answer*
    the model's closing reply, else the last tool result; ``output_from`` says which.

    Parameters
    ----------
    fn : callable
        Given the run's state; true when the objective is met.
    name : str, optional
        Reported as ``stopped_by="check:<name>"``.
    output : callable | None, optional
        Given the run's state; returns the output.
    answer : bool, optional
        One more model reply, without tools, as the output; it costs a call.
    """
    def _c(state):
        # A raise is the loop's to report, as `condition_error`, the way a
        # ceiling's or a nudge's is.
        return f"check:{name}" if fn(state) else None
    _c.kind, _c.name, _c.output, _c.answer = "check", name, output, answer
    return _c


# ── Synthetic tool schemas ───────────────────────────────────────────────────

