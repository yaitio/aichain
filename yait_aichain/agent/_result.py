"""
agent._result
=============

``AgentResult`` — structured return type for every ``agent.run()`` call.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from .. import _hints


@dataclass
class AgentResult:
    """
    What :meth:`Agent.run` returns: the answer, how the run ended, and what it
    cost. ``bool(result)`` is ``result.success``.

    Attributes
    ----------
    success : bool
        ``True`` when the agent answered, or a ``check()`` passed.
    output : str | dict | None
        The answer. ``None`` when *success* is ``False``.
    mode : str
        ``"agile"`` or ``"waterfall"``.
    steps_taken : int
        Model turns taken.
    tokens_used : int
        Tokens across every model call of the run, input and output together.
    cost : float | None
        Dollars, priced per model from the registry; ``None`` when nothing
        could be priced, never an invented zero. A run picked up with
        ``resume()`` includes what it spent before the pause.
    stopped_by : str | None
        What ended the run: ``"answered"``, ``"check:<name>"``,
        ``"step_count"``, ``"token_budget"``, ``"cost_budget"``,
        ``"truncated"`` (a reply cut at ``max_tokens`` — raise it),
        ``"llm_error"`` (the model call failed otherwise),
        ``"condition_error"`` (a check, ceiling, nudge or approver of the
        caller's raised; ``error`` says which), or the name of an error.
    plan : list[dict]
        In ``waterfall`` mode, the plan it wrote — ``{"goal": …}`` per step.
    adaptations : list
        What the library changed in this run's requests to fit the provider,
        each an ``Adaptation``; empty when nothing was.
    history : list[dict]
        Empty for an agent, kept so every result has the same fields; an
        agent's record is ``journal``.
    memory : dict
        Always empty; the agent keeps no memory store. Removed in 5.0.
    error : str | None
        Why it failed, when it did.
    journal : list[dict]
        What was attempted, in order, each entry with its outcome and the
        evidence behind it — ``check`` for a fact the code verified,
        ``model_claim`` for the model's word.
    acceptance : Acceptance | None
        What the answer rests on: ``checked``, ``claimed`` or ``unsupported``,
        with a fingerprint of the evidence, so ``still_holds()`` can tell when
        it has changed.
    beacons : list
        Beacons raised during the run, including by delegated workers — a
        blocker, a question, a contract that changed.

    Examples
    --------
    ::

        result = agent.run("Research fusion energy and summarise it in French")
        if result:
            print(result.output, result.cost, result.stopped_by)
        else:
            print("failed:", result.error)
    """

    success:     bool
    output:      Any
    mode:        str
    steps_taken: int
    tokens_used: int
    cost:        float | None = None
    #: What ended the run — ``"answered"``, ``"check:<name>"``, ``"step_count"``,
    #: ``"token_budget"``, ``"cost_budget"``, or an error name. Without it a run
    #: that finished and a run that ran out of budget look identical from the
    #: outside, which is exactly how a benchmark came to score four zeroes it
    #: could not tell apart.
    stopped_by:  str | None = None
    plan:        list[dict]     = field(default_factory=list)
    #: Everything the library changed about this run's requests to fit the
    #: provider — reasoning expressed as a thinking budget, sampling a
    #: reasoner refuses, a model swapped underneath. Empty when nothing was
    #: changed. Kept on the result because a warning does not survive into a
    #: run record, and a comparison that cannot say what its arms sent is not
    #: a comparison.
    adaptations: list           = field(default_factory=list)
    history:     list[dict]     = field(default_factory=list)
    memory:      dict           = field(default_factory=dict)
    error:       str | None     = None
    #: Append-only record of what was attempted (see ``agent._journal``): each
    #: entry carries an outcome and the evidence behind it. ``done`` entries
    #: whose evidence is ``model_claim`` were asserted, not verified.
    journal:     list[dict]     = field(default_factory=list)

    #: What this result rests on. ``kind`` is ``checked`` when a programmatic
    #: fact supports it, ``claimed`` when only a model's word does — a
    #: delegated worker's report is the common case — and ``unsupported``
    #: when nothing does. Reported, not enforced: an unsupported answer is
    #: still returned, and the caller can see that it is one.
    acceptance:  object | None = None

    #: Beacons raised during the run, including by delegated workers — a
    #: blocker, a question, a contract that changed under the task.
    beacons:     list           = field(default_factory=list)

    def __getattr__(self, name):
        raise _hints.missing(self, name)

    @property
    def usage(self):
        """Tokens and cost as a `Usage`, the field `ChainResult` and
        `PoolResult` carry too. An agent sums only the total across its
        turns, so the input/output split is left at zero rather than invented."""
        from ..models._usage import Usage
        if not self.tokens_used and self.cost is None:
            return None
        return Usage(total_tokens=self.tokens_used or 0, cost=self.cost)

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        status = "success" if self.success else f"failed({self.error!r})"
        return (
            f"AgentResult({status}, mode={self.mode!r}, "
            f"steps={self.steps_taken}, tokens={self.tokens_used})"
        )
