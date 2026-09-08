"""
agent._result
=============

``AgentResult`` — structured return type for every ``agent.run()`` call.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    """
    Structured return value from :meth:`Agent.run`.

    Attributes
    ----------
    success : bool
        ``True`` when the agent produced a final answer.
        ``False`` when it stopped due to a fatal error, budget exhaustion,
        or an unrecoverable step failure.

    output : str | dict | None
        The final answer produced by the agent.  ``None`` when
        *success* is ``False``.

    mode : str
        The mode the agent ran in — ``"waterfall"`` or ``"agile"``.

    steps_taken : int
        Number of plan steps that were executed (including retries).

    tokens_used : int
        Total tokens consumed across *all* LLM calls: planning,
        action determination, skill execution, and reflection.

    cost : float | None
        Estimated USD spent on those calls, priced per model from the provider
        registry.  ``None`` when nothing could be priced — a self-hosted model
        has no price per token, and this reports that rather than inventing a
        zero.

        Not derivable from ``tokens_used``: that field sums input and output
        together, and output costs five to six times input at frontier rates.

        Covers the calls made by *this* invocation.  A run picked up with
        :meth:`~Agent.resume` therefore reports only the resumed leg, because
        cost is not written to the checkpoint the way ``tokens_used`` is.

    plan : list[dict]
        The final plan the agent executed.  In agile mode this may
        differ from the original plan if replanning occurred.

    history : list[dict]
        Full step-by-step execution trace.  Each record contains:

        * ``step``       — 0-based step index in the plan
        * ``attempt``    — 1-based attempt counter for this step
        * ``step_goal``  — what the step was trying to achieve
        * ``action_type``— ``"tool"`` or ``"skill"``
        * ``action``     — the action dict produced by the orchestrator
        * ``output``     — raw output from the step
        * ``exec_error`` — exception message if execution failed, else ``None``
        * ``stored_as``  — memory key where output was stored, else ``None``
        * ``reflection`` — the orchestrator's assessment dict
        * ``tokens``     — tokens used for this step's LLM calls

    memory : dict
        Final snapshot of the agent's memory after all steps completed.

    error : str | None
        Human-readable reason for failure.  ``None`` when *success* is
        ``True``.

    Examples
    --------
    ::

        result = agent.run("Research fusion energy and write a summary in French")

        if result:
            print(result.output)
            print(f"Used {result.tokens_used} tokens in {result.steps_taken} steps")
        else:
            print("Failed:", result.error)

        # Inspect the full trace
        for rec in result.history:
            print(f"Step {rec['step']+1} [{rec['action_type']}] {rec['step_goal']}")
            print(f"  Assessment: {rec['reflection'].get('assessment')}")
    """

    success:     bool
    output:      Any
    mode:        str
    steps_taken: int
    tokens_used: int
    cost:        "float | None" = None
    #: What ended the run — ``"answered"``, ``"check:<name>"``, ``"step_count"``,
    #: ``"token_budget"``, ``"cost_budget"``, or an error name. Without it a run
    #: that finished and a run that ran out of budget look identical from the
    #: outside, which is exactly how a benchmark came to score four zeroes it
    #: could not tell apart.
    stopped_by:  "str | None" = None
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

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        status = "success" if self.success else f"failed({self.error!r})"
        return (
            f"AgentResult({status}, mode={self.mode!r}, "
            f"steps={self.steps_taken}, tokens={self.tokens_used})"
        )
