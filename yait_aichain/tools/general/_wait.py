"""``Wait`` and ``Gate``: tools that park a run until an external signal arrives.

``Wait`` pauses, and on resume the signal becomes the step's output; ``Gate``
wraps a tool and runs it on resume only when the signal approves. Both raise
the engine's ``Suspend``; the engine parks the run and never waits actively.
See docs/pipelines/pause-and-resume.md.
"""

from __future__ import annotations

from .._base import Tool
from ...state import Suspend

_EMPTY_PARAMS = {"type": "object", "properties": {}, "required": []}


class Wait(Tool):
    """
    Pause the run until an external signal arrives.

    Parameters
    ----------
    reason : str
        Why the run is paused (shown to whoever/whatever must act).
    resume_with : dict | None
        Schema of the expected resume signal, e.g. ``{"approved": "bool"}``.
    name : str
        Step name (default ``"wait"``).
    hint : dict | None
        Optional hint for an external scheduler/trigger (e.g.
        ``{"wake_at": "..."}`` or ``{"on": "event"}``); recorded, never acted on.

    On resume the ``signal`` passed to ``resume()`` becomes this step's output.
    """

    risk = "read"   # it waits; the run's next step decides

    suspends = True

    def __init__(self, reason: str = "", resume_with: dict | None = None,
                 *, name: str = "wait", hint: dict | None = None) -> None:
        self.name        = name
        self.description  = reason or "Pause the run until an external signal arrives."
        self.parameters   = _EMPTY_PARAMS
        self._reason      = reason
        self._resume_with = resume_with or {}
        self._hint        = hint

    def run(self, _signal=None, **kwargs):
        if _signal is None:
            raise Suspend(self._reason, self._resume_with, hint=self._hint)
        return _signal


class Gate(Tool):
    """
    Gate any Tool behind an external signal.

    On first reach the run pauses; on resume the wrapped tool runs with the
    same arguments when ``signal[decision_key]`` is ``True`` — the boolean,
    nothing that merely looks true — and is skipped when it is ``False``.
    Anything else is a ``TypeError`` naming the shape expected: a form, Slack
    or a query string sends ``"false"``, and a check for truthiness read that
    as yes. The gate carries the wrapped tool's risk, so a permission policy
    sees ``gate(wipe_table)`` as the destructive call it guards.

    Parameters
    ----------
    tool : Tool
        The tool to run only after approval (typically a side-effecting action
        such as sending a refund or an email).
    reason : str
        Why approval is needed.
    resume_with : dict | None
        Resume-signal schema (default ``{decision_key: "bool"}``).
    decision_key : str
        Key in the signal that grants approval (default ``"approved"``).
    name : str | None
        Step name (default ``"gate_<tool>"``).
    hint : dict | None
        Optional external-scheduler hint.
    """

    suspends = True

    def __init__(self, tool: Tool, *, reason: str = "",
                 resume_with: dict | None = None,
                 decision_key: str = "approved",
                 name: str | None = None,
                 hint: dict | None = None) -> None:
        if not isinstance(tool, Tool):
            if not callable(tool):
                raise TypeError(f"Gate wraps a tool; got {type(tool).__name__}")
            from .._base import tool as as_tool
            tool = as_tool(tool)               # a plain function, checked now, not after approval
        tool_name = tool.name
        self._tool         = tool
        self.risk          = tool.risk
        self._decision_key = decision_key
        self.name          = name or f"gate_{tool_name}"
        self.description   = (f"Gate '{tool_name}' behind approval. "
                              f"{getattr(tool, 'description', '')}").strip()
        # Mirror the wrapped tool's schema so the chain builds the right kwargs.
        self.parameters    = getattr(tool, "parameters", _EMPTY_PARAMS)
        self._reason       = reason or f"Approve running '{tool_name}'?"
        self._resume_with  = resume_with or {decision_key: "bool"}
        self._hint         = hint

    def run(self, _signal=None, **kwargs):
        if _signal is None:
            raise Suspend(self._reason, self._resume_with, hint=self._hint)
        decision = _signal.get(self._decision_key) if isinstance(_signal, dict) else None
        if decision is True:
            return self._tool.run(**kwargs)            # approved → run the action
        if decision is False:
            # The whole answer, so a refusal's reason reaches the model.
            return {**_signal, "skipped": True}
        raise TypeError(f"gate {self.name!r} resumes with {{{self._decision_key!r}: True}} or "
                        f"{{{self._decision_key!r}: False}}; got {_signal!r}")
