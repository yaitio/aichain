"""
tools._wait — Wait / Gate
=========================

Suspend tools: the developer-facing way to pause a run until an EXTERNAL signal
arrives. The signal source is irrelevant to the library — a human clicking
approve, a cron tick, a webhook from another service, a queue message: all of
them simply call ``chain.resume(run_id, signal)`` later. The engine only parks
the run; it never schedules or waits actively.

* ``Wait``  — a leaf step: pause; on resume the signal becomes this step's
  output (and thus enters ``variables``).
* ``Gate``  — wraps any Tool: pause for a decision; on resume run the wrapped
  tool when approved, or skip it.

Both are thin sugar over the internal ``Suspend`` signal that the engine
catches; nothing here is engine-specific.
"""

from __future__ import annotations

from ._base import Tool
from ..state import Suspend

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

    def __init__(self, reason: str = "", resume_with: "dict | None" = None,
                 *, name: str = "wait", hint: "dict | None" = None) -> None:
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

    On first reach the run pauses; on resume, if the signal grants approval
    (``signal[decision_key]`` is truthy) the wrapped tool runs with the same
    arguments, otherwise it is skipped.

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

    def __init__(self, tool: Tool, *, reason: str = "",
                 resume_with: "dict | None" = None,
                 decision_key: str = "approved",
                 name: "str | None" = None,
                 hint: "dict | None" = None) -> None:
        tool_name = getattr(tool, "name", "tool")
        self._tool         = tool
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
        if _signal.get(self._decision_key):
            return self._tool.run(**kwargs)            # approved → run the action
        return {self._decision_key: False, "skipped": True}
