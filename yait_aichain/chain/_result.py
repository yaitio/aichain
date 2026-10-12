"""
chain._result — ChainResult
===========================

What ``Chain.run()`` and ``Chain.resume()`` return, in the same shape as
``AgentResult`` and ``PoolResult``: ``output``, ``success``, ``error``,
``history``, ``usage``, ``tokens_used``, ``cost``.

A chain names its steps, so a step's output is read by its name:
``result["tweet"]``. Measured 2026-09-13: given the
documentation, a model reached for exactly that in twenty of forty-seven
failed attempts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .. import _hints
from ..models._usage import Spend


@dataclass
class ChainResult(Spend):
    """
    The outcome of one chain run.

    Attributes
    ----------
    output : str | dict | None
        The last successful step's output.
    success : bool
        ``True`` when every step completed. A step failure under
        ``on_step_error="skip"``, ``"collect"`` or ``"stop"`` leaves it
        ``False`` — the run carried on, but not everything happened.
    error : str | None
        The first failure, as ``"step N (name): message"``; ``None`` on success.
    history : list[dict]
        One record per step that ran: input, output, and ``failure`` when it failed.
    variables : dict
        Every variable at the end — the initial ones and each step's output
        under its key. ``result["key"]`` reads from here.
    usage : Usage | None
        Tokens and cost summed across the steps that reported them.
    run_id : str
        The run document's id.
    """

    output:    Any
    success:   bool
    error:     "str | None" = None
    history:   list = field(default_factory=list)
    variables: dict = field(default_factory=dict)
    usage:     Any = None
    run_id:    str = ""

    def __getattr__(self, name):
        raise _hints.missing(self, name)

    def __getitem__(self, key: str):
        try:
            return self.variables[key]
        except KeyError:
            raise KeyError(
                f"{key!r} is not a step output or variable of this run; "
                f"available: {sorted(self.variables)}") from None

    def get(self, key: str, default=None):
        return self.variables.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.variables

    def __bool__(self) -> bool:
        return self.success

    def __str__(self) -> str:
        return "" if self.output is None else str(self.output)

    def __repr__(self) -> str:
        state = "ok" if self.success else f"failed: {self.error}"
        return f"ChainResult({state}, steps={len(self.history)}, output={self.output!r:.60})"
