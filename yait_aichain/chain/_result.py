"""
chain._result — ChainResult, what ``Chain.run()`` and ``Chain.resume()`` return.

The same shape as ``AgentResult`` and ``PoolResult`` — ``output``, ``success``,
``error``, ``history``, ``usage``, ``tokens_used``, ``cost`` — and a step's
output read by its name: ``result["tweet"]``. See docs/pipelines/chain.md.
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
    renamed : dict
        Keys an output used to be found under, mapped to its key now —
        ``{"result": "fetch"}`` while the pre-4.2.7 key still works.
    """

    output:    Any
    success:   bool
    error:     str | None = None
    history:   list = field(default_factory=list)
    variables: dict = field(default_factory=dict)
    usage:     Any = None
    run_id:    str = ""
    #: An output's old keys, mapped to where it is now — ``{"result":
    #: "fetch"}``: reading the old one still works and says so.
    renamed:   dict = field(default_factory=dict, repr=False)

    @property
    def cost(self) -> float | None:
        """Dollars across every step that reported a price, agents included;
        None when no step could be priced — never an invented zero."""
        return getattr(self.usage, "cost", None)

    @property
    def evidence(self) -> str:
        """What the output rests on: ``"check"`` when every step that ran was a
        function or a tool — a programmatic fact — and ``"model_claim"`` when a
        Skill or an Agent produced part of it, so it is a model's word.

        The same two words the agent's journal uses, so a caller deciding
        whether to re-check a result reads one vocabulary for both."""
        kinds = {h.get("kind") for h in self.history}
        return "check" if kinds and kinds <= {"call", "tool"} else "model_claim"

    def __getattr__(self, name):
        raise _hints.missing(self, name)

    def __getitem__(self, key: str):
        if key in self.renamed:
            import warnings
            warnings.warn(f"result[{key!r}] is result[{self.renamed[key]!r}] since 4.2.7 — a step "
                          "is named after its runner; the old key goes in 5.0",
                          FutureWarning, stacklevel=2)
        if key not in self.variables and "." in key:
            # A path into a stored dict: result["page.body"].
            head, _, rest = key.partition(".")
            value = self.variables.get(head)
            for part in rest.split("."):
                if not isinstance(value, dict) or part not in value:
                    raise KeyError(f"{key!r}: no {part!r} there; available: "
                                   f"{sorted(value) if isinstance(value, dict) else sorted(self.variables)}")
                value = value[part]
            return value
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
