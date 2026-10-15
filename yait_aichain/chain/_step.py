"""
chain._step — one step of a Chain, said in full.

A tuple says the common cases — ``fetch``, ``(fetch, "page")``,
``(count, {"text": "page.body"})``, ``(count, "words", {"text": "page.body"})``
— and `Step` says the rest: a step's own error policy, and whether a dict it
returns is spread into the chain's variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._errors_policy import POLICIES


@dataclass
class Step:
    """
    One step of a :class:`~yait_aichain.Chain`.

    Parameters
    ----------
    runner : Skill | Tool | Agent | Chain | callable
        What runs.
    out : str | None, optional
        The key the output is stored under; the runner's name by default.
    inputs : dict | None, optional
        Where each input comes from — a variable's name, a path into one
        (``"page.body"``), or a template.
    on_error : str | None, optional
        This step's error policy, over the chain's: ``"raise"``, ``"stop"``,
        ``"skip"`` or ``"collect"``.
    spread : bool | None, optional
        A dict output's fields are also put among the chain's variables, as
        every dict output was before 4.2.7. ``None`` keeps that for now, with a
        warning; ``True`` keeps it on purpose; ``False`` keeps the dict under
        its key alone.
    options : dict | None, optional
        An Agent step's options: ``task_key``, ``output_field``.
    """

    runner: Any
    out: str | None = None
    inputs: dict | None = None
    on_error: str | None = None
    spread: bool | None = None
    options: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.on_error is not None and self.on_error not in POLICIES:
            raise ValueError(f"on_error is one of {sorted(POLICIES)}; got {self.on_error!r}")
        if self.out is not None and not isinstance(self.out, str):
            raise ValueError(f"out is the output's key, a str; got {self.out!r}")
        if self.inputs is not None and not isinstance(self.inputs, dict):
            raise ValueError(f"inputs maps each input to its source, a dict; got {self.inputs!r}")


__all__ = ["Step"]
