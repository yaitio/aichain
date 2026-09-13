"""
pool._result — PoolResult
=========================

What ``Pool.run()`` returns: the outputs in item order — iterate it, index it,
take its length — plus the fields ``ChainResult`` and ``AgentResult`` carry:
``output``, ``success``, ``error``, ``history``, ``usage``, ``tokens_used``,
``cost``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class PoolResult(Sequence):
    """
    The outcome of one pool run.

    Attributes
    ----------
    output : list
        One output per item, in item order; ``None`` for an item that failed
        or was never started.
    success : bool
        ``True`` when every item completed.
    error : str | None
        The first failure, as ``"item N: message"``; ``None`` on success.
    errors : list[str | None]
        One entry per item.
    history : list[dict]
        One record per item: status, output, error, failure, duration, usage.
    usage : Usage | None
        Summed across the items that reported usage.
    beacons : list
        Beacons raised by any item.
    """

    def __init__(self, outputs: list, *, success: bool, history: list,
                 usage: Any = None, beacons: "list | None" = None) -> None:
        self._outputs = list(outputs)
        self.success = success
        self.history = list(history)
        self.usage = usage
        self.beacons = list(beacons or [])

    @property
    def output(self) -> list:
        return list(self._outputs)

    @property
    def errors(self) -> list:
        return [r.get("error") for r in self.history]

    @property
    def error(self) -> "str | None":
        for r in self.history:
            if r.get("error"):
                return f"item {r['index']}: {r['error']}"
        return None

    @property
    def tokens_used(self) -> int:
        return (getattr(self.usage, "total_tokens", 0) or 0) if self.usage else 0

    @property
    def cost(self) -> "float | None":
        return getattr(self.usage, "cost", None) if self.usage else None

    def __getitem__(self, index):
        return self._outputs[index]

    def __len__(self) -> int:
        return len(self._outputs)

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        state = "ok" if self.success else f"failed: {self.error}"
        return f"PoolResult({state}, items={len(self._outputs)})"
