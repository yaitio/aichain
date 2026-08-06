"""
What one attempt looks like, on disk and in memory.

The whole package is built on one rule: **every number is computed from these
records, never from a running tally.** A tally is written once and read many
times, so an error in it is invisible; a record can be recounted, and was —
a report parser once returned 0.878 where the raw data held 0.798, and the
difference was found only because the raw data still existed.

That is also why ``Record`` is a plain dict on disk. Anything that reads JSON
lines can recount our numbers, and we can recount anyone else's: the τ²-bench
runs and the byheart retrieval bench both produce their own shapes, and
``Report.of()`` takes them once they are mapped onto these keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Iterator


@dataclass
class Case:
    """
    One unit of work, and what a correct answer to it looks like.

    ``id`` is the join key across arms and trials, so it has to be stable
    across runs — a position in a list is not an id, because inserting a case
    renumbers every one after it and a comparison against yesterday silently
    pairs the wrong rows.

    ``group`` is optional and exists for stratified reporting: the A+
    regression set splits its tasks into always-solved / always-failed /
    contested, and the interesting result lived entirely in how the groups
    behaved differently. A benchmark reported only in aggregate cannot show
    that.
    """

    id: str
    input: Any = None
    expect: Any = None
    group: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class Record:
    """
    One arm's attempt at one case, on one trial.

    ``ok`` is the scorer's verdict and ``score`` its magnitude, kept apart on
    purpose. All-or-nothing throws away the difference between three of four
    and none of four; a mean throws away whether anything was ever fully
    right. Both are cheap, so both are kept.

    ``error`` being set does not mean ``ok`` is False by definition — a
    scenario may recover — but it is what the validity guards count, and a
    cell with too many of them is rejected rather than reported.
    """

    arm: str
    case: str
    trial: int
    ok: bool = False
    score: float = 0.0
    output: Any = None
    error: str = ""
    group: str = ""
    seconds: float = 0.0
    tokens: int = 0
    cost: float = 0.0
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def write(records: Iterable[Record], path) -> int:
    """
    Append records as JSON lines.

    Append rather than rewrite: a run that dies half-way keeps what it had,
    and a truncated write costs one line instead of the file. The first A+ run
    produced 450 simulations and survives only as a table someone pasted into
    a chat, because nothing was ever written to a path that outlived the
    process.
    """
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read(path) -> "list[Record]":
    """Read back what ``write`` produced, skipping lines a crash left broken."""
    out: list[Record] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Record(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue                      # a partial last line, not data
    return out


def adapt(rows: Iterable[dict], **keys) -> "Iterator[Record]":
    """
    Map someone else's rows onto ``Record``.

    ``adapt(rows, arm="agent", case="task_id", ok="success")`` — the keyword is
    our field, the value is theirs. Anything unmapped keeps its own name and
    lands in ``meta``, so nothing is dropped on the way in.

    This exists because the two benches that most need this package do not run
    through it: τ²-bench owns its own loop, and the byheart retrieval bench
    scores documents rather than answers. Both can still be reported here, and
    reporting is where the mistakes were.
    """
    ours = {f for f in Record.__dataclass_fields__ if f != "meta"}
    for row in rows:
        mapped, rest = {}, dict(row)
        for our, theirs in keys.items():
            if theirs in rest:
                mapped[our] = rest.pop(theirs)
        for k in list(rest):
            if k in ours and k not in mapped:
                mapped[k] = rest.pop(k)
        mapped.setdefault("arm", "")
        mapped.setdefault("case", "")
        mapped.setdefault("trial", 0)
        mapped["meta"] = {**rest, **(mapped.get("meta") or {})}
        yield Record(**mapped)
