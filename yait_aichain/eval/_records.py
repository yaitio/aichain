"""What one attempt looks like, on disk and in memory.

Every number is computed from these records, never from a running tally, and
a ``Record`` is a plain dict on disk so anything that reads JSON lines can
recount it — and ``Report.of()`` can recount anyone else's.
See docs/control/eval.md.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any
from collections.abc import Iterable, Iterator


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

    Attributes
    ----------
    id : str
        The join key across arms and trials; stable from run to run.
    input : any
        What an arm is given.
    expect : any
        What a correct answer looks like, for the scorer.
    group : str
        A stratum to report apart — ``by_group()`` reads it.
    meta : dict
        Anything else an arm or a scorer needs.
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

    Attributes
    ----------
    arm : str
        The arm that made the attempt.
    case : str
        The case's id.
    trial : int
        Which attempt, from 1.
    ok : bool
        The scorer's verdict.
    score : float
        Its magnitude, 0 to 1.
    output : any
        What the arm returned.
    error : str
        The exception, if the arm raised.
    group : str
        The case's group.
    seconds : float
        How long the attempt took.
    tokens : int
        Tokens it reported.
    cost : float
        Dollars it reported.
    meta : dict
        The scorer's reasons and anything else recorded with it.
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
    cost: float | None = 0.0
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
    # A line a crash left without its newline would swallow the next record.
    torn = False
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            torn = fh.read(1) != b"\n"
    with open(path, "a", encoding="utf-8") as fh:
        if torn:
            fh.write("\n")
        for r in records:
            # `repr` for what JSON cannot hold — bytes, a set, a datetime — so
            # one odd output does not end the run and the attempts behind it.
            fh.write(json.dumps(r.as_dict(), ensure_ascii=False, default=repr) + "\n")
            n += 1
    return n


def read(path) -> list[Record]:
    """Read back what ``write`` produced, skipping lines a crash left broken."""
    out: list[Record] = []
    broken = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                broken += 1                   # a partial line a crash left, not data
                continue
            known = {k: v for k, v in row.items() if k in Record.__dataclass_fields__}
            extra = {k: v for k, v in row.items() if k not in Record.__dataclass_fields__}
            if extra:                         # written by a newer version: kept, not dropped
                known["meta"] = {**extra, **(known.get("meta") or {})}
            out.append(Record(**known))
    if broken:
        import warnings
        warnings.warn(f"{path}: {broken} line(s) could not be read and were skipped",
                      RuntimeWarning, stacklevel=2)
    return out


def _as_bool(value) -> bool:
    if isinstance(value, str):
        word = value.strip().lower()
        if word in ("true", "1", "yes", "pass", "passed", "ok"):
            return True
        if word in ("false", "0", "no", "fail", "failed", ""):
            return False
        raise ValueError(f"cannot read {value!r} as pass or fail")
    return bool(value)


def adapt(rows: Iterable[dict], **keys) -> Iterator[Record]:
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
        # Other people's rows say "false" and "3": read as what they mean,
        # since the string "false" is truthy.
        if "ok" in mapped:
            mapped["ok"] = _as_bool(mapped["ok"])
        for key, kind in (("trial", int), ("score", float), ("tokens", int), ("seconds", float)):
            if key in mapped and mapped[key] is not None:
                mapped[key] = kind(mapped[key])
        if mapped.get("cost") is not None:
            mapped["cost"] = float(mapped["cost"])
        mapped["meta"] = {**rest, **(mapped.get("meta") or {})}
        yield Record(**mapped)
