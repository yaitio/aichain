"""
Deciding whether one answer was right.

A scorer takes ``(case, output)`` and returns a verdict: ``True``/``False``, a
number in ``[0, 1]``, or both as a pair. The runner normalises whichever shape
comes back, so a scorer can be three lines of your own code and still fit.

Two rules are enforced here rather than left to the caller, because both have
already been paid for:

**A scorer that cannot decide must not say yes.** An LLM judge whose reply
fails to parse is a broken instrument, and the tempting default — treat it as a
pass — inflates every arm at once and looks like a good day. ``judge`` raises
instead, and the runner records the attempt as an error, where the validity
guards can see it.

**Silence is not precision.** A scenario that reports nothing has no wrong
answers in it, so any ratio computed per case hands it a perfect score. That
is not fixed here — it is fixed in :mod:`._report`, which pools counts across
cases before dividing. It is mentioned here because the scorer is where the
mistake feels natural.
"""

from __future__ import annotations

import re as _re
from typing import Any, Callable

Scorer = Callable[[Any, Any], Any]


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "answer", "output"):
            if isinstance(value.get(key), str):
                return value[key]
    return "" if value is None else str(value)


def _norm(s: str, fold: bool) -> str:
    s = " ".join(s.split())
    return s.casefold() if fold else s


# ── Rule-based ───────────────────────────────────────────────────────────────

def exact(*, fold: bool = True) -> Scorer:
    """The answer equals ``case.expect``, whitespace collapsed."""
    def _score(case, output):
        return _norm(_text(output), fold) == _norm(_text(case.expect), fold)
    return _score


def contains(*, fold: bool = True, all_of: bool = True) -> Scorer:
    """
    Every string in ``case.expect`` appears in the answer.

    ``expect`` may be one string or a list. With ``all_of=False`` any single
    hit passes, and the partial score is the fraction found — the difference
    between three of four and none of four is the whole reason a benchmark
    discriminates, and an all-or-nothing verdict throws it away.
    """
    def _score(case, output):
        want = case.expect if isinstance(case.expect, (list, tuple)) else [case.expect]
        hay  = _norm(_text(output), fold)
        hits = sum(1 for w in want if _norm(_text(w), fold) in hay)
        frac = hits / len(want) if want else 0.0
        return (hits == len(want) if all_of else hits > 0), frac
    return _score


def regex(*, flags: int = 0) -> Scorer:
    """``case.expect`` is a pattern the answer must match somewhere."""
    def _score(case, output):
        return bool(_re.search(_text(case.expect), _text(output), flags))
    return _score


def numeric(*, tolerance: float = 0.0) -> Scorer:
    """The first number in the answer equals ``case.expect``, within tolerance."""
    def _score(case, output):
        found = _re.search(r"-?\d+(?:[.,]\d+)?", _text(output))
        if not found:
            return False, 0.0
        try:
            got  = float(found.group(0).replace(",", "."))
            want = float(case.expect)
        except (TypeError, ValueError):
            return False, 0.0
        return abs(got - want) <= tolerance, 1.0 if abs(got - want) <= tolerance else 0.0
    return _score


# ── Model-based ──────────────────────────────────────────────────────────────

_JUDGE_PROMPT = (
    "You are grading one answer against a reference. Reply with exactly one "
    "word on the first line — PASS or FAIL — and nothing else on that line.\n\n"
    "{rubric}\n\n"
    "QUESTION:\n{question}\n\nREFERENCE:\n{expect}\n\nANSWER:\n{answer}"
)

_VERDICT = _re.compile(r"^\W*(PASS|FAIL)\b", _re.IGNORECASE)


def judge(model, *, rubric: str = "The answer is correct if it states the same "
                                  "fact as the reference. Wording may differ.",
          prompt: str = _JUDGE_PROMPT) -> Scorer:
    """
    An LLM decides. Use when the answer is prose and no rule can check it.

    The judge is part of the instrument, not part of the system under test, so
    two things follow. It must be **the same model and prompt across every
    arm** — a judge swapped mid-run makes the arms incomparable in a way that
    leaves no trace in the numbers. And it must be **checked before it is
    trusted**: run the oracle and noise controls from :mod:`._report`, which
    feed it known-correct and known-wrong answers. A judge that does not score
    the oracle near 1.0 is measuring something other than correctness.

    An unparseable verdict raises. That is deliberate: the alternative is to
    guess, and guessing "pass" raises every arm at once, which reads as a good
    result rather than as a broken judge.
    """
    from ..models import Model
    from ..skills import Skill

    m = model if isinstance(model, Model) else Model(model)

    def _score(case, output):
        text = prompt.format(rubric=rubric,
                             question=_text(case.input),
                             expect=_text(case.expect),
                             answer=_text(output))
        skill = Skill(m, {"messages": [{"role": "user", "parts": [text]}]})
        reply = _text(skill.run())
        found = _VERDICT.match(reply.strip())
        if not found:
            raise ValueError(
                f"judge returned no verdict for case {case.id!r}: "
                f"{reply.strip()[:120]!r}")
        passed = found.group(1).upper() == "PASS"
        usage  = getattr(skill, "last_usage", None)
        return {"ok": passed, "score": 1.0 if passed else 0.0,
                # billed to the instrument, not to the arm — kept apart so a
                # cost table cannot quietly include the cost of grading
                "judge_tokens": getattr(usage, "total_tokens", 0) or 0,
                "judge_cost":   getattr(usage, "cost", 0.0) or 0.0}
    return _score


# ── Combination ──────────────────────────────────────────────────────────────

def all_of(*scorers: Scorer) -> Scorer:
    """Passes when every scorer passes; the score is the mean of theirs."""
    def _score(case, output):
        verdicts = [normalise(s(case, output)) for s in scorers]
        return (all(v[0] for v in verdicts),
                sum(v[1] for v in verdicts) / len(verdicts) if verdicts else 0.0)
    return _score


def normalise(verdict: Any) -> "tuple[bool, float, dict]":
    """
    Whatever a scorer returned, as ``(ok, score, extra)``.

    Accepts a bool, a number, a ``(ok, score)`` pair, or a dict carrying at
    least ``ok``. Anything else is a programming error and says so — a scorer
    that silently returns ``None`` would otherwise mark every case failed and
    look like a bad model.
    """
    if isinstance(verdict, bool):
        return verdict, 1.0 if verdict else 0.0, {}
    if isinstance(verdict, (int, float)):
        return bool(verdict), float(verdict), {}
    if isinstance(verdict, dict):
        ok = bool(verdict.get("ok"))
        extra = {k: v for k, v in verdict.items() if k not in ("ok", "score")}
        return ok, float(verdict.get("score", 1.0 if ok else 0.0)), extra
    if isinstance(verdict, tuple) and len(verdict) == 2:
        return bool(verdict[0]), float(verdict[1]), {}
    raise TypeError(
        f"a scorer must return bool, number, (ok, score) or a dict with 'ok'; "
        f"got {type(verdict).__name__}")
