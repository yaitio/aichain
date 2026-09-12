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
            # Fail-closed accounting: excluded from the count, never recorded
            # as a clean verdict. Raising — which this did until 2026-09-12 —
            # was right about the important half (guessing "pass" raises every
            # arm at once and reads as a good result) and wrong about the
            # rest: it ends the whole run over one row a judge could not read.
            # The arm that a judge abstained on twenty times has not scored
            # 0.95 out of the eighty it could read; it has an instrument that
            # does not work, and that is only visible if the abstentions are
            # counted and shown.
            return abstain(f"no verdict for case {case.id!r}: "
                           f"{reply.strip()[:120]!r}")
        passed = found.group(1).upper() == "PASS"
        usage  = getattr(skill, "last_usage", None)
        return {"ok": passed, "score": 1.0 if passed else 0.0,
                # billed to the instrument, not to the arm — kept apart so a
                # cost table cannot quietly include the cost of grading
                "judge_tokens": getattr(usage, "total_tokens", 0) or 0,
                "judge_cost":   getattr(usage, "cost", 0.0) or 0.0}
    return _score


def abstain(why: str) -> dict:
    """A verdict of *no verdict* — the judge could not judge this one.

    ``ok`` is False so nothing downstream that ignores the flag can read an
    abstention as a pass; ``abstained`` is what :class:`~._report.Report`
    reads, and it removes the row from the denominator rather than counting it
    as a failure. Those two are different claims and the difference is the
    whole point: a judge that cannot read an answer has said nothing about it.
    """
    return {"ok": False, "score": 0.0, "abstained": True, "why": why}


def pairwise(model, *, rubric: str = "Which answer better fulfils the "
                                     "question? Consider accuracy first, "
                                     "then completeness, then concision.",
             champion: str = "the reference") -> Scorer:
    """
    A judge compares the candidate with the champion, **in both orderings**.

    LLM judges have a documented preference for whichever answer they read
    first, and a single-ordering comparison measures that preference as much
    as it measures quality. So the pair is put twice, A/B and B/A, and the
    candidate wins only by winning both. A disagreement between the two
    orderings *is* the position bias showing itself, and it resolves **to the
    champion**: the burden is on the challenger, which is what keeps a
    best-so-far from drifting on noise.

    The champion is ``case.expect`` — the same field :func:`judge` reads as
    the reference.

    ``extra`` carries what each ordering said, so a run can report how often
    they disagreed. That number is the judge's own reliability, measured for
    free while grading, and a pair of orderings that disagree half the time
    means the comparison is a coin toss wearing a rubric.
    """
    from ..models import Model
    from ..skills import Skill

    m = model if isinstance(model, Model) else Model(model)

    def _ask(question, first, second):
        text = (f"{rubric}\n\nAnswer with exactly one word on the first "
                f"line — FIRST or SECOND — and nothing else on that line.\n\n"
                f"QUESTION:\n{question}\n\nFIRST:\n{first}\n\n"
                f"SECOND:\n{second}")
        skill = Skill(m, {"messages": [{"role": "user", "parts": [text]}]})
        reply = _text(skill.run()).strip()
        head = _re.match(r"^\W*(FIRST|SECOND)\b", reply, _re.IGNORECASE)
        usage = getattr(skill, "last_usage", None)
        return (head.group(1).upper() if head else None,
                getattr(usage, "total_tokens", 0) or 0,
                getattr(usage, "cost", 0.0) or 0.0)

    def _score(case, output):
        question  = _text(case.input)
        candidate = _text(output)
        reference = _text(case.expect)

        first_up,  t1, c1 = _ask(question, candidate, reference)
        second_up, t2, c2 = _ask(question, reference, candidate)
        if first_up is None or second_up is None:
            return abstain(f"unreadable comparison for case {case.id!r}")

        # The candidate led in run one and trailed in run two, so winning
        # means FIRST then SECOND.
        won_both = (first_up == "FIRST" and second_up == "SECOND")
        agreed   = ((first_up == "FIRST") == (second_up == "SECOND"))
        return {"ok": won_both, "score": 1.0 if won_both else 0.0,
                "orderings_agreed": agreed,
                "champion": champion,
                "judge_tokens": t1 + t2, "judge_cost": c1 + c2}
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
