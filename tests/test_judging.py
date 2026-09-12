"""
Judging: a comparison that is not measuring position, and a judge that is
allowed to say it cannot judge.

Both are `PLAN.md` items, and both exist because the first thing that needs
them — an answer-quality metric for the byheart study — cannot be built out of
what was here. Every number in that study is recall of evidence; nothing
scores the answer.

The two failures they defend against are different in kind. A single-ordering
pairwise comparison measures the judge's documented preference for whatever it
read first as much as it measures quality, and the resulting number looks
exactly like a quality score. A judge whose answer does not parse, recorded as
a clean verdict, is worse: it moves an arm's mean without anybody choosing
that it should.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.eval import Report, abstain, pairwise                # noqa: E402
from yait_aichain.eval._records import Case, Record                    # noqa: E402


class _ScriptedJudge:
    """Answers each comparison from a script, and remembers what it was
    shown — which is how the ordering is checked at all."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts) - 1,
                                len(self.replies) - 1)]


class _FakeSkill:
    """Stands in for `Skill` at the seam `pairwise` actually uses.

    It imports `Skill` and `Model` inside the function body, so the patch has
    to land on the source modules rather than on `_scorers` — worth stating,
    because the first version of this harness patched the wrong module and
    every test failed on the harness instead of on the code.
    """

    judge = None
    last_usage = None

    def __init__(self, model, spec):
        self._prompt = spec["messages"][0]["parts"][0]

    def run(self):
        return type(self).judge(self._prompt)


def _scorer(judge, **kw):
    import yait_aichain.models as models_pkg
    import yait_aichain.skills as skills_pkg

    class _FakeModel:
        """A class, not a lambda: `pairwise` does `isinstance(model, Model)`
        before constructing, so the stand-in has to be a type."""
        def __init__(self, name):
            self.name = name

    _FakeSkill.judge = staticmethod(judge)
    real_skill, real_model = skills_pkg.Skill, models_pkg.Model
    skills_pkg.Skill = _FakeSkill
    models_pkg.Model = _FakeModel
    scorer = pairwise("fake-model", **kw)

    def _restore_after(case, output):
        try:
            return scorer(case, output)
        finally:
            skills_pkg.Skill, models_pkg.Model = real_skill, real_model
    return _restore_after


def _case():
    return Case(id="c1", input="What is the capital?",
                expect="Paris, the capital of France.")


class TestThePairIsPutBothWays(unittest.TestCase):

    def test_the_candidate_appears_first_then_second(self):
        """The whole debias. Asking once and calling it a comparison measures
        the judge's first-position preference alongside the answer."""
        judge = _ScriptedJudge("FIRST", "SECOND")
        _scorer(judge)(_case(), "Paris.")
        self.assertEqual(len(judge.prompts), 2)
        first, second = judge.prompts
        self.assertLess(first.index("Paris."), first.index("Paris, the"))
        self.assertLess(second.index("Paris, the"), second.index("Paris."))

    def test_winning_both_orderings_wins(self):
        verdict = _scorer(_ScriptedJudge("FIRST", "SECOND"))(_case(), "Paris.")
        self.assertTrue(verdict["ok"])
        self.assertTrue(verdict["orderings_agreed"])

    def test_losing_both_loses(self):
        verdict = _scorer(_ScriptedJudge("SECOND", "FIRST"))(_case(), "Berlin.")
        self.assertFalse(verdict["ok"])
        self.assertTrue(verdict["orderings_agreed"])

    def test_a_disagreement_goes_to_the_champion(self):
        """The disagreement *is* the position bias showing itself. Resolving
        it to the champion puts the burden on the challenger, which is what
        stops a best-so-far drifting on noise."""
        for script in (("FIRST", "FIRST"), ("SECOND", "SECOND")):
            with self.subTest(script=script):
                verdict = _scorer(_ScriptedJudge(*script))(_case(), "Paris.")
                self.assertFalse(verdict["ok"])
                self.assertFalse(verdict["orderings_agreed"])

    def test_the_disagreement_is_reported_not_just_used(self):
        """How often the two orderings disagree is the judge's own
        reliability, measured for free while grading. A pair that disagrees
        half the time is a coin toss wearing a rubric."""
        verdict = _scorer(_ScriptedJudge("FIRST", "FIRST"))(_case(), "Paris.")
        self.assertIn("orderings_agreed", verdict)

    def test_an_unreadable_comparison_abstains(self):
        verdict = _scorer(_ScriptedJudge("I cannot decide"))(_case(), "Paris.")
        self.assertTrue(verdict["abstained"])
        self.assertFalse(verdict["ok"])

    def test_the_judge_is_billed_apart(self):
        """A cost table must not quietly include the cost of grading."""
        verdict = _scorer(_ScriptedJudge("FIRST", "SECOND"))(_case(), "Paris.")
        self.assertIn("judge_tokens", verdict)


class TestAnAbstentionIsNotAFailure(unittest.TestCase):
    """`ok=False` so nothing that ignores the flag can read it as a pass, and
    excluded from the denominator so nothing reads it as a wrong answer.
    Those are different claims."""

    def _report(self, *flags):
        return Report([
            Record(arm="a", case=f"c{i}", trial=0, ok=not silent,
                   score=0.0 if silent else 1.0, output="x",
                   meta={"abstained": True} if silent else {})
            for i, silent in enumerate(flags)])

    def test_it_leaves_the_denominator(self):
        """Three scored, all passed, one unreadable: 1.0 of what was scored,
        not 0.75 of what was attempted."""
        self.assertEqual(self._report(False, False, False, True).mean("a"),
                         1.0)

    def test_it_is_counted_and_shown(self):
        report = self._report(False, False, True)
        self.assertEqual(report.abstentions("a"), 1)
        self.assertIn("n/j", report.table())

    def test_an_arm_the_judge_mostly_could_not_read_is_rejected(self):
        """Excluding rows keeps the survivors honest; past a share there are
        no survivors worth printing, and 0.95 of the fifth it managed to read
        is not a result."""
        report = self._report(False, True, True, True)
        self.assertIn("a", report.reject())
        self.assertIn("unscored", report.reject()["a"])

    def test_a_few_do_not_reject_the_arm(self):
        report = self._report(*([False] * 9 + [True]))
        self.assertEqual(report.reject(), {})
        self.assertEqual(report.abstentions("a"), 1)

    def test_the_helper_never_looks_like_a_pass(self):
        self.assertFalse(abstain("unreadable")["ok"])
        self.assertEqual(abstain("unreadable")["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
