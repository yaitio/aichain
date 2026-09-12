"""
"What happens when a step fails" is one decision with one vocabulary.

`Chain` took `on_step_error` in {raise, stop, skip}; `Pool` took `on_error` in
{raise, collect, skip}. Two names, three values each, and only `raise` meant
the same thing on both — so a caller who learned one primitive guessed wrong
at the other. `skip` was the trap: in a chain it meant "carry on, loudly", and
in a pool the silent option was called `collect` while `skip` added a warning.

Four words now, one meaning each, on both. What each primitive does with
`stop` differs in the obvious way — a chain stops running steps, a pool stops
starting items — and that is the difference between a sequence and a fan-out,
not between two vocabularies.

And a failure is recorded as more than `str(exc)`. That was the least useful
part of it: `KeyError('tenant')` renders as `'tenant'`, which in a run record
reads as a value rather than a fault, and a `TimeoutError` with no message
renders as nothing at all.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Chain, Pool                                # noqa: E402
from yait_aichain._errors_policy import POLICIES, describe          # noqa: E402
from yait_aichain.tools._base import Tool                           # noqa: E402


class Boom(Tool):
    name = "boom"
    parameters = {"type": "object", "properties": {"input": {"type": "string"}}}

    def run(self, input=None, options=None):
        raise KeyError("tenant")


class Fine(Tool):
    name = "fine"
    parameters = {"type": "object", "properties": {"input": {"type": "string"}}}

    def run(self, input=None, options=None):
        return "ok"


class TestBothSpeakTheSameWords(unittest.TestCase):

    def test_the_vocabularies_are_identical(self):
        from yait_aichain.chain._chain import _VALID_ON_STEP_ERROR
        from yait_aichain.pool._pool import _VALID_ON_ERROR
        self.assertEqual(_VALID_ON_STEP_ERROR, _VALID_ON_ERROR, POLICIES)

    def test_every_word_is_accepted_by_both(self):
        for policy in sorted(POLICIES):
            with self.subTest(policy=policy):
                Chain(steps=[(Fine(), "a")], on_step_error=policy)
                Pool(runner=Fine(), items=[{}], on_error=policy)

    def test_a_wrong_word_names_the_whole_set(self):
        for build in (lambda: Chain(steps=[(Fine(), "a")],
                                    on_step_error="collectt"),
                      lambda: Pool(runner=Fine(), items=[{}],
                                   on_error="collectt")):
            with self.subTest():
                with self.assertRaises(ValueError) as caught:
                    build()
                for word in POLICIES:
                    self.assertIn(word, str(caught.exception))


class TestTheNewWordsWork(unittest.TestCase):

    def test_a_chain_can_carry_on_silently(self):
        """`collect` is new to Chain: a pipeline of generated steps that
        expects a few to fail should not have to choose between a warning per
        failure and no record at all."""
        chain = Chain(steps=[(Boom(), "a"), (Fine(), "b")],
                      on_step_error="collect")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chain.run(variables={"input": "x"})
        self.assertEqual([w for w in caught if "skipped" in str(w.message)], [])

    def test_and_still_records_what_failed(self):
        chain = Chain(steps=[(Boom(), "a"), (Fine(), "b")],
                      on_step_error="collect")
        chain.run(variables={"input": "x"})
        failed = [h for h in chain.history if h.get("error")]
        self.assertTrue(failed)
        self.assertEqual(failed[0]["failure"]["type"], "KeyError")

    def test_skip_still_warns(self):
        chain = Chain(steps=[(Boom(), "a"), (Fine(), "b")],
                      on_step_error="skip")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chain.run(variables={"input": "x"})
        self.assertTrue([w for w in caught if "skipped" in str(w.message)])

    def test_a_pool_can_stop(self):
        """`stop` is new to Pool. It starts no more items; the ones already
        in flight are let finish, because killing a call mid-request costs
        the tokens anyway and loses the answer."""
        pool = Pool(runner=Boom(), items=[{"input": "x"} for _ in range(4)],
                    on_error="stop", max_flows=1)
        pool.run()
        self.assertTrue(any(h["status"] == 3 for h in pool.history))


class TestAFailureIsMoreThanItsMessage(unittest.TestCase):

    def test_the_type_is_kept(self):
        """`KeyError('tenant')` renders as `'tenant'` — a value, not a
        fault."""
        try:
            raise KeyError("tenant")
        except KeyError as exc:
            record = describe(exc)
        self.assertEqual(record["type"], "KeyError")
        self.assertEqual(record["message"], "'tenant'")

    def test_the_traceback_is_kept(self):
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            record = describe(exc)
        self.assertIn("RuntimeError: boom", record["traceback"])

    def test_an_empty_message_still_says_something(self):
        """A `TimeoutError()` renders as the empty string, so the message
        alone tells a reader nothing at all."""
        try:
            raise TimeoutError()
        except TimeoutError as exc:
            record = describe(exc)
        self.assertEqual(record["message"], "")
        self.assertEqual(record["type"], "TimeoutError")

    def test_a_pool_records_it_too(self):
        pool = Pool(runner=Boom(), items=[{"input": "x"}], on_error="collect")
        pool.run()
        self.assertEqual(pool.history[0]["failure"]["type"], "KeyError")


if __name__ == "__main__":
    unittest.main()
