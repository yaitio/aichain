"""
Chain, Pool and Agent return results of one shape (3.0.0).

`.output`, `.success`, `.error`, `.history`, `.usage`, `.tokens_used`, `.cost`
on all three, and `bool(result)` is `success`. Before 3.0.0 a chain returned
its last step's output as a bare value and a pool a bare list, while an agent
returned an `AgentResult`: three primitives, three ways to ask "did it work
and what did it produce". Measured in `evals/agent_builds`, the bare chain
value was the commonest single mistake a model made — it indexed the result by
step name. A chain result now supports exactly that.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Chain, ChainResult, Pool, PoolResult          # noqa: E402
from yait_aichain.agent import AgentResult                             # noqa: E402
from yait_aichain.eval import Case, Eval                               # noqa: E402
from yait_aichain.models._usage import Usage                           # noqa: E402
from yait_aichain.tools._base import Tool                              # noqa: E402

FIELDS = ("output", "success", "error", "history", "usage", "tokens_used", "cost")


class Upper(Tool):
    name = "upper"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    def run(self, text=None, options=None):
        return text.upper()


class Boom(Tool):
    name = "boom"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    def run(self, text=None, options=None):
        raise RuntimeError("step broke")


class Priced:
    """A Skill-shaped step with a usage report."""
    name = "priced"

    def __init__(self, reply):
        self.reply, self.last_usage = reply, None

    def run(self, variables=None):
        self.last_usage = Usage(input_tokens=3, output_tokens=4, total_tokens=7, cost=0.01)
        return self.reply


class TestOneShape(unittest.TestCase):

    def test_all_three_carry_the_same_fields(self):
        results = [Chain(steps=[(Priced("a"), "a")]).run(),
                   Pool(Upper(), [{"text": "a"}]).run(),
                   AgentResult(success=True, output="a", mode="agile",
                               steps_taken=1, tokens_used=7, cost=0.01)]
        for result in results:
            for name in FIELDS:
                with self.subTest(result=type(result).__name__, field=name):
                    self.assertTrue(hasattr(result, name), f"{type(result).__name__} has no {name}")
            self.assertEqual(bool(result), result.success)


class TestChainResult(unittest.TestCase):

    def test_output_and_every_step_by_its_key(self):
        result = Chain(steps=[(Priced("draft"), "draft"), (Upper(), "shout", {"text": "draft"})]).run()
        self.assertIsInstance(result, ChainResult)
        self.assertEqual(result.output, "DRAFT")
        self.assertEqual(result["draft"], "draft")
        self.assertEqual(result["shout"], "DRAFT")
        self.assertEqual(str(result), "DRAFT")
        self.assertTrue(result.success)
        self.assertTrue(result)
        self.assertIsNone(result.error)

    def test_an_unknown_key_names_the_ones_that_exist(self):
        result = Chain(steps=[(Priced("x"), "draft")]).run()
        with self.assertRaises(KeyError) as ctx:
            result["tweet"]
        self.assertIn("draft", str(ctx.exception))

    def test_a_skipped_failure_is_not_success(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = Chain(steps=[(Priced("a"), "text"), (Boom(), "x"), (Upper(), "y")],
                           on_step_error="skip").run()
        self.assertFalse(result.success)
        self.assertFalse(result)
        self.assertEqual(result.output, "A")
        self.assertEqual(result.error, "step 1 (boom): step broke")

    def test_usage_tokens_and_cost(self):
        result = Chain(steps=[(Priced("a"), "a"), (Priced("b"), "b")]).run()
        self.assertEqual(result.tokens_used, 14)
        self.assertAlmostEqual(result.cost, 0.02)
        self.assertEqual(result.usage.input_tokens, 6)

    def test_a_chain_inside_a_chain_contributes_its_output(self):
        inner = Chain(steps=[(Priced("inner"), "draft")])
        outer = Chain(steps=[(inner, "nested"), (Upper(), "shout", {"text": "nested"})])
        self.assertEqual(outer.run().output, "INNER")

    def test_a_failed_inner_chain_fails_its_step(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inner = Chain(steps=[(Boom(), "x")], on_step_error="collect")
        with self.assertRaises(RuntimeError) as ctx:
            Chain(steps=[(inner, "nested")]).run({"text": "a"})
        self.assertIn("step broke", str(ctx.exception))


class TestPoolResult(unittest.TestCase):

    def test_a_sequence_of_outputs_with_the_shared_fields(self):
        result = Pool(Upper(), [{"text": "a"}, {"text": "b"}]).run()
        self.assertIsInstance(result, PoolResult)
        self.assertEqual(list(result), ["A", "B"])
        self.assertEqual(result[1], "B")
        self.assertEqual(len(result), 2)
        self.assertEqual(result.output, ["A", "B"])
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_a_failed_item_is_not_success(self):
        result = Pool(Boom(), [{"text": "a"}]).run()
        self.assertFalse(result)
        self.assertEqual(result.output, [None])
        self.assertEqual(result.errors, ["step broke"])
        self.assertEqual(result.error, "item 0: step broke")

    def test_a_chain_runner_contributes_its_output(self):
        chain = Chain(steps=[(Upper(), "shout")])
        self.assertEqual(Pool(chain, [{"text": "a"}, {"text": "b"}]).run().output, ["A", "B"])

    def test_a_chain_runner_that_fails_is_a_failed_item(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chain = Chain(steps=[(Boom(), "x")], on_step_error="collect")
        result = Pool(chain, [{"text": "a"}]).run()
        self.assertFalse(result)
        self.assertIn("step broke", result.error)


class TestEvalRecordsTheOutputNotTheObject(unittest.TestCase):

    def test_a_chain_arm(self):
        import tempfile
        chain = Chain(steps=[(Priced("yes"), "answer")])
        report = Eval([Case(id="c1", expect="yes")], {"chain": lambda case: chain.run()},
                      lambda case, out: out == case.expect,
                      out=os.path.join(tempfile.mkdtemp(), "ledger.jsonl"), verbose=False).run()
        record = report.records[0]
        self.assertEqual(record.output, "yes")
        self.assertTrue(record.ok)
        self.assertEqual(record.tokens, 7)
        self.assertTrue(record.meta["success"])


if __name__ == "__main__":
    unittest.main()
