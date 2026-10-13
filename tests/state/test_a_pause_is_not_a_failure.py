"""
A pause is neither a failure nor a finished step.

`SuspendedResult` is falsy, so a Pool told the caller "Agent failed: unknown
error" for an item whose agent had parked — and lost the run_id, leaving its
document orphaned in the store — while a Chain counted a parked nested chain
as done and handed the `SuspendedResult` on as a variable. A function or a
Skill raising `Suspend` in a chain was re-run without the signal on resume and
paused again, for ever.
"""

import json
import os
import unittest
from unittest.mock import MagicMock

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import Agent, Chain, Model, Pool
from yait_aichain.state import InMemoryStore, Suspend, SuspendedResult
from yait_aichain.tools import Wait


def _parking_agent():
    m = Model("gpt-4o", api_key="k")
    m.client._post = MagicMock(return_value=json.dumps({
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "wait", "arguments": "{}"}}]}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode())
    m.client._auth_headers = MagicMock(return_value={})
    return Agent(m, tools=[Wait(reason="approve?")], store=InMemoryStore())


class TestAPoolKeepsAParkedItem(unittest.TestCase):

    def test_the_item_is_its_suspended_result(self):
        agent = _parking_agent()
        result = Pool(agent, [{"task": "x"}]).run()
        (item,) = result.output
        self.assertIsInstance(item, SuspendedResult)
        self.assertIsNotNone(agent.store.load(item.run_id))
        self.assertEqual(result.errors, [None])


class TestAChainDoesNotCarryAChildsPause(unittest.TestCase):

    def test_a_parked_nested_chain_is_refused_by_name(self):
        inner = Chain(steps=[Wait(reason="hold")])
        with self.assertRaisesRegex(RuntimeError, "parked its own run"):
            Chain(steps=[(inner, "x")]).run()

    def test_suspend_from_a_function_step_is_refused(self):
        def pauses(**_):
            raise Suspend("not like this")
        with self.assertRaisesRegex(RuntimeError, "only a tool step can"):
            Chain(steps=[(pauses, "x")]).run()


class TestASuspendedResultReadsLikeAResult(unittest.TestCase):

    def test_the_common_fields(self):
        r = SuspendedResult(run_id="r", awaiting={"reason": "approve?"})
        self.assertEqual((r.success, r.suspended, r.output, r.cost, r.tokens_used, bool(r)),
                         (False, True, None, None, 0, False))
        self.assertIn("approve?", r.error)


if __name__ == "__main__":
    unittest.main()
