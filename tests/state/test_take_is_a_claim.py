"""
Resuming a parked run claims it, once.

The loop was load → run the tail → delete at the very end, which is not
atomic: two deliveries of one signal — an SQS message seen twice, a webhook
retried — both loaded the run and both refunded (`[100, 100]`). A resume now
takes the document in one step, so the second delivery finds nothing; and a
run that fails after the approved action is kept with the action's result,
so a retry continues after it instead of doing it again.
"""

import json
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import Agent, Model
from yait_aichain.state import FileStore, InMemoryStore
from yait_aichain.tools import Gate, tool


class TestTakeIsAtomic(unittest.TestCase):

    def test_of_many_takers_exactly_one_gets_it(self):
        for store in (InMemoryStore(), FileStore(tempfile.mkdtemp())):
            with self.subTest(type(store).__name__):
                store.save("r", {"run_id": "r"})
                got, start = [], threading.Barrier(8)

                def take():
                    start.wait()
                    got.append(store.take("r"))
                threads = [threading.Thread(target=take) for _ in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.assertEqual(sum(1 for g in got if g), 1)
                self.assertIsNone(store.load("r"))


refunds: list = []


@tool(risk="financial")
def refund(amount: int) -> str:
    """Refund an amount."""
    refunds.append(amount)
    return "refunded"


def _reply(*specs, text=None):
    msg = ({"content": None, "tool_calls": [
               {"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
               for i, n, a in specs]} if specs else {"content": text})
    return json.dumps({"choices": [{"message": msg}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()


def _agent(*answers, store=None):
    m = Model("gpt-4o", api_key="k")
    it = iter(answers)

    def post(*a, **k):
        nxt = next(it)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt
    m.client._post = MagicMock(side_effect=post)
    m.client._auth_headers = MagicMock(return_value={})
    return Agent(m, tools=[Gate(refund)], store=store or InMemoryStore())


class TestAnApprovedActionRunsOnce(unittest.TestCase):

    def setUp(self):
        refunds.clear()

    def test_a_second_delivery_does_nothing(self):
        agent = _agent(_reply(("c1", "gate_refund", {"amount": 100})), _reply(text="done"))
        parked = agent.run("refund 100")
        agent.resume(parked.run_id, signal={"approved": True})
        with self.assertRaises(KeyError):
            agent.resume(parked.run_id, signal={"approved": True})
        self.assertEqual(refunds, [100])

    def test_a_failure_after_the_action_is_retried_without_repeating_it(self):
        from yait_aichain import AuthenticationError        # not retried inside the call
        agent = _agent(_reply(("c1", "gate_refund", {"amount": 100})),
                       AuthenticationError(401, "key rotated"), _reply(text="done"))
        parked = agent.run("refund 100")
        first = agent.resume(parked.run_id, signal={"approved": True})
        self.assertEqual((first.success, first.stopped_by), (False, "llm_error"))
        self.assertIsNotNone(agent.store.load(parked.run_id))       # kept for the retry
        again = agent.resume(parked.run_id)
        self.assertTrue(again.success)
        self.assertEqual(refunds, [100])
        self.assertIsNone(agent.store.load(parked.run_id))


if __name__ == "__main__":
    unittest.main()
