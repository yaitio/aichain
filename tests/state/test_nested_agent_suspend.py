"""
A nested Agent that suspends (Wait/Gate) inside a Chain must propagate the
suspension up — chain.run() returns a SuspendedResult (not a "failed" agent
step), and chain.resume() continues the child agent run. (1.3.3)
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agent import Agent
from chain import Chain
from tools import Wait
from models import Model
from state import InMemoryStore, SuspendedResult


def _resp(obj):
    return json.dumps({"choices": [{"message": {"content": json.dumps(obj)}}]}).encode()


def _orchestrator(*responses):
    m = Model("gpt-4o", api_key="k")
    enc = [_resp(r) for r in responses]
    call = [0]
    def side(*a, **k):
        i = min(call[0], len(enc) - 1); call[0] += 1; return enc[i]
    m.client._post = MagicMock(side_effect=side)
    m.client._auth_headers = MagicMock(return_value={})
    return m


_PLAN    = {"steps": [{"id": 1, "type": "tool", "tool_name": "wait", "goal": "await"}]}
_ACTION  = {"type": "tool", "tool_name": "wait", "kwargs": {}}
_REFLECT = {"decision": "continue", "store_as": "approval"}


class TestNestedAgentSuspend(unittest.TestCase):

    def _chain(self):
        agent = Agent(orchestrator=_orchestrator(_PLAN, _ACTION),
                      tools=[Wait(reason="approval", resume_with={"ok": "bool"})])
        chain = Chain(steps=[(agent, "result")], store=InMemoryStore())
        return agent, chain

    def test_chain_suspends_instead_of_failing(self):
        agent, chain = self._chain()
        res = chain.run(variables={"task": "do it"})
        # The chain must report suspension, NOT a failed agent step.
        self.assertIsInstance(res, SuspendedResult)
        self.assertEqual(res.document["steps"][0]["status"], "suspended")
        self.assertIn("child_run_id", res.document["steps"][0]["suspend"])

    def test_resume_continues_the_child_agent(self):
        agent, chain = self._chain()
        res = chain.run(variables={"task": "do it"})
        self.assertIsInstance(res, SuspendedResult)

        # after the Wait, the orchestrator only needs to reflect once
        agent.orchestrator.client._post = MagicMock(side_effect=[_resp(_REFLECT)])
        out = chain.resume(res.run_id, signal={"ok": True})

        self.assertNotIsInstance(out, SuspendedResult)     # the chain completed
        # the resumed agent's output (the Wait signal) flowed through the chain
        self.assertEqual(out, {"ok": True})
        # the parked chain run is cleaned up on completion
        self.assertIsNone(chain._store.load(res.run_id))


if __name__ == "__main__":
    unittest.main()
