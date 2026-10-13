"""
A secret a chain carries reaches the tool that needs it and nothing else.

REST recommends ``token_field`` from a chain variable; every pause wrote that
token to the store, ``SuspendedResult.document`` and each history record, and
an Agent step put it in the GIVEN block sent to the model's provider.
"""

import json
import os
import pickle
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent
from yait_aichain.chain import Chain
from yait_aichain.models import Model
from yait_aichain.state import InMemoryStore, Secret, SuspendedResult
from yait_aichain.tools import Wait
from yait_aichain.tools._base import Tool


class UsesToken(Tool):
    name = "uses_token"
    parameters = {"type": "object", "properties": {"token": {"type": "string"}}}

    def __init__(self):
        self.seen = None

    def run(self, token=None):
        self.seen = token.reveal() if isinstance(token, Secret) else token
        return "called"


class TestSecret(unittest.TestCase):

    def test_it_never_prints(self):
        s = Secret("sk-123")
        self.assertNotIn("sk-123", repr(s) + str(s) + json.dumps({"t": s}, default=str))
        with self.assertRaises(TypeError):
            pickle.dumps(s)

    def test_a_pause_does_not_write_it_down(self):
        store, tool = InMemoryStore(), UsesToken()
        chain = Chain(steps=[(tool, "a"), Wait(reason="hold")], store=store)
        res = chain.run(variables={"token": Secret("sk-123")})
        self.assertIsInstance(res, SuspendedResult)
        self.assertEqual(tool.seen, "sk-123")
        self.assertNotIn("sk-123", json.dumps(store.load(res.run_id), default=str))
        self.assertNotIn("sk-123", json.dumps(chain.history, default=str))

    def test_an_agent_is_not_given_it(self):
        agent = Agent(Model("gpt-4o", api_key="k"))
        opening = agent.opening("do it", {"token": Secret("sk-123"), "city": "Oslo"})
        text = json.dumps(opening)
        self.assertNotIn("sk-123", text)
        self.assertNotIn("token", text)
        self.assertIn("Oslo", text)


if __name__ == "__main__":
    unittest.main()
