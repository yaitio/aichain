"""
Usage accounting + retry fixes (1.3.4 #49):
  - Skill.last_usage is reset on each run() (no stale value after a failure).
  - NetworkError (status 0) is retried within a model when max_retries > 0.
  - chain.last_usage includes an Agent step's tokens (AgentResult.tokens_used).
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from skills import Skill
from chain import Chain
from agent import Agent
from models import Model
from tools._base import Tool
from clients._errors import NetworkError, AuthenticationError


def _resp(content, tokens=0):
    return json.dumps({
        "choices": [{"message": {"content": content}}],
        "usage": {"input_tokens": tokens, "output_tokens": 0},
    }).encode()


def _text_skill(model, **kw):
    return Skill(model=model,
                 input={"messages": [{"role": "user", "parts": ["hi"]}]}, **kw)


class TestLastUsageReset(unittest.TestCase):

    def test_last_usage_is_none_after_failure(self):
        m = Model("gpt-4o", api_key="k")
        m.client._auth_headers = MagicMock(return_value={})
        m.client._post = MagicMock(return_value=_resp("ok", 42))
        skill = _text_skill(m)
        skill.run()
        self.assertIsNotNone(skill.last_usage)             # set on success

        # now a non-transient failure must not leave the stale value
        m.client._post = MagicMock(side_effect=AuthenticationError(401, "bad key"))
        with self.assertRaises(AuthenticationError):
            skill.run()
        self.assertIsNone(skill.last_usage)


class TestNetworkErrorRetried(unittest.TestCase):

    def test_network_error_is_retried(self):
        m = Model("gpt-4o", api_key="k")
        m.client._auth_headers = MagicMock(return_value={})
        calls = [0]
        def post(*a, **k):
            calls[0] += 1
            if calls[0] == 1:
                raise NetworkError(0, "connection refused")
            return _resp("recovered", 5)
        m.client._post = MagicMock(side_effect=post)
        skill = _text_skill(m, max_retries=2, retry_delay=0)
        self.assertEqual(skill.run(), "recovered")
        self.assertEqual(calls[0], 2)                      # retried once after NetworkError


class TestChainIncludesAgentTokens(unittest.TestCase):

    def test_agent_step_tokens_in_chain_usage(self):
        class Noop(Tool):
            name = "noop"
            parameters = {"type": "object", "properties": {}, "required": []}
            def run(self, **kw): return "ok"

        m = Model("gpt-4o", api_key="k")
        m.client._auth_headers = MagicMock(return_value={})
        seq = [
            _resp(json.dumps({"steps": [{"id": 1, "type": "tool", "tool_name": "noop", "goal": "g"}]}), 100),
            _resp(json.dumps({"type": "tool", "tool_name": "noop", "kwargs": {}}), 50),
            _resp(json.dumps({"decision": "final_answer", "final_answer": "done"}), 30),
        ]
        call = [0]
        def post(*a, **k):
            i = min(call[0], len(seq) - 1); call[0] += 1; return seq[i]
        m.client._post = MagicMock(side_effect=post)

        agent = Agent(orchestrator=m, tools=[Noop()])
        chain = Chain(steps=[(agent, "result")])
        chain.run(variables={"task": "do it"})

        self.assertIsNotNone(chain.last_usage)
        self.assertEqual(chain.last_usage.total_tokens, 180)   # 100 + 50 + 30


if __name__ == "__main__":
    unittest.main()
