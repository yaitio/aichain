"""
Agent LLM calls go through the client.send() seam (1.3.4 #48), so async
providers (e.g. Qwen image-synthesis) and the transport path stay consistent
with Skill — not a raw _post that bypasses send().
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agent import Agent
from tools._base import Tool
from models import Model


def _json(obj):
    return json.dumps({"choices": [{"message": {"content": json.dumps(obj)}}]}).encode()


class Noop(Tool):
    name = "noop"
    parameters = {"type": "object", "properties": {}, "required": []}
    def run(self, **kw):
        return "ok"


class TestAgentUsesSendSeam(unittest.TestCase):

    def test_orchestrator_calls_go_through_send(self):
        plan   = {"steps": [{"id": 1, "type": "tool", "tool_name": "noop", "goal": "g"}]}
        action = {"type": "tool", "tool_name": "noop", "kwargs": {}}
        reflect = {"decision": "final_answer", "final_answer": "done"}
        seq = [_json(plan), _json(action), _json(reflect)]

        m = Model("gpt-4o", api_key="k")
        m.client._auth_headers = MagicMock(return_value={})
        call = [0]
        def send(*a, **k):
            i = min(call[0], len(seq) - 1); call[0] += 1; return seq[i]
        m.client.send = MagicMock(side_effect=send)
        # _post must NOT be used by the agent path
        m.client._post = MagicMock(side_effect=AssertionError("agent bypassed send()"))

        agent = Agent(orchestrator=m, tools=[Noop()])
        res = agent.run("do it")
        self.assertTrue(res.success)
        self.assertTrue(m.client.send.called)


if __name__ == "__main__":
    unittest.main()
