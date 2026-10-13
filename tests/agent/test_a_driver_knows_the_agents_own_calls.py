"""
A driver of `step()` can tell the agent's own calls from the world's.

`pool`, `write_plan` and `delegate` exist only inside this library. The
τ²-bench adapter handed every call `step()` returned to the benchmark, which
answered "tool not found" 962 times — every plan of the waterfall arm and
every fan-out of both aichain arms (counted 2026-10-14). `execute()` ran them
all along; nothing on the seam said which calls needed it. `owns()` does.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent
from yait_aichain.models import Model
from yait_aichain.models._calls import ToolCall
from yait_aichain.tools._base import Tool


class Lookup(Tool):
    name = "lookup"
    description = "Look a thing up."
    parameters = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    risk = "read"

    def run(self, q, options=None):
        return {"q": q}


def _agent(mode="agile"):
    m = Model("gpt-4o", api_key="k")
    m.client._post = MagicMock(return_value=json.dumps(
        {"choices": [{"message": {"content": "ok"}}], "usage": {}}).encode())
    m.client._auth_headers = MagicMock(return_value={})
    return Agent(model=m, tools=[Lookup()], mode=mode)


class TestOwns(unittest.TestCase):

    def test_the_library_calls_are_the_agents(self):
        agent = _agent("waterfall")
        for name in ("pool", "write_plan", "delegate"):
            with self.subTest(name=name):
                self.assertTrue(agent.owns(ToolCall(id="c", name=name, arguments={})))
                self.assertTrue(agent.owns(name))

    def test_the_given_tools_are_the_drivers_to_place(self):
        self.assertFalse(_agent().owns(ToolCall(id="c", name="lookup", arguments={"q": "x"})))

    def test_an_owned_call_runs_through_execute(self):
        agent = _agent("waterfall")
        state = agent.new_state()
        result, error = agent.execute(
            ToolCall(id="p", name="write_plan", arguments={"items": ["a", "b"]}), state)
        self.assertIsNone(error)
        self.assertEqual(state["plan"], ["a", "b"])
        self.assertIn("Plan recorded", str(result))


if __name__ == "__main__":
    unittest.main()
