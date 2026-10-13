"""
`verified()`: an answer after an unchecked change is held once for a check.

The commonest agent failure: "done" with nothing run. Only a call clears the
gate; the model saying it checked does not.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent, step_count, verified                 # noqa: E402
from yait_aichain.tools._base import Tool                                  # noqa: E402
from test_loop import _calls, _model, _text                                # noqa: E402


class Write(Tool):
    name, risk = "write", "write"
    parameters = {"type": "object", "properties": {"x": {"type": "string"}}}

    def run(self, x=""):
        return "written"


class Read(Tool):
    name, risk = "read", "read"
    parameters = {"type": "object", "properties": {"x": {"type": "string"}}}

    def run(self, x=""):
        return "contents"


def _agent(*script):
    events = []
    agent = Agent(_model(*script), tools=[Write(), Read()], verbose=0, hooks=[events.append],
                  stop_when=[step_count(10), verified()])
    return agent, events


class TestVerified(unittest.TestCase):

    def test_an_unchecked_change_is_held_once(self):
        agent, events = _agent(_calls(("c1", "write", {"x": "a"})), _text("done"), _text("done, really"))
        result = agent.run("change it")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "done, really")
        self.assertEqual([e.payload["name"] for e in events if e.type == "nudge.fired"], ["verified"])

    def test_a_look_after_the_change_clears_it(self):
        agent, events = _agent(_calls(("c1", "write", {"x": "a"})), _calls(("c2", "read", {"x": "a"})),
                               _text("done"))
        self.assertEqual(agent.run("change it").output, "done")
        self.assertFalse([e for e in events if e.type == "nudge.fired"])

    def test_no_change_no_gate(self):
        agent, events = _agent(_calls(("c1", "read", {"x": "a"})), _text("done"))
        self.assertEqual(agent.run("look").output, "done")
        self.assertFalse([e for e in events if e.type == "nudge.fired"])


if __name__ == "__main__":
    unittest.main()
