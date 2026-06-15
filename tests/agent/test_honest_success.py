"""
Agent honest-success + final-output selection (1.3.4 #47):
  H3 — an execution error on an EARLIER step (orchestrator chose "continue")
       fails the run, not only an error on the last step.
  M1 — the final output is the last executed step's output, even if None —
       not a stale earlier output.
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


def _orchestrator(*responses):
    m = Model("gpt-4o", api_key="k")
    call = [0]
    def side(*a, **k):
        i = min(call[0], len(responses) - 1); call[0] += 1; return responses[i]
    m.client._post = MagicMock(side_effect=side)
    m.client._auth_headers = MagicMock(return_value={})
    return m


class Boom(Tool):
    name = "boom"
    parameters = {"type": "object", "properties": {}, "required": []}
    def run(self, **kw):
        raise RuntimeError("tool exploded")


class Ok(Tool):
    name = "ok"
    parameters = {"type": "object", "properties": {}, "required": []}
    def run(self, **kw):
        return "ok"


class Nothing(Tool):
    name = "nothing"
    parameters = {"type": "object", "properties": {}, "required": []}
    def run(self, **kw):
        return None


_TWO_STEP = lambda a, b: {"steps": [
    {"id": 1, "type": "tool", "tool_name": a, "goal": "g1"},
    {"id": 2, "type": "tool", "tool_name": b, "goal": "g2"},
]}
_CONT = {"decision": "continue", "store_as": "x"}


class TestHonestSuccess(unittest.TestCase):

    def test_earlier_step_error_fails_the_run(self):
        orch = _orchestrator(
            _json(_TWO_STEP("boom", "ok")),
            _json({"type": "tool", "tool_name": "boom", "kwargs": {}}), _json(_CONT),
            _json({"type": "tool", "tool_name": "ok", "kwargs": {}}),   _json(_CONT),
        )
        agent = Agent(orchestrator=orch, tools=[Boom(), Ok()])
        res = agent.run("do it")
        self.assertFalse(res.success)          # earlier step failed → not success
        self.assertIn("execution error", res.error)


class TestFinalOutputSelection(unittest.TestCase):

    def test_final_none_not_stale_earlier_output(self):
        orch = _orchestrator(
            _json(_TWO_STEP("ok", "nothing")),
            _json({"type": "tool", "tool_name": "ok", "kwargs": {}}),      _json(_CONT),
            _json({"type": "tool", "tool_name": "nothing", "kwargs": {}}), _json(_CONT),
        )
        agent = Agent(orchestrator=orch, tools=[Ok(), Nothing()])
        res = agent.run("do it")
        self.assertTrue(res.success)
        self.assertIsNone(res.output)          # final step's output, not "ok"


if __name__ == "__main__":
    unittest.main()
