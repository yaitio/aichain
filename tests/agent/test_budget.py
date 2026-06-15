"""
Agent token-budget enforcement (1.3.4 #46):
  H1 — the budget is checked WITHIN a step (no action+execute+reflect overshoot).
  H2 — agile replan is capped (no non-progressing loop).
  C1 — tokens from a failed-to-parse orchestrator reply are still counted.
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


def _raw(content_str, tokens=0):
    return json.dumps({
        "choices": [{"message": {"content": content_str}}],
        "usage": {"input_tokens": tokens, "output_tokens": 0},
    }).encode()


def _json(obj, tokens=0):
    return _raw(json.dumps(obj), tokens)


def _orchestrator(*responses):
    m = Model("gpt-4o", api_key="k")
    call = [0]
    def side(*a, **k):
        i = min(call[0], len(responses) - 1); call[0] += 1; return responses[i]
    m.client._post = MagicMock(side_effect=side)
    m.client._auth_headers = MagicMock(return_value={})
    return m


class Noop(Tool):
    name = "noop"
    parameters = {"type": "object", "properties": {}, "required": []}
    def run(self, **kw):
        return "ok"


_PLAN   = {"steps": [{"id": 1, "type": "tool", "tool_name": "noop", "goal": "g"}]}
_ACTION = {"type": "tool", "tool_name": "noop", "kwargs": {}}


class TestWithinStepBudget(unittest.TestCase):

    def test_no_overshoot_within_a_step(self):
        # action call alone exceeds the budget → execution + reflection skipped
        orch = _orchestrator(_json(_PLAN, 0), _json(_ACTION, 100))
        agent = Agent(orchestrator=orch, tools=[Noop()], max_tokens=10)
        res = agent.run("do it")
        self.assertFalse(res.success)
        # plan + action only — NO reflection call after the budget was blown
        self.assertEqual(orch.client._post.call_count, 2)


class TestReplanCap(unittest.TestCase):

    def test_replan_loop_is_capped(self):
        replan = {"decision": "replan", "goto_step": 0,
                  "revised_plan": [{"id": 1, "type": "tool", "tool_name": "noop", "goal": "g"}]}
        seq = [_json(_PLAN)] + [_json(_ACTION), _json(replan)] * 6
        agent = Agent(orchestrator=_orchestrator(*seq), tools=[Noop()],
                      mode="agile", max_steps=2, max_tokens=10_000_000)
        res = agent.run("do it")
        self.assertFalse(res.success)
        self.assertIn("Replan limit", res.error)


class TestFailedParseTokensCounted(unittest.TestCase):

    def test_unparseable_action_tokens_still_counted(self):
        orch = _orchestrator(_json(_PLAN, 0),
                             _raw("not json", 50),     # action attempt 1
                             _raw("still not", 50))     # action retry
        agent = Agent(orchestrator=orch, tools=[Noop()], max_tokens=1_000_000)
        res = agent.run("do it")
        self.assertFalse(res.success)
        self.assertGreaterEqual(res.tokens_used, 100)   # the 2 failed calls counted


if __name__ == "__main__":
    unittest.main()
