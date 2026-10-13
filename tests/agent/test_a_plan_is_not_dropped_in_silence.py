"""
An answer that leaves planned goals untouched is held once.

A waterfall run that wrote three goals, made one call and answered ended as a
clean success; "plan not completed" was recorded only when a ceiling stopped
the run. Found reconciling the τ²-bench defects, 2026-10-14.

Nothing marks a goal done, so the fact read is the count: calls made since the
plan, a pool counted by its items. Fewer than the goals, and the first answer
is held with the goals listed — once, like `verified()`. The model may still
answer; it does so knowing what it left.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent, step_count
from yait_aichain.models import Model
from yait_aichain.tools._base import Tool

USAGE = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}


def _text(content):
    return json.dumps({"choices": [{"message": {"content": content}}], "usage": USAGE}).encode()


def _calls(*specs):
    return json.dumps({"choices": [{"message": {"content": None, "tool_calls": [
        {"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
        for i, n, a in specs]}}], "usage": USAGE}).encode()


def _model(*responses):
    m = Model("gpt-4o", api_key="k")
    m.sent = []
    turn = [0]

    def side(path, body, headers, *a, **k):
        m.sent.append(body)
        reply = responses[min(turn[0], len(responses) - 1)]
        turn[0] += 1
        return reply

    m.client._post = MagicMock(side_effect=side)
    m.client._auth_headers = MagicMock(return_value={})
    return m


class Note(Tool):
    name = "note"
    description = "Write a note."
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    risk = "read"

    def run(self, text, options=None):
        return {"written": text}


PLAN = ("p", "write_plan", {"items": ["read the file", "fix the bug", "run the tests"]})


def _agent(*responses):
    return Agent(model=_model(*responses), tools=[Note()], mode="waterfall",
                 stop_when=[step_count(10)])


def _held(model):
    return [b for b in model.sent if "Before you answer" in json.dumps(b)]


class TestAPlanWithUntouchedGoals(unittest.TestCase):

    def test_the_first_answer_is_held_with_the_goals_named(self):
        agent = _agent(_calls(PLAN), _calls(("c1", "note", {"text": "read"})),
                       _text("done"), _text("done, tests not run: no runner"))
        result = agent.run("fix it")
        held = _held(agent.model)
        self.assertEqual(len(held), 1)
        said = json.dumps(held[0])
        self.assertIn("3 goal(s)", said)
        self.assertIn("1 call(s)", said)
        self.assertIn("run the tests", said)
        # Held once, not refused: the second answer ends the run.
        self.assertTrue(result.success)
        self.assertEqual(result.output, "done, tests not run: no runner")

    def test_the_hold_is_announced(self):
        agent = _agent(_calls(PLAN), _text("done"), _text("done"))
        seen = []
        agent.hooks.append(seen.append)
        agent.run("fix it")
        fired = [e for e in seen if e.type == "nudge.fired"]
        self.assertEqual([e.payload["name"] for e in fired], ["plan"])


class TestNothingToHold(unittest.TestCase):

    def test_as_many_calls_as_goals(self):
        agent = _agent(_calls(PLAN),
                       _calls(("a", "note", {"text": "1"}), ("b", "note", {"text": "2"})),
                       _calls(("c", "note", {"text": "3"})),
                       _text("done"))
        self.assertTrue(agent.run("fix it").success)
        self.assertEqual(_held(agent.model), [])

    def test_a_pool_counts_its_items(self):
        pool = ("q", "pool", {"tool": "note", "items": ["1", "2", "3"]})
        agent = _agent(_calls(PLAN), _calls(pool), _text("done"))
        self.assertTrue(agent.run("fix it").success)
        self.assertEqual(_held(agent.model), [])

    def test_no_plan_no_hold(self):
        agent = Agent(model=_model(_text("done")), tools=[Note()])
        self.assertTrue(agent.run("hi").success)
        self.assertEqual(_held(agent.model), [])


if __name__ == "__main__":
    unittest.main()
