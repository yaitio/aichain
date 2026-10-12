"""
The synthetic `pool` tool is no way around the agent's rules.

It was: the permission gate looked the call up by its own name, `pool` was in
no tool map, so the gate passed it, and the fan-out then ran the named tool
directly — no policy, no approver, no argument check. A model told by a web
page to call `pool(tool="local_run", items=["rm -rf ."])` was obeyed. Its
`prompt=` form spent outside `max_cost` and outside the result's cost, and
the model read the fan-out's repr, `PoolResult(ok, items=3)`, instead of
the results. And `execute()` — the seam a serverless driver runs tools
through — applied no policy at all.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from test_loop import _ANSWER, _calls, _model, _text  # noqa: E402

from yait_aichain import Agent, PermissionPolicy  # noqa: E402
from yait_aichain.tools._base import Tool  # noqa: E402


class Danger(Tool):
    name        = "danger"
    description = "Do something irreversible."
    parameters  = {"type": "object", "properties": {"target": {"type": "string"}},
                   "required": ["target"]}
    risk        = "destructive"

    def __init__(self):
        self.ran = []

    def run(self, target, options=None):
        self.ran.append(target)
        return {"done": target}


def _tool_turn(m):
    return [x for x in m.sent[-1]["messages"] if x.get("role") == "tool"][-1]["content"]


class TestThePolicyDecidesOnWhatThePoolRuns(unittest.TestCase):

    def test_a_denied_tool_does_not_run_through_pool(self):
        tool = Danger()
        m = _model(_calls(("p", "pool", {"tool": "danger", "items": ["a", "b"]})), _ANSWER)
        Agent(m, tools=[tool], permissions=PermissionPolicy({"destructive": "deny"})).run("go")
        self.assertEqual(tool.ran, [])
        self.assertIn("denied by policy", _tool_turn(m))

    def test_an_approval_is_asked_once_with_the_whole_list(self):
        tool, asked = Danger(), []
        m = _model(_calls(("p", "pool", {"tool": "danger", "items": ["a", "b"]})), _ANSWER)
        Agent(m, tools=[tool], permissions=PermissionPolicy({"destructive": "approve"}),
              approve=lambda req: asked.append(req) or True).run("go")
        self.assertEqual(len(asked), 1)
        self.assertEqual((asked[0].tool, asked[0].arguments["items"]), ("danger", ["a", "b"]))
        self.assertEqual(sorted(tool.ran), ["a", "b"])

    def test_every_item_is_checked_before_any_runs(self):
        tool = Danger()
        m = _model(_calls(("p", "pool", {"tool": "danger", "items": [{"target": "a"}, {"nope": 1}]})),
                   _ANSWER)
        Agent(m, tools=[tool]).run("go")
        self.assertEqual(tool.ran, [])
        self.assertIn("pool item 1", _tool_turn(m))


class TestWhatThePoolReturnsAndCosts(unittest.TestCase):

    def test_the_model_reads_the_results(self):
        tool = Danger()
        m = _model(_calls(("p", "pool", {"tool": "danger", "items": ["a", "b"]})), _ANSWER)
        Agent(m, tools=[tool]).run("go")
        self.assertEqual(json.loads(_tool_turn(m)), [{"done": "a"}, {"done": "b"}])

    def test_the_prompt_form_is_spent_by_the_run(self):
        m = _model(_calls(("p", "pool", {"prompt": "Summarise {item}", "items": ["a", "b", "c"]}),
                          prompt_tokens=100, completion_tokens=20),
                   _text("summary", prompt_tokens=1000, completion_tokens=100),
                   _text("summary", prompt_tokens=1000, completion_tokens=100),
                   _text("summary", prompt_tokens=1000, completion_tokens=100),
                   _ANSWER)
        result = Agent(m, tools=[Danger()]).run("go")
        # three fan-out calls of 1,100 tokens each sit inside the total
        self.assertGreaterEqual(result.tokens_used, 3 * 1100)

    def test_a_list_too_long_is_refused(self):
        m = _model(_calls(("p", "pool", {"prompt": "x {item}", "items": list(range(500))})), _ANSWER)
        Agent(m, tools=[Danger()]).run("go")
        self.assertIn("at most 200 items", _tool_turn(m))


class TestExecuteIsGovernedToo(unittest.TestCase):

    def _agent(self, rule, approve=None):
        tool = Danger()
        agent = Agent(_model(_ANSWER), tools=[tool],
                      permissions=PermissionPolicy({"destructive": rule}), approve=approve)
        return agent, tool

    def _call(self):
        from yait_aichain.models import ToolCall
        return ToolCall("c", "danger", {"target": "prod"})

    def test_a_denied_call_is_an_error_and_does_not_run(self):
        agent, tool = self._agent("deny")
        result, error = agent.execute(self._call())
        self.assertEqual((result, tool.ran), (None, []))
        self.assertIn("denied by policy", error)

    def test_approve_with_nobody_to_ask_refuses(self):
        agent, tool = self._agent("approve")
        _, error = agent.execute(self._call())
        self.assertEqual(tool.ran, [])
        self.assertIn("no approver", error)

    def test_an_approver_saying_yes_lets_it_run(self):
        agent, tool = self._agent("approve", approve=lambda req: True)
        self.assertEqual(agent.execute(self._call()), ({"done": "prod"}, None))


if __name__ == "__main__":
    unittest.main()
