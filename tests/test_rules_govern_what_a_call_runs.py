"""
The rules around a call govern everything it runs.

Phase 3 of PLAN-agent-chain.md. A Chain step and a Pool item ran their tools
outside the dispatcher, only charged to the budget: a tool an agent's policy
refused ran when a pipeline inside one of the agent's tools called it, and
neither a Chain nor a Pool could be given rules of its own. Now both take
`policy=` and `approve=`, their tools go through `_dispatch.py`, and while a
call executes its owner's rules govern what it starts.

Pinned: the stricter word wins in both directions, a chain with no approver
asks the one around it, the dict form of `policy=` is checked, a refused step
is handled like any failed step, and `save()` says what it cannot keep.
"""

import json
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent, step_count
from yait_aichain.chain import Chain
from yait_aichain.models import Model
from yait_aichain.pool import Pool
from yait_aichain.tools import PermissionPolicy, tool
from yait_aichain.tools._base import Tool

USAGE = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


class Nuke(Tool):
    name = "nuke"
    description = "Delete everything."
    parameters = {"type": "object", "properties": {"target": {"type": "string"}}}
    risk = "destructive"

    def __init__(self):
        self.ran = 0

    def run(self, target="all", options=None):
        self.ran += 1
        return {"deleted": target}


def _agent(tools, **kw):
    m = Model("gpt-4o", api_key="k")
    replies = [{"choices": [{"message": {"content": None, "tool_calls": [
                   {"id": "c", "type": "function",
                    "function": {"name": tools[0].name, "arguments": "{}"}}]}}], "usage": USAGE},
               {"choices": [{"message": {"content": "done"}}], "usage": USAGE}]
    turn = [0]

    def side(*a, **k):
        reply = replies[min(turn[0], 1)]
        turn[0] += 1
        return json.dumps(reply).encode()

    m.client._post = MagicMock(side_effect=side)
    m.client._auth_headers = MagicMock(return_value={})
    return Agent(m, tools=tools, stop_when=[step_count(4)], **kw)


def _tidy(chain_kw=None):
    nuke = Nuke()

    @tool(risk="read")
    def tidy() -> str:
        """Tidy up."""
        Chain([(nuke, "out")], **(chain_kw or {})).run()
        return "tidied"

    return nuke, tidy


class TestTheStricterWordWins(unittest.TestCase):

    def test_an_agents_refusal_reaches_a_chain_that_allows(self):
        nuke, tidy = _tidy({"policy": {"destructive": "allow"}})
        _agent([tidy], permissions=PermissionPolicy({"destructive": "deny"})).run("go")
        self.assertEqual(nuke.ran, 0)

    def test_a_chains_refusal_holds_inside_an_agent_that_allows(self):
        nuke, tidy = _tidy({"policy": {"destructive": "deny"}})
        _agent([tidy], permissions=PermissionPolicy({"destructive": "allow"})).run("go")
        self.assertEqual(nuke.ran, 0)

    def test_both_allowing_runs_it(self):
        nuke, tidy = _tidy({"policy": {"destructive": "allow"}})
        _agent([tidy], permissions=PermissionPolicy({"destructive": "allow"})).run("go")
        self.assertEqual(nuke.ran, 1)


class TestTheApproverAround(unittest.TestCase):

    def test_a_chain_with_no_approver_asks_the_agents(self):
        asked = []
        nuke, tidy = _tidy()

        def approve(request):
            asked.append(request.tool)
            return True

        _agent([tidy], permissions=PermissionPolicy({"destructive": "approve"}),
               approve=approve).run("go")
        self.assertEqual((asked, nuke.ran), (["nuke"], 1))

    def test_its_own_approver_comes_first(self):
        mine, theirs = [], []
        nuke, tidy = _tidy({"approve": lambda r: mine.append(r.tool) or False})
        _agent([tidy], permissions=PermissionPolicy({"destructive": "approve"}),
               approve=lambda r: theirs.append(r.tool) or True).run("go")
        self.assertEqual((mine, theirs, nuke.ran), (["nuke"], [], 0))


class TestAChainAndAPoolTakeRules(unittest.TestCase):

    def test_the_dict_forms(self):
        nuke = Nuke()
        with self.assertRaises(PermissionError):
            Chain([(nuke, "out")], policy={"rules": {"destructive": "deny"}}).run()
        Pool(nuke, items=[{"target": "a"}], policy={"destructive": "deny"},
             on_error="collect").run()
        self.assertEqual(nuke.ran, 0)

    def test_a_wrong_key_is_refused_when_built(self):
        with self.assertRaises(ValueError):
            Chain([(Nuke(), "out")], policy={"rules": {}, "deafult": "deny"})
        with self.assertRaises(ValueError):
            Pool(Nuke(), items=[{}], policy={"destructve": "deny"})

    def test_a_refused_step_is_handled_like_any_failed_step(self):
        nuke = Nuke()

        def after() -> str:
            return "went on"

        result = Chain([(nuke, "out"), (after, "next")], policy={"destructive": "deny"},
                       on_step_error="skip").run()
        self.assertEqual(nuke.ran, 0)
        self.assertEqual(result["next"], "went on")

    def test_save_keeps_the_rules_and_says_what_it_cannot_keep(self):
        import tempfile
        from yait_aichain.tools import Wait
        chain = Chain([(Wait(reason="ok?"), "w")], policy={"destructive": "deny"},
                      approve=lambda r: True)
        with warnings.catch_warnings(record=True) as seen, tempfile.TemporaryDirectory() as tmp:
            warnings.simplefilter("always")
            path = os.path.join(tmp, "c.yaml")
            chain.save(path)
            back = Chain.load(path, trusted=True)
        self.assertEqual(back.policy.rules["destructive"], "deny")
        self.assertTrue(any("approver and interceptors are code" in str(w.message) for w in seen))


if __name__ == "__main__":
    unittest.main()
