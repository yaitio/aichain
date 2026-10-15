"""
The agent's interface, phase 7 of PLAN-agent-chain.md (4.2.9).

`policy={...}` holds everything that decides what happens to a call — rules,
the default, who answers an "approve" (a function, "park" or "deny") and the
interceptors — with its keys checked. `approve: "park"` parks the run with the
question and `resume()` brings the answer. `team={...}` says the members, and
whether the lead may add one — one role made once and reused, within `max`.
The old spellings work, and warn.
"""

import json
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import Block, Interceptor
from yait_aichain.agent import Agent, cost_budget, step_count
from yait_aichain.agent._agent import DELEGATE_WHEN
from yait_aichain.models import Model
from yait_aichain.state import InMemoryStore, SuspendedResult
from yait_aichain.tools import PermissionPolicy, tool

USAGE = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def _model(*replies):
    m = Model("gpt-4o", api_key="k")
    m.sent = []
    turn = [0]

    def side(path, body, headers, *a, **k):
        m.sent.append(body)
        reply = replies[min(turn[0], len(replies) - 1)]
        turn[0] += 1
        return json.dumps(reply).encode()

    m.client._post = MagicMock(side_effect=side)
    m.client._auth_headers = MagicMock(return_value={})
    return m


def _text(content):
    return {"choices": [{"message": {"content": content}}], "usage": USAGE}


def _call(name, arguments, call_id="c1"):
    return {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments)}}]}}], "usage": USAGE}


REFUNDED: list = []


@tool(risk="financial")
def refund(amount: int) -> str:
    """Refund an amount."""
    REFUNDED.append(amount)
    return f"refunded {amount}"


def _quiet(make):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        value = make()
    return value, [str(w.message) for w in seen]


class TestPolicy(unittest.TestCase):

    def setUp(self):
        REFUNDED.clear()

    def test_rules_and_an_approver_in_one_place(self):
        agent = Agent(_model(_call("refund", {"amount": 5}), _text("done")), tools=[refund],
                      policy={"rules": {"financial": "approve"}, "approve": lambda r: True})
        agent.run("refund 5")
        self.assertEqual(REFUNDED, [5])

    def test_deny_refuses_every_approval(self):
        model = _model(_call("refund", {"amount": 5}), _text("done"))
        Agent(model, tools=[refund],
              policy={"rules": {"financial": "approve"}, "approve": "deny"}).run("refund 5")
        self.assertEqual(REFUNDED, [])
        self.assertIn("refuses every call that needs approval", json.dumps(model.sent[-1]))

    def test_interceptors_live_in_the_policy(self):
        class No(Interceptor):
            def before(self, call, run):
                return Block("not today")

        Agent(_model(_call("refund", {"amount": 5}), _text("done")), tools=[refund],
              policy={"interceptors": [No()]}).run("refund 5")
        self.assertEqual(REFUNDED, [])

    def test_its_keys_are_checked(self):
        for bad in ({"rules": {}, "aprove": "park"}, {"approve": "later"},
                    {"interceptors": "x"}):
            with self.subTest(policy=bad), self.assertRaises(ValueError):
                Agent(_model(_text("x")), policy=bad)

    def test_the_old_spellings_warn_and_work(self):
        agent, said = _quiet(lambda: Agent(
            _model(_call("refund", {"amount": 5}), _text("done")), tools=[refund],
            permissions=PermissionPolicy({"financial": "approve"}), approve=lambda r: True))
        agent.run("refund 5")
        self.assertEqual(REFUNDED, [5])
        self.assertEqual(sum("goes in 5.0" in m for m in said), 2)

    def test_two_spellings_of_one_thing_are_refused(self):
        with self.assertRaises(ValueError):
            Agent(_model(_text("x")), policy={"approve": "deny"}, approve=lambda r: True)


class TestApprovalParks(unittest.TestCase):

    def setUp(self):
        REFUNDED.clear()

    def _agent(self, store=None):
        return Agent(_model(_call("refund", {"amount": 80}), _text("refunded"), _text("not refunded")),
                     tools=[refund], store=store or InMemoryStore(),
                     policy={"rules": {"financial": "approve"}, "approve": "park"})

    def test_the_run_parks_with_the_question_and_a_yes_runs_the_call(self):
        agent = self._agent()
        parked = agent.run("refund 80")
        self.assertIsInstance(parked, SuspendedResult)
        self.assertEqual(REFUNDED, [])
        self.assertTrue(parked.awaiting["approval"])
        self.assertEqual(parked.awaiting["tool"], "refund")
        result = agent.resume(parked.run_id, signal={"approved": True})
        self.assertEqual(REFUNDED, [80])
        self.assertTrue(result.success)

    def test_a_no_refuses_it_with_the_reason(self):
        agent = self._agent()
        parked = agent.run("refund 80")
        agent.resume(parked.run_id, signal={"approved": False, "reason": "over budget"})
        self.assertEqual(REFUNDED, [])
        self.assertIn("over budget", json.dumps(agent.model.sent[-1]))

    def test_park_needs_a_store(self):
        with self.assertRaises(ValueError):
            Agent(_model(_text("x")), policy={"approve": "park"})

    def test_a_leads_park_does_not_reach_inside_a_worker(self):
        worker = Agent(_model(_call("refund", {"amount": 5}), _text("worker done")),
                       tools=[refund], name="w", description="Refunds.")
        lead = Agent(_model(_call("delegate", {"task": "t", "worker": "w"}, "d"), _text("done")),
                     team=[worker], store=InMemoryStore(), stop_when=[step_count(4)],
                     policy={"rules": {"financial": "approve"}, "approve": "park"})
        result = lead.run("go")
        self.assertNotIsInstance(result, SuspendedResult)
        self.assertEqual(REFUNDED, [])
        self.assertIn("the agent around", json.dumps(worker.model.sent[-1]))


class TestTeam(unittest.TestCase):

    # An added member shares the lead's model, so the script alternates:
    # the lead's turn, the member's answer, the lead's next turn.

    def test_extend_makes_a_role_once_and_reuses_it(self):
        lead = Agent(_model(_call("delegate", {"task": "a", "instructions": "translator"}, "d1"),
                            _text("first done"),
                            _call("delegate", {"task": "b", "instructions": "translator"}, "d2"),
                            _text("second done"),
                            _text("done")),
                     team={"extend": True}, stop_when=[step_count(6)])
        made = []
        original = Agent.__init__

        def counting(self, *a, **k):
            made.append(1)
            original(self, *a, **k)

        from unittest.mock import patch
        with patch.object(Agent, "__init__", counting):
            lead.run("go")
        self.assertEqual(len(made), 1)                 # the second delegate reused it
        self.assertIn("second done", json.dumps(lead.model.sent[-1]))

    def test_max_bounds_the_team(self):
        lead = Agent(_model(_call("delegate", {"task": "a", "instructions": "one"}, "d1"),
                            _text("first done"),
                            _call("delegate", {"task": "b", "instructions": "two"}, "d2"),
                            _text("done")),
                     team={"extend": True, "max": 1}, stop_when=[step_count(6)])
        lead.run("go")
        self.assertIn("at its max of 1", json.dumps(lead.model.sent[-1]))

    def test_its_keys_and_shapes_are_checked(self):
        for bad in ({"members": [], "extnd": True}, {"extend": True, "max": 0},
                    {"members": ["not an agent"]}):
            with self.subTest(team=bad), self.assertRaises(ValueError):
                Agent(_model(_text("x")), team=bad)

    def test_auto_warns(self):
        _, said = _quiet(lambda: Agent(_model(_text("x")), team="auto"))
        self.assertTrue(any("team='auto' goes in 5.0" in m for m in said))

    def test_the_lead_reads_when_to_delegate(self):
        worker = Agent(_model(_text("x")), name="w", description="Does it.")
        lead = Agent(_model(_text("done")), team={"members": [worker], "extend": True})
        lead.run("go")
        delegate = next(t for t in lead.model.sent[-1]["tools"]
                        if t["function"]["name"] == "delegate")["function"]
        self.assertIn(DELEGATE_WHEN, delegate["description"])
        self.assertEqual(delegate["parameters"]["required"], ["task"])
        self.assertIn("instructions", delegate["parameters"]["properties"])


class TestAResumedCallGoesThroughTheDispatcher(unittest.TestCase):
    """A parked tool completed on resume ran `tool.run()` directly — no
    interceptors, no budget, no rules around it. It goes through the
    dispatcher now, as every other call does."""

    def test_an_interceptor_sees_the_completed_call(self):
        from yait_aichain.tools import Wait

        class Stamp(Interceptor):
            def after(self, call, result, run):
                return {"stamped": result}

        agent = Agent(_model(_call("wait", {}), _text("done")), tools=[Wait(reason="ok?")],
                      store=InMemoryStore(), policy={"interceptors": [Stamp()]})
        parked = agent.run("go")
        agent.resume(parked.run_id, signal={"ok": 1})
        self.assertIn("stamped", json.dumps(agent.model.sent[-1]))


class TestCostBudget(unittest.TestCase):

    def test_it_warns_toward_max_cost(self):
        _, said = _quiet(lambda: Agent(_model(_text("x")),
                                       stop_when=[step_count(3), cost_budget(1.0)]))
        self.assertTrue(any("give max_cost= instead" in m for m in said))


if __name__ == "__main__":
    unittest.main()
