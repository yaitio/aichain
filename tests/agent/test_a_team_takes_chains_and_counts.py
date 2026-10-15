"""
A team member may be a Chain, which takes its inputs by name and runs under the
lead's rules, budget and hooks; a team can bound its delegations; a member that
had nothing the lead lacks and finished in one step is said to have cost more
than it saved; and the items of an agent's `pool` are its call's children.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain import Budget, BudgetExceeded, Chain, Skill, Tracer
from yait_aichain.agent import Agent, step_count
from yait_aichain.testing import ScriptedModel, call
from yait_aichain.tools import tool

PUSHED: list = []


def fetch_page(url: str) -> str:
    """Fetch a page.

    Parameters
    ----------
    url : str
        Article URL.
    """
    return f"<page of {url}>"


@tool(risk="privileged")
def push(text: str) -> str:
    """Push it.

    Parameters
    ----------
    text : str
        What to push.
    """
    PUSHED.append(text)
    return "pushed"


@tool(risk="read")
def double(x: int) -> int:
    """Double a number.

    Parameters
    ----------
    x : int
        The number.
    """
    return 2 * x


def _reader(price=None):
    summary = Skill(ScriptedModel(["a summary"], usage={"input": 1000, "output": 1000},
                                  price=price), prompt="Summarise {page}", name="summary")
    return Chain([(fetch_page, "page"), summary], name="reader",
                 description="Reads a web page and summarises it")


def _told(model, turn=1):
    return json.loads(model.requests[turn].messages[-1]["parts"][0]["text"])


class TestAChainIsAMember(unittest.TestCase):

    def setUp(self):
        PUSHED.clear()

    def test_it_takes_its_inputs_by_name(self):
        lead_model = ScriptedModel([call("delegate", task="read", worker="reader",
                                         inputs={"url": "https://x"}), "done"])
        Agent(lead_model, team=[_reader()]).run("go")
        told = _told(lead_model)
        self.assertEqual((told["success"], told["output"]), (True, "a summary"))

    def test_the_lead_reads_its_inputs(self):
        lead_model = ScriptedModel(["done"])
        Agent(lead_model, team=[_reader()]).run("go")
        delegate = next(t for t in lead_model.requests[0].tools
                        if t["function"]["name"] == "delegate")["function"]
        self.assertIn("inputs: url (required — Article URL.)",
                      delegate["parameters"]["properties"]["worker"]["description"])
        self.assertIn("inputs", delegate["parameters"]["properties"])

    def test_the_task_fills_a_single_text_input(self):
        lead_model = ScriptedModel([call("delegate", task="https://y", worker="reader"), "done"])
        Agent(lead_model, team=[_reader()]).run("go")
        self.assertTrue(_told(lead_model)["success"])

    def test_a_missing_input_is_named(self):
        two = Chain([Skill(ScriptedModel(["x"]), prompt="{a} {b}", name="both")], name="two",
                    description="Needs two")
        lead_model = ScriptedModel([call("delegate", task="t", worker="two"), "done"])
        Agent(lead_model, team=[two]).run("go")
        self.assertIn("needs the inputs ['a', 'b']", json.dumps(lead_model.requests[1].messages))

    def test_the_leads_rules_hold_inside_it(self):
        pusher = Chain([push], name="pusher", description="Pushes")
        lead_model = ScriptedModel([call("delegate", task="go live", worker="pusher"), "done"])
        Agent(lead_model, team=[pusher], policy={"rules": {"privileged": "deny"}}).run("go")
        self.assertEqual(PUSHED, [])

    def test_the_leads_budget_holds_inside_it(self):
        lead_model = ScriptedModel([call("delegate", task="read", worker="reader",
                                         inputs={"url": "u"}), "done"])
        budget = Budget(0.001)
        lead = Agent(lead_model, team=[_reader(price={"input": 1.0, "output": 3.0})],
                     max_cost=budget)
        result = lead.run("go")
        # The member's call began under the limit and spent past it; the lead's
        # next turn would begin over it, so it does not begin.
        self.assertAlmostEqual(budget.spent, 0.004)
        self.assertEqual(len(lead_model.requests), 1)
        self.assertFalse(result.success)

    def test_the_leads_hooks_hear_it(self):
        tracer = Tracer()
        lead_model = ScriptedModel([call("delegate", task="read", worker="reader",
                                         inputs={"url": "u"}), "done"])
        Agent(lead_model, team=[_reader()], hooks=[tracer]).run("go")
        names = [(e.type, e.name) for e in tracer.events]
        self.assertIn(("run.started", "reader"), names)
        self.assertIn(("step.started", "summary"), names)

    def test_a_member_is_an_agent_or_a_chain(self):
        with self.assertRaises(ValueError):
            Agent(ScriptedModel(["x"]), team=[Skill(ScriptedModel(["x"]), prompt="x")])


class TestDelegationIsCounted(unittest.TestCase):

    def _worker(self):
        return Agent(ScriptedModel(["worker done"]), name="w", description="Does it.")

    def test_max_delegations_bounds_a_run(self):
        lead_model = ScriptedModel([call("delegate", task="a", worker="w"),
                                    call("delegate", task="b", worker="w"), "done"])
        Agent(lead_model, team={"members": [self._worker()], "max_delegations": 1},
              stop_when=[step_count(5)]).run("go")
        self.assertIn("max_delegations", json.dumps(lead_model.requests[2].messages))

    def test_its_value_is_checked(self):
        with self.assertRaises(ValueError):
            Agent(ScriptedModel(["x"]), team={"members": [self._worker()], "max_delegations": 0})

    def test_small_work_with_nothing_the_lead_lacks_is_said(self):
        model = ScriptedModel([call("delegate", task="a", instructions="helper"),
                               "worker done", "done"])
        Agent(model, team={"extend": True}, stop_when=[step_count(5)]).run("go")
        told = _told(model, 2)
        self.assertIn("cheaper done with your own calls", told["note"])
        self.assertLessEqual(told["steps"], 1)

    def test_a_member_with_its_own_model_is_not_told_off(self):
        worker = Agent(ScriptedModel(["worker done"], name="other"), name="w",
                       description="Does it.")
        lead_model = ScriptedModel([call("delegate", task="a", worker="w"), "done"])
        Agent(lead_model, team=[worker]).run("go")
        self.assertNotIn("note", _told(lead_model))


class TestAPoolsItemsAreItsChildren(unittest.TestCase):

    def _run(self, items):
        tracer = Tracer()
        model = ScriptedModel([call("pool", tool="double", items=items), "done"])
        Agent(model, tools=[double], hooks=[tracer]).run("go")
        return tracer

    def test_each_item_is_a_call_under_the_pool_call(self):
        tracer = self._run([{"x": 1}, {"x": 2}])
        pool_id = next(e.payload["id"] for e in tracer.events
                       if e.type == "tool_call.started" and e.name == "pool")
        children = [e for e in tracer.events if e.type == "tool_call.started" and e.name == "double"]
        self.assertEqual(len(children), 2)
        self.assertEqual({e.payload["parent_call_id"] for e in children}, {pool_id})
        self.assertEqual(len({e.run_id for e in tracer.events}), 1)
        self.assertNotIn("run.started", [e.type for e in tracer.events][1:])

    def test_long_arguments_are_cut_and_said(self):
        tracer = self._run([{"x": "9" * 10_000}])
        child = next(e for e in tracer.events if e.type == "tool_call.started" and e.name == "double")
        self.assertFalse(child.payload["complete"])
        self.assertLessEqual(len(child.payload["arguments"].encode()), 8 * 1024)

    def test_no_result_is_written_for_a_child(self):
        tracer = self._run([{"x": 1}])
        ended = next(e for e in tracer.events if e.type == "tool_call.ended" and e.name == "double")
        self.assertNotIn("result", ended.payload)


if __name__ == "__main__":
    unittest.main()
