"""
A run a check ends has an output. It returned ``success=True`` with
``output=None``, so a verified run handed its caller nothing: a lead's
delegate, a Chain step, a Pool item and an Eval arm all received ``None``,
and the eval rejected agents that checked their work as empty.
"""

import os
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain import Agent, Chain, Pool
from yait_aichain.agent import check, step_count
from yait_aichain.eval import Case, Eval, exact
from yait_aichain.state import InMemoryStore
from yait_aichain.testing import ScriptedModel, call
from yait_aichain.tools import gate, tool

STORE: dict = {}
ALLOW = {"rules": {"write": "allow", "read": "allow"}, "default": "deny"}


@tool(risk="write")
def save(answer: str) -> dict:
    """Store the answer.

    answer: the answer.
    """
    STORE["answer"] = answer
    return {"saved": answer}


@tool(risk="read")
def look() -> dict:
    """Read what is stored."""
    return {"stored": STORE.get("answer")}


def _agent(script, tools=(save,), name="keeper", model=None, **check_options):
    STORE.clear()
    return Agent(model or ScriptedModel(script), tools=list(tools), name=name,
                 description="stores answers", policy=ALLOW,
                 stop_when=[check(lambda s: "answer" in STORE, name="stored", **check_options),
                            step_count(5)])


def setUpModule():
    warnings.simplefilter("ignore")


class TestTheOutput(unittest.TestCase):

    def test_by_default_it_is_the_last_tool_result(self):
        r = _agent([call("save", answer="Everest")]).run("Store it.")
        self.assertEqual((r.success, r.stopped_by, r.output, r.output_from),
                         (True, "check:stored", {"saved": "Everest"}, "tool_result"))

    def test_the_check_can_name_it(self):
        r = _agent([call("save", answer="Everest")], output=lambda s: STORE["answer"]).run("Store it.")
        self.assertEqual((r.output, r.output_from), ("Everest", "check"))

    def test_the_model_can_write_it_and_the_call_is_counted(self):
        model = ScriptedModel([call("save", answer="Everest"), "Everest, 8,849 m."],
                              usage={"input": 100, "output": 10}, price={"input": 1.0, "output": 1.0})
        r = _agent(None, model=model, answer=True).run("Store it.")
        self.assertEqual((r.output, r.output_from, r.stopped_by),
                         ("Everest, 8,849 m.", "check_answer", "check:stored"))
        self.assertEqual(r.tokens_used, 220)
        self.assertAlmostEqual(r.cost, 0.00022)

    def test_a_closing_turn_that_fails_falls_back_and_says_why(self):
        model = ScriptedModel([call("save", answer="Everest"), "never asked"],
                              usage={"input": 1_000_000, "output": 0}, price={"input": 1.0, "output": 1.0})
        STORE.clear()
        agent = Agent(model, tools=[save], policy=ALLOW, max_cost=0.5,
                      stop_when=[check(lambda s: "answer" in STORE, name="stored", answer=True)])
        r = agent.run("Store it.")
        self.assertEqual((r.success, r.output, r.output_from), (True, {"saved": "Everest"}, "tool_result"))
        why = [e["reason"] for e in r.journal if "closing answer" in e["action"]]
        self.assertTrue(why and "Budget" in why[0], why)

    def test_a_closing_turn_that_calls_a_tool_falls_back(self):
        r = _agent([call("save", answer="Everest"), call("save", answer="x")], answer=True).run("Store it.")
        self.assertEqual((r.output, r.output_from), ({"saved": "Everest"}, "tool_result"))

    def test_an_output_that_raises_is_a_condition_error(self):
        r = _agent([call("save", answer="Everest")], output=lambda s: 1 / 0).run("Store it.")
        self.assertEqual((r.success, r.stopped_by), (False, "condition_error"))


class TestEveryCallerGetsIt(unittest.TestCase):

    def test_a_lead_through_delegate(self):
        lead = Agent(ScriptedModel([call("delegate", task="Store it.", worker="keeper"), "Done."]),
                     team=[_agent([call("save", answer="Everest")], output=lambda s: STORE["answer"])])
        seen = [e["observation"] for e in lead.run("Get it stored.").journal if "delegate" in e["action"]]
        self.assertIn('"output": "Everest"', seen[0])

    def test_a_chain_step_and_a_pool_item(self):
        chain = Chain(steps=[(_agent([call("save", answer="Everest")]), "answer")])
        self.assertEqual(chain.run(variables={"task": "Store it."})["answer"], {"saved": "Everest"})
        pool = Pool(_agent([call("save", answer="Everest")]), items=[{"task": "Store it."}]).run()
        self.assertEqual(pool.output, [{"saved": "Everest"}])

    def test_an_eval_arm_is_not_rejected_as_empty(self):
        out = os.path.join(tempfile.mkdtemp(), "ledger.jsonl")
        arm = lambda case: _agent([call("save", answer="Everest")],
                                  output=lambda s: STORE["answer"]).run("Store it.")
        report = Eval([Case("c", expect="Everest")], {"agent": arm}, exact(), out=out, verbose=False).run()
        self.assertEqual(report.reject(), {})

    def test_a_stream_ends_as_the_run_does(self):
        agent = _agent([call("save", answer="Everest")])
        for _ in agent.stream("Store it."):
            pass
        ran = _agent([call("save", answer="Everest")]).run("Store it.")
        self.assertEqual((agent.last_result.output, agent.last_result.output_from),
                         (ran.output, ran.output_from))

    def test_a_run_resumed_after_a_gate(self):
        STORE.clear()
        agent = Agent(ScriptedModel([call("gate_save", answer="Everest"), call("look")]),
                      tools=[gate(save), look], policy=ALLOW, store=InMemoryStore(),
                      stop_when=[check(lambda s: "answer" in STORE, name="stored"), step_count(5)])
        parked = agent.run("Store it, once approved.")
        r = agent.resume(parked.run_id, signal={"approved": True})
        self.assertEqual((r.stopped_by, r.output, r.output_from),
                         ("check:stored", {"stored": "Everest"}, "tool_result"))


class TestTheAgentsBudgetHoldsAcrossTurns(unittest.TestCase):
    """Each turn's Skill took the agent's Budget for its own and replaced it at
    the start of its run, so max_cost held for one turn at a time."""

    def test_the_turn_after_the_ceiling_does_not_begin(self):
        model = ScriptedModel([call("look"), "never asked"],
                              usage={"input": 1_000_000, "output": 0}, price={"input": 1.0, "output": 1.0})
        r = Agent(model, tools=[look], policy=ALLOW, max_cost=0.5).run("Look.")
        self.assertEqual((r.success, r.stopped_by, r.cost), (False, "cost_budget", 1.0))
        self.assertEqual(len(model.requests), 1)


class TestTheContract(unittest.TestCase):

    def test_no_success_without_an_output(self):
        runs = {
            "answered": _agent(["Everest."]).run("Which?"),
            "check": _agent([call("save", answer="Everest")]).run("Store it."),
            "check, nothing ran": _agent([call("save", answer="Everest")],
                                         output=None).run("Store it."),
        }
        for kind, r in runs.items():
            with self.subTest(kind=kind):
                self.assertTrue(r.success)
                self.assertIsNotNone(r.output)
        self.assertEqual(runs["answered"].output_from, "answer")

    def test_a_ceiling_still_fails_with_nothing(self):
        STORE.clear()
        agent = Agent(ScriptedModel([call("look")] * 3), tools=[look], policy=ALLOW,
                      stop_when=[check(lambda s: False, name="never"), step_count(2)])
        r = agent.run("Look.")
        self.assertEqual((r.success, r.stopped_by, r.output, r.output_from),
                         (False, "step_count", None, "none"))

    def test_a_check_that_raises_is_still_a_condition_error(self):
        STORE.clear()
        agent = Agent(ScriptedModel([call("look")]), tools=[look], policy=ALLOW,
                      stop_when=[check(lambda s: 1 / 0, name="broken")])
        self.assertEqual(agent.run("Look.").stopped_by, "condition_error")

    def test_the_finished_event_says_where_the_output_came_from(self):
        seen = []
        agent = _agent([call("save", answer="Everest")])
        agent.hooks = [lambda e: seen.append(e) if e.type == "run.finished" else None]
        agent.run("Store it.")
        self.assertEqual(seen[-1].payload["output_from"], "tool_result")


if __name__ == "__main__":
    unittest.main()
