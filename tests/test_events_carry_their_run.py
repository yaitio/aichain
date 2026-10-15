"""
Every event of a run carries that run's id — a Chain's, a Pool's, an
Agent's — including the model calls a Skill makes inside it; and a pause's
answer reaches the next step without a warning about the habit going in 5.0.
"""

import os
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Budget, BudgetExceeded, Chain, Pool, Skill, Tracer
from yait_aichain.agent import Agent
from yait_aichain.state import FileStore
from yait_aichain.testing import ScriptedModel
from yait_aichain.tools import tool, wait


@tool(risk="read")
def echo(reply: str) -> str:
    """Echo.

    Parameters
    ----------
    reply : str
        The text.
    """
    return reply


class TestModelCallsCarryTheRun(unittest.TestCase):

    def _skill(self, tracer):
        return Skill(ScriptedModel(["x"]), prompt="{n}", name="s", hooks=[tracer])

    def test_inside_a_chain(self):
        tracer = Tracer()
        result = Chain([self._skill(tracer)], hooks=[tracer]).run({"n": 1})
        calls = [e for e in tracer.events if e.type.startswith("llm_call.")]
        self.assertTrue(calls)
        self.assertEqual({e.run_id for e in calls}, {result.run_id})

    def test_inside_a_pool_across_its_threads(self):
        tracer = Tracer()
        Pool(self._skill(tracer), items=[{"n": i} for i in range(6)], max_flows=3,
             hooks=[tracer]).run()
        ids = {e.run_id for e in tracer.events}
        self.assertEqual(len(ids), 1)
        self.assertTrue(next(iter(ids)).startswith("pool-"))

    def test_a_skill_alone_has_none(self):
        tracer = Tracer()
        self._skill(tracer).run(variables={"n": 1})
        self.assertEqual({e.run_id for e in tracer.events}, {None})

    def test_an_agent_stamps_a_skill_inside_its_tool(self):
        tracer = Tracer()
        inner = self._skill(tracer)

        @tool(risk="read")
        def ask(n: str) -> str:
            """Ask.

            Parameters
            ----------
            n : str
                What.
            """
            return inner.run(variables={"n": n})

        from yait_aichain.testing import call
        Agent(ScriptedModel([call("ask", n="1"), "done"]), tools=[ask], hooks=[tracer]).run("go")
        inner_calls = [e for e in tracer.events if e.type == "llm_call.started" and e.name == "scripted"]
        self.assertEqual(len({e.run_id for e in inner_calls}), 1)
        self.assertIsNotNone(inner_calls[0].run_id)


class TestAPausesAnswerIsItsVariables(unittest.TestCase):

    def test_no_warning_and_the_next_step_reads_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            chain = Chain([wait(reason="ok?", resume_with={"reply": "str"}), echo],
                          store=FileStore(tmp))
            parked = chain.run()
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                result = chain.resume(parked.run_id, signal={"reply": "hello"})
        self.assertEqual(result.output, "hello")


class TestAKeyItsNameCannotBe(unittest.TestCase):

    def test_the_warning_says_the_name_is_not_a_key(self):
        with self.assertWarnsRegex(FutureWarning, "'step 1' is not one a key can be"):
            Chain([Skill(ScriptedModel(["x"]), prompt="x", name="step 1")])


class TestARefusedCallIsNotARequest(unittest.TestCase):

    def test_the_budget_refuses_before_it_is_sent(self):
        model = ScriptedModel(["x"], usage={"input": 1000, "output": 1000},
                              price={"input": 1.0, "output": 3.0})
        chain = Chain([Skill(model, prompt="x", name=f"s{i}") for i in range(4)],
                      max_cost=Budget(0.01))
        with self.assertRaises(BudgetExceeded):
            chain.run()
        self.assertEqual(len(model.requests), 3)


if __name__ == "__main__":
    unittest.main()
