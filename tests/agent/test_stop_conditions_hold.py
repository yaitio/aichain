"""
Every stop condition does what it says, on both seams.

``stop_when=[check(...)]`` dropped the default ceiling and the loop had none;
``cost_budget`` on a model with no price never fired; two nudges of one name
shared one latch; the latch was lost on resume; ``step()`` counted no steps,
so a driver's ``step_count`` never moved; the journal cut an action at 200
characters, and ``repeating()`` fired on calls that differed past the cut.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent, check, cost_budget, nudge, step_count   # noqa: E402
from yait_aichain.agent._agent import _action                                 # noqa: E402
from yait_aichain.state._run import AgentRun                                  # noqa: E402
from yait_aichain.models import Model                                         # noqa: E402
from yait_aichain.models._calls import ToolCall                               # noqa: E402
from test_loop import Echo, _calls, _model                                    # noqa: E402


class TestCeilings(unittest.TestCase):

    def test_a_list_with_no_ceiling_gets_one_and_says_so(self):
        with self.assertWarnsRegex(UserWarning, "no ceiling"):
            agent = Agent(Model("gpt-4o", api_key="k"), stop_when=[check(lambda s: False)])
        self.assertTrue(any(getattr(c, "name", "") == "step_count" for c in agent.stop_when))

    def test_a_budget_on_a_model_with_no_price_is_said(self):
        with self.assertWarnsRegex(UserWarning, "no price"):
            Agent(Model("private/llama3.3", api_key="k"),
                  stop_when=[step_count(3), cost_budget(1.0)])

    def test_a_priced_model_says_nothing(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # The one warning it does give since 4.2.9 is that cost_budget goes
            # in 5.0; what is pinned is that a priced model gets no price one.
            warnings.filterwarnings("ignore", category=FutureWarning)
            Agent(Model("gpt-4o", api_key="k"), stop_when=[step_count(3), cost_budget(1.0)])


class TestNudgeLatch(unittest.TestCase):

    def test_two_nudges_of_one_name_both_speak(self):
        agent = Agent(Model("gpt-4o", api_key="k"),
                      stop_when=[step_count(5), nudge(lambda s: True, "first"),
                                 nudge(lambda s: True, "second")])
        state = agent.new_state()
        self.assertEqual(agent.nudges(state), ["first", "second"])
        self.assertEqual(agent.nudges(state), [])          # once per streak

    def test_the_latch_survives_a_pause(self):
        agent = Agent(Model("gpt-4o", api_key="k"),
                      stop_when=[step_count(5), nudge(lambda s: True, "once")])
        state = agent.new_state()
        agent.nudges(state)
        resumed = {**agent.new_state(), **AgentRun.portable(state)}
        self.assertEqual(agent.nudges(resumed), [])


class TestTheStepSeam(unittest.TestCase):

    def test_step_counts_a_decision_and_stopped_reads_the_ceiling(self):
        agent = Agent(model=_model(_calls(("c1", "echo", {"value": "x"}))),
                      tools=[Echo()], stop_when=[step_count(1)], verbose=0)
        state = agent.new_state()
        self.assertIsNone(agent.stopped(state))
        agent.step(agent.opening("go"), state)
        self.assertEqual(state["steps"], 1)
        self.assertEqual(agent.stopped(state), "step_count")


class TestJournalAction(unittest.TestCase):

    def test_calls_that_differ_past_the_cut_are_different_actions(self):
        long = "x" * 300
        a = _action(ToolCall(id="1", name="search", arguments={"q": long + "a"}))
        b = _action(ToolCall(id="2", name="search", arguments={"q": long + "b"}))
        self.assertNotEqual(a, b)
        self.assertIn("search", a)


if __name__ == "__main__":
    unittest.main()
