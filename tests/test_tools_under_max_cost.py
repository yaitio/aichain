"""
A tool that costs money is held to `max_cost`, as a model call is.

The budget was counted in Skill alone: search, fetch, decisions, speech and
embeddings spent outside it, and a decision tool reported its cost in its
answer while nothing stopped the next call.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent"))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import Budget, BudgetExceeded                               # noqa: E402
from yait_aichain.agent import Agent                                          # noqa: E402
from yait_aichain.chain import Chain                                          # noqa: E402
from yait_aichain.tools._base import Tool                                     # noqa: E402
from test_loop import _calls, _model, _text                                   # noqa: E402


class Paid(Tool):
    name = "paid"
    priced = True
    parameters = {"type": "object", "properties": {"q": {"type": "string"}}}

    def __init__(self, cost=0.30):
        self.cost, self.calls = cost, 0

    def run(self, q=""):
        self.calls += 1
        return {"answer": q, "cost": self.cost}


class Free(Paid):
    name = "free"
    priced = False


class TestToolsUnderBudget(unittest.TestCase):

    def test_a_chain_charges_a_priced_tool_and_stops_at_the_ceiling(self):
        budget, tool = Budget(0.50), Paid()
        chain = Chain(steps=[(tool, "a"), (tool, "b"), (tool, "c")], max_cost=budget)
        with self.assertRaises(BudgetExceeded):
            chain.run(variables={"q": "x"})
        self.assertEqual(tool.calls, 2)                     # 0.30, 0.60 — the third never began
        self.assertAlmostEqual(budget.spent, 0.60)

    def test_an_unpriced_tool_is_not_charged(self):
        budget = Budget(0.10)
        Chain(steps=[(Free(), "a"), (Free(), "b")], max_cost=budget).run(variables={"q": "x"})
        self.assertEqual(budget.spent, 0.0)

    def test_an_agent_charges_its_tools_to_its_budget(self):
        budget, tool = Budget(1.0), Paid(0.25)
        Agent(_model(_calls(("c1", "paid", {"q": "x"})), _text("done")), tools=[tool],
              max_cost=budget, verbose=0).run("go")
        self.assertGreaterEqual(budget.spent, 0.25)                  # the tool, and the model calls


if __name__ == "__main__":
    unittest.main()
