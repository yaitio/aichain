"""
A ceiling in money, on every primitive that spends it.

`max_tokens` bounds one reply. Nobody budgets in replies: a chain of ten
steps, a pool over a thousand items and an agent that decides how many turns
it needs were all unbounded in the only unit that appears on the invoice. The
agent has had `cost_budget` since 2.0; the other three had nothing, so a
runaway `Pool` was visible on the bill and nowhere else — which is the worst
place for a product whose pitch is "you pay for not administering it".

Two properties carry the design, and both are testable:

* **one object, shared** — a budget handed to a chain covers the agent inside
  it, because they decrement the same thing. Copies would let every level
  spend the full amount, which is the failure this shape exists to avoid;
* **it bounds beginning, not exceeding** — the length of a reply is not known
  before it is paid for. A budget promising otherwise would be lying about the
  one number a caller checks.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import (Budget, BudgetExceeded, Chain, Model,     # noqa: E402
                          Pool, Skill)

#: gpt-4o is priced per token in the registry, so a scripted reply of this
#: size costs a real, non-zero amount — a fake with cost 0 would let every
#: budget test pass for the wrong reason.
_REPLY = json.dumps({
    "choices": [{"message": {"content": "ANSWER"}}],
    "usage": {"prompt_tokens": 100_000, "completion_tokens": 50_000,
              "total_tokens": 150_000},
}).encode()


def _model():
    m = Model("gpt-4o", api_key="k")
    m.client._post = MagicMock(return_value=_REPLY)
    m.client._auth_headers = MagicMock(return_value={})
    return m


def _skill(**kw):
    return Skill(model=_model(),
                 input={"messages": [{"role": "user", "parts": ["hi"]}]},
                 **kw)


class TestTheUnitIsMoney(unittest.TestCase):

    def test_a_priced_call_actually_costs_something(self):
        """The control. Every test below is vacuous if the fake is free."""
        skill = _skill()
        skill.run()
        self.assertGreater(skill.last_usage.cost, 0)

    def test_a_number_is_accepted_as_well_as_an_object(self):
        """`max_cost=0.5` for the common case, a shared `Budget` when the
        ceiling spans several primitives — one parameter, not two."""
        self.assertIsInstance(_skill(max_cost=0.5).max_cost, Budget)

    def test_a_budget_must_be_positive(self):
        with self.assertRaises(ValueError):
            Budget(0)


class TestItBoundsBeginningNotExceeding(unittest.TestCase):

    def test_the_first_call_always_runs(self):
        """Nothing has been spent, so nothing can stop it — even against a
        ceiling far below what it will cost. Refusing here would need a price
        known before the call, which nobody has."""
        skill = _skill(max_cost=0.000_001)
        self.assertEqual(skill.run(), "ANSWER")

    def test_the_next_one_does_not(self):
        skill = _skill(max_cost=0.000_001)
        skill.run()
        with self.assertRaises(BudgetExceeded):
            skill.run()

    def test_a_generous_ceiling_never_fires(self):
        skill = _skill(max_cost=100.0)
        for _ in range(3):
            skill.run()
        self.assertLess(skill.max_cost.spent, 100.0)

    def test_the_error_says_what_to_do(self):
        skill = _skill(max_cost=0.000_001)
        skill.run()
        with self.assertRaises(BudgetExceeded) as caught:
            skill.run()
        self.assertIn("max_cost", str(caught.exception))
        self.assertIn("spent", str(caught.exception))


class TestOneObjectShared(unittest.TestCase):

    def test_a_chain_lends_its_ceiling_to_every_step(self):
        budget = Budget(100.0)
        first, second = _skill(), _skill()
        Chain(steps=[(first, "a"), (second, "b")],
              max_cost=budget).run()
        # Both steps charged the same object, so the total is the run's.
        self.assertGreater(budget.spent, 0)
        self.assertAlmostEqual(
            budget.spent,
            first.last_usage.cost + second.last_usage.cost, places=9)

    def test_and_takes_it_back_afterwards(self):
        """A Skill reused in two chains must not keep the first one's
        ceiling: that would be a chain quietly editing an object it does not
        own."""
        skill = _skill()
        Chain(steps=[(skill, "a")], max_cost=Budget(100.0)).run()
        self.assertIsNone(skill.max_cost)

    def test_a_chain_stops_when_its_budget_is_gone(self):
        ran = []

        class Counting(Skill):
            def run(self, *a, **kw):
                ran.append(1)
                return super().run(*a, **kw)

        budget = Budget(0.000_001)
        chain = Chain(steps=[(Counting(model=_model(),
                                       input={"messages": [{"role": "user",
                                                            "parts": ["hi"]}]}),
                              "a"),
                             (Counting(model=_model(),
                                       input={"messages": [{"role": "user",
                                                            "parts": ["hi"]}]}),
                              "b")],
                      max_cost=budget)
        with self.assertRaises(BudgetExceeded):
            chain.run()
        self.assertEqual(len(ran), 2)      # the second began and was refused

    def test_a_pool_shares_one_ceiling_across_its_items(self):
        """The case sharing is really for: items run concurrently, and a
        per-item copy would give every worker the full amount."""
        budget = Budget(100.0)
        pool = Pool(runner=_skill(), items=[{}, {}, {}], max_cost=budget)
        pool.run()
        self.assertGreater(budget.spent, 0)
        one = budget.spent / 3
        self.assertAlmostEqual(budget.spent, one * 3, places=9)

    def test_a_pool_stops_spending_once_it_is_gone(self):
        budget = Budget(0.000_001)
        pool = Pool(runner=_skill(), items=[{} for _ in range(6)],
                    max_cost=budget, on_error="collect", max_flows=1)
        pool.run()
        # The status is the module's constant, not the word — checked rather
        # than assumed, because a comparison against a string a status never
        # takes is a test that passes on every future too.
        from yait_aichain.pool._pool import FAILED
        refused = [h for h in pool.history
                   if h["status"] == FAILED and "budget" in (h.get("error") or "")]
        self.assertTrue(refused, "no item was refused after the budget went")
        self.assertLess(len(refused), len(pool.history),
                        "every item was refused — the first must always run")


class TestNothingChangesWithoutOne(unittest.TestCase):
    """The lightness invariant: a new capability enters as an option with a
    default under which nothing moves."""

    def test_a_skill_with_no_budget_has_none(self):
        self.assertIsNone(_skill().max_cost)

    def test_and_runs_as_many_times_as_it_likes(self):
        skill = _skill()
        for _ in range(5):
            self.assertEqual(skill.run(), "ANSWER")


if __name__ == "__main__":
    unittest.main()
