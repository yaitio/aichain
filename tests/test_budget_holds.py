"""
A budget takes only money it can count.

`Budget(nan)` was never exhausted, `charge(-10)` gave money back, `True` was a
dollar, a NaN cost from a proxy turned the budget off, and the message showed
$0.0000 for a fraction of a cent.
"""

import math
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Budget, BudgetExceeded
from yait_aichain.models._usage import extract_usage


class TestBudget(unittest.TestCase):

    def test_a_limit_is_finite_and_positive(self):
        for bad in (math.nan, math.inf, 0, -1, True, "1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Budget(bad)

    def test_a_charge_is_finite_and_not_negative(self):
        b = Budget(1.0)
        for bad in (-10, math.nan, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                b.charge(bad)
        b.charge(None)
        b.charge(0)
        self.assertEqual(b.spent, 0.0)

    def test_a_small_spend_is_shown(self):
        self.assertIn("0.000050", str(BudgetExceeded(0.00005, 0.00001)))

    def test_a_nan_cost_reported_by_a_proxy_is_set_aside(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            usage = extract_usage({"usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                             "cost": float("nan")}})
        self.assertIsNone(usage.cost)
        self.assertTrue(any("not a price" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
