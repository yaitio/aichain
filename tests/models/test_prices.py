"""
A cost estimate is the bill's floor, said so — not a guess at its middle.

Perplexity's per-request fee was left out (about eight times low); past a
long-context threshold the cache and image rates stayed at the short-context
price; a model's price was looked up by its bare name across every provider,
so one's own `private/qwen3-32b` was priced at DashScope's rate; and calls
with no price at all left a total that looked complete.
"""

import unittest

from yait_aichain.models._usage import Usage, attach_cost, estimate_cost


class TestPrices(unittest.TestCase):

    def test_perplexity_charges_per_request(self):
        self.assertAlmostEqual(estimate_cost(Usage(), "sonar"), 0.005)

    def test_the_provider_decides_the_price(self):
        priced = estimate_cost(Usage(input_tokens=1_000_000), "qwen3-32b")
        mine = estimate_cost(Usage(input_tokens=1_000_000), "qwen3-32b", provider="private")
        self.assertIsNotNone(priced)
        self.assertIsNone(mine)

    def test_an_unpriced_call_is_counted(self):
        u = attach_cost(Usage(input_tokens=10), "a-model-nobody-priced")
        self.assertEqual((u.cost, u.unpriced_calls), (None, 1))
        both = u + attach_cost(Usage(input_tokens=10, cost=0.01), "x")
        self.assertEqual((both.cost, both.unpriced_calls), (0.01, 1))


if __name__ == "__main__":
    unittest.main()
