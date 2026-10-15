"""
A chain's result says what it cost and what it rests on.

`result.cost` is the dollars every step reported — None, never a made-up zero,
when nothing could be priced. `result.evidence` is the journal's vocabulary:
"check" when only functions and tools produced the output, "model_claim" when a
Skill or an Agent did, so a caller can tell a computed answer from a model's
word without reading the steps.
"""

import json
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain import Chain, Model, Skill


def _skill():
    m = Model("gpt-4o", api_key="k")
    m.client._post = MagicMock(return_value=json.dumps(
        {"choices": [{"message": {"content": "said"}}],
         "usage": {"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000}}).encode())
    m.client._auth_headers = MagicMock(return_value={})
    return Skill(m, prompt="Say {topic}")


def double(n: int) -> int:
    return n * 2


class TestEvidence(unittest.TestCase):

    def test_functions_alone_are_a_check(self):
        result = Chain([(double, "twice")]).run(variables={"n": 2})
        self.assertEqual((result["twice"], result.evidence), (4, "check"))

    def test_a_skill_makes_it_a_model_claim(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = Chain([(_skill(), "said")]).run(variables={"topic": "x"})
        self.assertEqual(result.evidence, "model_claim")


class TestCost(unittest.TestCase):

    def test_nothing_priced_is_none(self):
        self.assertIsNone(Chain([(double, "twice")]).run(variables={"n": 2}).cost)

    def test_a_priced_step_is_counted(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = Chain([(_skill(), "said")]).run(variables={"topic": "x"})
        self.assertGreater(result.cost, 0)
        self.assertEqual(result.cost, result.usage.cost)


if __name__ == "__main__":
    unittest.main()
