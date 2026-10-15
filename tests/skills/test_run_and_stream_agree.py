"""
`run()` and `stream()` are one path: on the same answer they give the same
result, the same usage and cost, and the same budget charge.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain import Budget, Skill
from yait_aichain.testing import ScriptedModel

SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}


def _both(reply, output=None):
    out = []
    for streamed in (False, True):
        budget = Budget(1.0)
        skill = Skill(ScriptedModel([reply], usage={"input": 100, "output": 50},
                                    price={"input": 1.0, "output": 2.0}),
                      prompt="x", output=output, max_cost=budget)
        if streamed:
            list(skill.stream())
            result = skill.last_result
        else:
            result = skill.run()
        out.append((result, skill.last_usage.total_tokens, skill.last_usage.cost, budget.spent))
    return out


class TestRunAndStreamAgree(unittest.TestCase):

    def test_text(self):
        buffered, streamed = _both("one two three")
        self.assertEqual(buffered, streamed)

    def test_json(self):
        buffered, streamed = _both('{"n": 1}', {"format": {"type": "json"}})
        self.assertEqual(buffered, streamed)
        self.assertEqual(buffered[0], {"n": 1})

    def test_json_schema(self):
        buffered, streamed = _both('{"n": 2}', {"format": {"type": "json_schema", "schema": SCHEMA}})
        self.assertEqual(buffered, streamed)

    def test_a_schema_violation_raises_on_both(self):
        from yait_aichain.clients._errors import InvalidStructuredOutputError
        for streamed in (False, True):
            skill = Skill(ScriptedModel(['{"n": "x"}']), prompt="x",
                          output={"format": {"type": "json_schema", "schema": SCHEMA}})
            with self.subTest(streamed=streamed), self.assertRaises(InvalidStructuredOutputError):
                list(skill.stream()) if streamed else skill.run()


if __name__ == "__main__":
    unittest.main()
