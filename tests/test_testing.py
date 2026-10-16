"""
`yait_aichain.testing`: the stand-ins go through the library's own request and
response path, so a test written on them exercises the code that runs live.
"""

import json
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Budget, BudgetExceeded, Chain, Model, Pool, Skill
from yait_aichain.agent import Agent
from yait_aichain.testing import ScriptedModel, call, recorded
from yait_aichain.tools import tool

DONE: list = []


@tool(risk="financial")
def refund(amount: int) -> str:
    """Refund an amount.

    Parameters
    ----------
    amount : int
        Dollars to refund.
    """
    DONE.append(amount)
    return f"refunded {amount}"


class TestScriptedModel(unittest.TestCase):

    def setUp(self):
        DONE.clear()

    def test_a_skill_gets_the_answers_in_order_and_the_last_repeats(self):
        skill = Skill(ScriptedModel(["one", "two"]), prompt="x")
        self.assertEqual([skill.run() for _ in range(3)], ["one", "two", "two"])

    def test_an_agent_runs_the_scripted_call(self):
        model = ScriptedModel([call("refund", amount=42), "Refunded."])
        result = Agent(model, tools=[refund]).run("refund 42")
        self.assertEqual((result.output, DONE), ("Refunded.", [42]))
        self.assertIn("refund", [t["function"]["name"] for t in model.requests[0].tools])
        self.assertEqual(model.requests[1].messages[-1]["role"], "tool")

    def test_several_calls_in_one_turn(self):
        model = ScriptedModel([[call("refund", amount=1), call("refund", amount=2)], "ok"])
        Agent(model, tools=[refund]).run("go")
        self.assertEqual(DONE, [1, 2])

    def test_a_dict_is_a_json_answer(self):
        skill = Skill(ScriptedModel([{"a": 1}]), prompt="x", output={"format": {"type": "json"}})
        self.assertEqual(skill.run(), {"a": 1})

    def test_a_stream_arrives_word_by_word_and_events_are_real(self):
        skill = Skill(ScriptedModel(["one two three"]), prompt="x")
        self.assertEqual(list(skill.stream()), ["one ", "two ", "three"])
        model = ScriptedModel([call("refund", amount=5), "All done."])
        types = [e.type for e in Agent(model, tools=[refund]).stream("go")]
        self.assertLess(types.index("tool_call.ended"), types.index("text.delta"))
        self.assertEqual((types[0], types[-1]), ("run.started", "run.finished"))

    def test_usage_and_price_make_a_cost_a_budget_counts(self):
        model = ScriptedModel(["x"], usage={"input": 1000, "output": 1000},
                              price={"input": 1.0, "output": 3.0})
        skill = Skill(model, prompt="x")
        skill.run()
        self.assertAlmostEqual(skill.last_usage.cost, 0.004)
        with self.assertRaises(BudgetExceeded):
            Chain([Skill(model, prompt="x", name=f"s{i}") for i in range(4)],
                  max_cost=Budget(0.01)).run()

    def test_without_a_price_the_cost_is_none(self):
        skill = Skill(ScriptedModel(["x"]), prompt="x")
        skill.run()
        self.assertIsNone(skill.last_usage.cost)

    def test_a_pool_shares_one_script_safely(self):
        model = ScriptedModel(["x"])
        result = Pool(Skill(model, prompt="{n}"), items=[{"n": i} for i in range(20)], max_flows=8).run()
        self.assertTrue(result.success)
        self.assertEqual(len(model.requests), 20)

    def test_bad_arguments_are_refused(self):
        for bad in ([], "x"):
            with self.subTest(replies=bad), self.assertRaises(ValueError):
                ScriptedModel(bad)
        with self.assertRaises(ValueError):
            ScriptedModel(["x"], price={"input": 1})

    def test_it_is_a_model(self):
        self.assertIsInstance(ScriptedModel(["x"]), Model)


class TestRecorded(unittest.TestCase):

    KEY = "sk-test-abcdefgh1234"

    def _run(self, name, **options):
        model = Model(name, options=options, api_key=self.KEY)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with recorded(model, reply="hello") as rec:
                skill = Skill(model, prompt="Hi")
                out = skill.run()
        return model, rec, skill, out

    def test_every_text_family_answers_and_is_priced(self):
        for name in ("gpt-5.5", "gpt-4o", "claude-haiku-5-5", "gemini-3.1-flash-lite", "deepseek-chat"):
            with self.subTest(model=name):
                _, rec, skill, out = self._run(name)
                self.assertEqual(out, "hello")
                self.assertEqual(len(rec.requests), 1)
                self.assertGreater(skill.last_usage.cost, 0)

    def test_the_key_is_a_fingerprint(self):
        for name in ("gpt-4o", "claude-haiku-5-5", "gemini-3.1-flash-lite"):
            with self.subTest(model=name):
                _, rec, _, _ = self._run(name)
                text = json.dumps(rec.requests[0].headers, ensure_ascii=False)
                self.assertNotIn(self.KEY, text)
                self.assertIn("…1234", text)

    def test_adaptations_are_the_librarys_own(self):
        _, rec, skill, _ = self._run("gpt-5.5", temperature=0.2)
        self.assertIn(("declined", "temperature"),
                      [(a.kind, a.option) for a in skill.last_adaptations])
        self.assertNotIn("temperature", json.dumps(rec.requests[0].body))

    def test_the_transport_comes_back(self):
        model, _, _, _ = self._run("gpt-4o")
        self.assertNotIn("send", model.client.__dict__)

    def test_a_stream_inside_says_it_is_not_recorded(self):
        model = Model("gpt-4o", api_key=self.KEY)
        with recorded(model), self.assertRaises(NotImplementedError):
            list(Skill(model, prompt="x").stream())


if __name__ == "__main__":
    unittest.main()
