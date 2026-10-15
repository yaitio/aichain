"""
One answer for Chain and Pool: what a runner is, what it reads, what its output
is known to hold, and running it — and a chain built to read a field a step's
output is known not to have is refused when it is built.
"""

import dataclasses
import os
import sys
import unittest
from typing import TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Chain, Skill
from yait_aichain._runners import inputs, invoke, kind_of, outputs
from yait_aichain.agent import Agent
from yait_aichain.chain import Step
from yait_aichain.testing import ScriptedModel
from yait_aichain.tools import tool


class Page(TypedDict):
    title: str
    body: str


@dataclasses.dataclass
class Price:
    amount: float
    currency: str


def fetch_page(url: str, lang: str = "en") -> Page:
    """Fetch a page.

    Parameters
    ----------
    url : str
        Article URL.
    lang : str
        Language.
    """
    return {"title": "t", "body": "b"}


def price() -> Price:
    return Price(1.0, "EUR")


def count(text: str) -> int:
    return len(text)


@tool(risk="read")
def lookup(order_id: str) -> str:
    """Look an order up.

    Parameters
    ----------
    order_id : str
        The order.
    """
    return "found"


SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}


def _skill(text="x", **kw):
    return Skill(ScriptedModel([text]), **kw)


class TestWhatARunnerIs(unittest.TestCase):

    def test_kinds(self):
        self.assertEqual([kind_of(r) for r in (count, lookup, Agent(ScriptedModel(["x"])),
                                               _skill(prompt="x"))],
                         ["call", "tool", "agent", "run"])
        with self.assertRaises(TypeError):
            kind_of(42)

    def test_what_each_reads(self):
        self.assertEqual(inputs(fetch_page),
                         {"url": {"type": "string", "description": "Article URL.", "required": True},
                          "lang": {"type": "string", "description": "Language.", "default": "en"}})
        self.assertEqual(inputs(lookup)["order_id"]["required"], True)
        self.assertEqual(list(inputs(_skill(prompt="Hi {name}"))), ["name"])
        self.assertEqual(list(inputs(Agent(ScriptedModel(["x"])), task_key="q")), ["q"])

    def test_what_its_output_is_known_to_hold(self):
        self.assertEqual(outputs(fetch_page), {"title", "body"})
        self.assertEqual(outputs(price), {"amount", "currency"})
        self.assertEqual(outputs(_skill(prompt="x", output={"format": {"type": "json_schema",
                                                                        "schema": SCHEMA}})),
                         {"name"})
        self.assertIsNone(outputs(count))
        self.assertIsNone(outputs(_skill(prompt="x")))


class TestInvoke(unittest.TestCase):

    def test_a_skill_on_a_copy_with_its_own_usage(self):
        skill = _skill("hi", prompt="x")
        done = invoke(skill, {}, mirror=False)
        self.assertEqual(done.output, "hi")
        self.assertIsNotNone(done.usage)
        self.assertIsNone(skill.last_usage)

    def test_an_agent_reads_its_task(self):
        done = invoke(Agent(ScriptedModel(["answer"])), {"task": "go"})
        self.assertEqual(done.output, "answer")
        with self.assertRaises(ValueError):
            invoke(Agent(ScriptedModel(["x"])), {})

    def test_a_failed_chain_raises(self):
        def boom():
            raise RuntimeError("no")
        with self.assertRaises(RuntimeError):
            invoke(Chain([boom], on_step_error="collect"), {})


class TestAFieldIsCheckedWhenTheChainIsBuilt(unittest.TestCase):

    def test_a_misspelt_field_of_a_typed_function_says_the_right_one(self):
        with self.assertRaisesRegex(ValueError, "did you mean 'page.body'"):
            Chain([(fetch_page, "page"), Step(count, inputs={"text": "page.bdy"})])

    def test_a_misspelt_field_of_a_schema_answer(self):
        who = _skill('{"name": "a"}', prompt="x", name="who",
                     output={"format": {"type": "json_schema", "schema": SCHEMA}})
        with self.assertRaisesRegex(ValueError, "did you mean 'who.name'"):
            Chain([who, Step(count, inputs={"text": "who.nme"})])

    def test_the_right_field_builds_and_runs(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            chain = Chain([(fetch_page, "page"), Step(count, inputs={"text": "page.body"})])
            self.assertEqual(chain.run({"url": "u"}).output, 1)

    def test_an_unknown_shape_is_left_to_the_run(self):
        Chain([(count, "n"), Step(count, inputs={"text": "n.anything"})])


if __name__ == "__main__":
    unittest.main()
