"""
What a chain needs from its caller, and the forms that are going away.

4.2.8 (PLAN-agent-chain.md, 6.5–6.7). `input_schema()` reads what every step
already says — a function's signature and docstring, a tool's schema, a
Skill's placeholders, an Agent's task, a nested chain's own schema — leaves out
what an earlier step writes, and says when an input has no description,
because a caller or a model filling it has only its name. `chain.accumulated`,
`chain.history` and `on_step_error=` on `run()` warn; they go in 5.0.
"""

import os
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import Agent, Chain, Model, Skill
from yait_aichain.tools._base import Tool


def fetch_page(url: str) -> str:
    """Fetch a page.

    Parameters
    ----------
    url : str
        Article URL.
    """
    return "page"


def count(text: str, unique: bool = False) -> int:
    """Count words.

    text: the text to count.
    unique: count each word once.
    """
    return len(text.split())


class Lookup(Tool):
    name = "lookup"
    description = "Look a thing up."
    parameters = {"type": "object", "properties": {
        "q": {"type": "string", "description": "What to look up."},
        "limit": {"type": "integer", "description": "How many."}}, "required": ["q"]}
    risk = "read"

    def run(self, q, limit=3, options=None):
        return q


def _quiet(make):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        value = make()
    return value, [str(w.message) for w in seen]


class TestTheSchema(unittest.TestCase):

    def test_the_plans_example(self):
        translator = Skill(Model("gpt-4o", api_key="k"),
                           prompt="Translate {fetch_page} to {language}", name="translator")
        chain = Chain([fetch_page, translator], variables={"language": "English"})
        schema, said = _quiet(chain.input_schema)
        self.assertEqual(schema, {
            "url": {"type": "string", "description": "Article URL.", "required": True},
            "language": {"type": "string", "default": "English"}})
        self.assertTrue(any("'language' has no description" in m for m in said))

    def test_inputs_describes_what_the_steps_do_not(self):
        translator = Skill(Model("gpt-4o", api_key="k"), prompt="Into {language}",
                           name="translator")
        chain = Chain([translator], inputs={"language": "Target language, e.g. French"})
        schema, said = _quiet(chain.input_schema)
        self.assertEqual(schema["language"]["description"], "Target language, e.g. French")
        self.assertEqual(said, [])

    def test_a_tool_and_a_default(self):
        schema, _ = _quiet(Chain([(Lookup(), "found"), (count, "n", {"text": "found"})])
                           .input_schema)
        self.assertEqual(schema["q"], {"type": "string", "description": "What to look up.",
                                       "required": True})
        self.assertNotIn("found", schema)                  # written by the step before
        self.assertEqual(schema["limit"], {"type": "integer", "description": "How many."})

    def test_a_function_default_is_not_required(self):
        schema, _ = _quiet(Chain([(count, "n")]).input_schema)
        self.assertEqual(schema["unique"], {"type": "boolean", "description": "count each word once.",
                                            "default": False})

    def test_an_agent_and_a_nested_chain(self):
        inner = Chain([fetch_page], name="inner")
        agent = Agent(Model("gpt-4o", api_key="k"), name="writer", description="Writes.")
        schema, _ = _quiet(Chain([(inner, "page"), (agent, "text")]).input_schema)
        self.assertEqual(set(schema), {"url", "task"})


class TestTheFormsThatAreGoing(unittest.TestCase):

    def test_each_warns_and_still_works(self):
        chain = Chain([(count, "n", {"text": "words"})])
        result, said = _quiet(lambda: chain.run(variables={"words": "a b"}, on_step_error="raise"))
        self.assertEqual(result["n"], 2)
        self.assertTrue(any("on_step_error= on run()" in m for m in said))
        acc, said = _quiet(lambda: chain.accumulated)
        self.assertEqual(acc["n"], 2)
        self.assertTrue(any("result.variables" in m for m in said))
        hist, said = _quiet(lambda: chain.history)
        self.assertEqual(len(hist), 1)
        self.assertTrue(any("result.history" in m for m in said))


class TestSaved(unittest.TestCase):

    def test_inputs_and_policy_survive_a_save(self):
        translator = Skill(Model("gpt-4o", api_key="k"), prompt="Into {language}",
                           name="translator")
        chain = Chain([(translator, "out")], inputs={"language": "Target language"},
                      policy={"destructive": "deny"})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.yaml")
            chain.save(path)
            back = Chain.load(path, api_key="k")
        self.assertEqual(_quiet(back.input_schema)[0], _quiet(chain.input_schema)[0])
        self.assertEqual(back.policy.rules["destructive"], "deny")


if __name__ == "__main__":
    unittest.main()
