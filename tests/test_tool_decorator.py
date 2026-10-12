"""
@tool — any function as a tool (3.4.0).

Planned for 2.1.0 and never shipped. The function's name is the tool's name,
its docstring's first line the description, its parameters the schema, each
described by its ``name: text`` line. What cannot be made into a schema — no
description, ``*args``, an annotation with no JSON type — is refused when the
tool is made, by name, rather than sent to a model that cannot use it.
"""

import json
import os
import sys
import unittest
from typing import Literal, Optional
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Agent, Chain, Model
from yait_aichain.agent import step_count
from yait_aichain.tools import FunctionTool, Tool, tool


@tool(risk="read")
def word_count(text: str, unique: bool = False) -> dict:
    """Count the words in a text.

    text: the text to count.
    unique: count each word once.
    """
    words = text.split()
    return {"words": len(set(words)) if unique else len(words)}


class TestTheTool(unittest.TestCase):

    def test_it_is_a_tool(self):
        self.assertIsInstance(word_count, Tool)
        self.assertIsInstance(word_count, FunctionTool)
        self.assertEqual((word_count.name, word_count.description, word_count.risk),
                         ("word_count", "Count the words in a text.", "read"))

    def test_the_schema_comes_from_the_signature_and_the_docstring(self):
        self.assertEqual(word_count.parameters, {
            "type": "object",
            "properties": {"text":   {"type": "string",  "description": "the text to count."},
                           "unique": {"type": "boolean", "description": "count each word once."}},
            "required": ["text"]})

    def test_it_is_called_as_every_tool_is(self):
        self.assertEqual(word_count.run(text="to be or not to be"), {"words": 6})
        self.assertEqual(word_count.run(text="a a b", unique=True), {"words": 2})
        self.assertEqual(word_count(text="a b").output, {"words": 2})       # the safe call
        self.assertEqual(word_count("a b c").output, {"words": 3})          # positional, as Tool.__call__ passes
        self.assertIs(word_count.__wrapped__.__name__, "word_count")

    def test_a_wrong_argument_is_the_functions_own_error(self):
        with self.assertRaises(TypeError):
            word_count.run(txt="a")
        self.assertFalse(word_count(txt="a").success)

    def test_annotations_to_json_types(self):
        @tool
        def pick(colour: Literal["red", "green"], n: Optional[int] = None, ratio: float = 1.0,
                 tags: "list[str]" = (), meta: dict = None, anything=None):
            """Pick."""
        props = pick.parameters["properties"]
        self.assertEqual(props["colour"], {"type": "string", "enum": ["red", "green"]})
        self.assertEqual(props["n"], {"type": "integer"})
        self.assertEqual(props["ratio"], {"type": "number"})
        self.assertEqual(props["tags"], {"type": "array", "items": {"type": "string"}})
        self.assertEqual(props["meta"], {"type": "object"})
        self.assertEqual(props["anything"], {"type": "string"})
        self.assertEqual(pick.parameters["required"], ["colour"])

    def test_risk_is_write_unless_the_author_says_otherwise(self):
        @tool
        def act(x: str):
            """Act."""
        self.assertEqual(act.risk, "write")

    def test_settings_override(self):
        t = tool(lambda city: city, name="weather", description="The weather in a city.", risk="external")
        self.assertEqual((t.name, t.description, t.risk), ("weather", "The weather in a city.", "external"))


class TestWhatCannotBeATool(unittest.TestCase):

    def test_no_description(self):
        def f(x): return x
        with self.assertRaisesRegex(ValueError, "no description"):
            tool(f)

    def test_variadic_parameters(self):
        def f(*args):
            """F."""
        with self.assertRaisesRegex(TypeError, r"\*args has no schema"):
            tool(f)

    def test_an_annotation_with_no_json_type(self):
        def f(x: object):
            """F."""
        with self.assertRaisesRegex(TypeError, "parameter 'x': no JSON schema"):
            tool(f)

    def test_a_name_providers_refuse(self):
        def f(x: str):
            """F."""
        with self.assertRaisesRegex(ValueError, "letters, digits"):
            tool(f, name="has spaces")

    def test_an_unknown_risk(self):
        def f(x: str):
            """F."""
        with self.assertRaisesRegex(ValueError, "risk 'safe'"):
            tool(f, risk="safe")


class TestWhereItRuns(unittest.TestCase):

    def test_in_a_chain_with_a_template(self):
        res = Chain(steps=[(word_count, "n", {"text": "Count: {sentence}"})]).run(
            variables={"sentence": "to be or not"})
        self.assertEqual(res.output, {"words": 5})

    def test_in_an_agent(self):
        model = Model("gpt-4o", api_key="k")
        sent = []
        replies = [
            {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "word_count", "arguments": json.dumps({"text": "a b c"})}}]}}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            {"choices": [{"message": {"content": "3 words"}}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        ]

        def post(path, body, headers, *a, **k):
            sent.append(body)
            return json.dumps(replies[min(len(sent) - 1, 1)]).encode()

        model.client._post = MagicMock(side_effect=post)
        model.client._auth_headers = MagicMock(return_value={})
        result = Agent(model, tools=[word_count], stop_when=[step_count(3)]).run("count")
        self.assertTrue(result.success)
        tool_turns = [m for m in sent[-1]["messages"] if m.get("role") == "tool"]
        self.assertEqual(json.loads(tool_turns[0]["content"]), {"words": 3})

    def test_a_chain_that_holds_one_cannot_be_saved(self):
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "c.yaml")
        with self.assertRaisesRegex(TypeError, "a function cannot be saved"):
            Chain(steps=[(word_count, "n")]).save(path)
        self.assertFalse(os.path.exists(path))


class TestWhatTheSchemaIsReadFrom(unittest.TestCase):

    def test_a_numpy_docstring_gives_descriptions_not_types(self):
        def scale(value: float, factor: float = 2.0) -> float:
            """Multiply a value.

            Parameters
            ----------
            value : float
                The number to scale.
            factor : float, optional
                How much to scale it by,
                twice by default.
            """
            return value * factor
        props = tool(scale).parameters["properties"]
        self.assertEqual(props["value"]["description"], "The number to scale.")
        self.assertEqual(props["factor"]["description"], "How much to scale it by, twice by default.")

    def test_a_partial_is_named_and_described_by_its_function(self):
        import functools

        def greet(greeting: str, name: str) -> str:
            """Greet someone."""
            return f"{greeting}, {name}"
        t = tool(functools.partial(greet, "hello"))
        self.assertEqual((t.name, t.description), ("greet", "Greet someone."))
        self.assertEqual(list(t.parameters["properties"]), ["name"])
        self.assertEqual(t.run(name="Ann"), "hello, Ann")

    def test_a_default_gives_the_type_where_there_is_no_annotation(self):
        def page(query, count=3, exact=False, tags=None):
            """Search."""
        props = tool(page).parameters["properties"]
        self.assertEqual([props[k]["type"] for k in ("query", "count", "exact", "tags")],
                         ["string", "integer", "boolean", "string"])

    def test_an_undefined_annotation_names_the_function(self):
        def broken(x: "Missing") -> str:      # noqa: F821
            """Broken."""
        with self.assertRaisesRegex(TypeError, "@tool broken"):
            tool(broken)

    def test_a_name_with_a_newline_is_refused(self):
        def ok(x: str) -> str:
            """Ok."""
        with self.assertRaises(ValueError):
            tool(ok, name="ok\n")


if __name__ == "__main__":
    unittest.main()
