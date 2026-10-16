"""
A schema reaches Qwen where DashScope would drop it. Older models turn a
json_schema into json_object, which asks for "json" in the messages and is
HTTP 400 without it; beside tools, newer models answer in prose. Measured
2026-10-16: the schema in the prompt took 21 failures in 80 to 4.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain import Model, Skill

MSG = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
JSON = {"modalities": ["text"], "format": {"type": "json_schema", "schema": SCHEMA}}
TOOL = {"type": "function", "function": {"name": "lookup", "description": "Look up.",
                                         "parameters": {"type": "object"}}}


def _build(name, tools=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, body, made = Model(name, api_key="k").build(MSG, JSON, tools)
    return body, [(a.kind, a.option, a.sent) for a in made if a.option == "format"]


def _prompt(body):
    return " ".join(m["content"] for m in body["messages"]
                    if m["role"] == "system" and isinstance(m["content"], str))


class TestOlderModels(unittest.TestCase):

    def test_json_object_with_the_schema_in_the_prompt(self):
        for name in ("qwen3.6-plus", "qwen3.5-plus", "qwen3.6-flash", "qwen-max"):
            for tools in (None, [TOOL]):
                with self.subTest(model=name, tools=bool(tools)):
                    body, said = _build(name, tools)
                    self.assertEqual(body["response_format"], {"type": "json_object"})
                    self.assertIn('"city"', _prompt(body))
                    self.assertIn("JSON", _prompt(body))
                    self.assertEqual(said, [("adapted", "format", "json_object")])


class TestNewerModels(unittest.TestCase):

    def test_without_tools_the_schema_goes_as_it_is(self):
        body, said = _build("qwen3.7-flash")
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(_prompt(body), "")
        self.assertEqual(said, [])

    def test_beside_tools_the_schema_is_in_the_prompt_too(self):
        body, said = _build("qwen3.7-flash", [TOOL])
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertIn('"city"', _prompt(body))
        self.assertEqual(said, [("adapted", "format", "json_schema + prompt")])


class TestQwenSchemaLive(unittest.TestCase):
    """Two calls per model generation, about $0.001."""

    @unittest.skipUnless(os.getenv("DASHSCOPE_API_KEY"), "DASHSCOPE_API_KEY not set")
    def test_an_older_model_answers_a_schema(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name in ("qwen3.6-flash", "qwen-max"):
                with self.subTest(model=name):
                    got = Skill(Model(name), prompt="City in: I live in Lisbon",
                                output={"format": {"type": "json_schema", "schema": SCHEMA}}).run()
                    self.assertEqual(got["city"], "Lisbon")


if __name__ == "__main__":
    unittest.main()
