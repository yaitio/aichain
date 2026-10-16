"""
Each Claude model is sent what its registry entry says it takes — structured
output, thinking, sampling — and never what the API refuses for it: no forced
tool_choice to a model that refuses one, no thinking budget to Claude 4.7 and
later, no sampling control where none is taken. A schema is adapted to what
structured output accepts, and the answer is still checked against the
caller's own.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain import Model, Skill
from yait_aichain.clients._errors import InvalidStructuredOutputError, TruncatedResponseError
from yait_aichain.models._data import PROVIDERS
from yait_aichain.testing import recorded

MSG = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
SCHEMA = {"type": "object", "properties": {"city": {"type": "string", "minLength": 3},
                                           "n": {"type": "integer", "maximum": 5}},
          "required": ["city"]}
JSON = {"modalities": ["text"], "format": {"type": "json_schema", "schema": SCHEMA}}
CLAUDE = sorted(PROVIDERS["anthropic"]["models"])


def _mech(name):
    from yait_aichain.clients._families.anthropic import _mechanisms
    return _mechanisms(PROVIDERS["anthropic"], name)


def _build(name, output=None, tools=None, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, body, made = Model(name, options=options, api_key="k").build(
            MSG, output or {"modalities": ["text"], "format": {"type": "text"}}, tools)
    return body, {(a.kind, a.option) for a in made}


class TestWhatEachModelIsSent(unittest.TestCase):

    def test_structured_output_follows_the_data(self):
        for name in CLAUDE:
            with self.subTest(model=name):
                body, _ = _build(name, JSON)
                choice = (body.get("tool_choice") or {}).get("type")
                if not _mech(name)["forced_tools"]:
                    self.assertNotIn(choice, ("tool", "any"))
                if _mech(name)["structured"] == "output_config":
                    self.assertEqual(body["output_config"]["format"]["type"], "json_schema")

    def test_reasoning_follows_the_data(self):
        for name in CLAUDE:
            for level in ("low", "medium", "high"):
                with self.subTest(model=name, level=level):
                    body, _ = _build(name, reasoning=level)
                    if _mech(name)["reasoning"] == "adaptive":
                        self.assertEqual(body["thinking"], {"type": "adaptive"})
                        self.assertEqual(body["output_config"]["effort"], level)
                    else:
                        self.assertEqual(body["thinking"]["type"], "enabled")

    def test_no_sampling_control_reaches_a_model_that_takes_none(self):
        for name in CLAUDE:
            if _mech(name)["sampling"] != "fixed":
                continue
            with self.subTest(model=name):
                body, said = _build(name, temperature=0.2, top_p=0.9, top_k=5)
                self.assertFalse({"temperature", "top_p", "top_k"} & set(body))
                self.assertLessEqual({("declined", "temperature"), ("declined", "top_p"),
                                      ("declined", "top_k")}, said)

    def test_turning_thinking_off_is_each_models_own_word(self):
        self.assertEqual(_build("claude-haiku-5-5", reasoning="none")[0]["thinking"],
                         {"type": "disabled"})
        self.assertEqual(_build("claude-sonnet-5-5", reasoning="none")[0]["thinking"],
                         {"type": "between_tools"})
        body, said = _build("claude-opus-5-5", reasoning="none")
        self.assertEqual(body["thinking"], {"type": "adaptive"})
        self.assertIn(("declined", "reasoning"), said)
        self.assertNotIn("thinking", _build("claude-sonnet-4-6", reasoning="none")[0])

    def test_an_alias_reads_its_dated_entry_and_an_unknown_name_the_newest(self):
        self.assertEqual(_mech("claude-sonnet-4-5")["reasoning"], "budget")
        self.assertEqual(_mech("claude-sonnet-5-5-20261201")["thinking_off"], "between_tools")
        self.assertEqual(_mech("claude-unreleased-9")["reasoning"], "adaptive")


class TestASchemaIsAdapted(unittest.TestCase):

    def test_unsupported_keywords_go_once_said_and_objects_close(self):
        body, said = _build("claude-sonnet-5-5", JSON)
        sent = body["output_config"]["format"]["schema"]
        self.assertNotIn("minLength", sent["properties"]["city"])
        self.assertNotIn("maximum", sent["properties"]["n"])
        self.assertIs(sent["additionalProperties"], False)
        self.assertIn(("adapted", "format.schema"), said)

    def test_the_answer_is_still_checked_against_the_callers_schema(self):
        model = Model("claude-sonnet-5-5", api_key="k")
        with recorded(model, reply='{"city": "Li", "n": 9}'), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(InvalidStructuredOutputError):
                Skill(model, prompt="x", output={"format": {"type": "json_schema", "schema": SCHEMA}}).run()

    def test_a_recursive_schema_is_refused_before_the_call(self):
        tree = {"type": "object", "properties": {"kids": {"type": "array", "items": {"$ref": "#/$defs/n"}}},
                "$defs": {"n": {"type": "object", "properties": {"kids": {"$ref": "#/$defs/n"}}}}}
        with self.assertRaisesRegex(ValueError, "recursive"):
            _build("claude-sonnet-5-5", {"format": {"type": "json_schema", "schema": tree}})

    def test_too_many_optional_properties_are_refused_naming_the_limit(self):
        wide = {"type": "object", "properties": {f"p{i}": {"type": "string"} for i in range(30)}}
        with self.assertRaisesRegex(ValueError, "at most 24 optional"):
            _build("claude-sonnet-5-5", {"format": {"type": "json_schema", "schema": wide}})


class TestCombinationsWork(unittest.TestCase):

    TOOL = {"type": "function", "function": {"name": "search", "description": "s",
                                             "parameters": {"type": "object"}}}

    def test_a_schema_with_tools_and_with_reasoning(self):
        body, _ = _build("claude-sonnet-5-5", JSON, tools=[self.TOOL], reasoning="low")
        self.assertEqual([t["name"] for t in body["tools"]], ["search"])
        self.assertEqual(set(body["output_config"]), {"format", "effort"})

    def test_a_forced_tool_model_still_refuses_both(self):
        from yait_aichain.clients._families import anthropic
        data = {**PROVIDERS["anthropic"], "models": {"old": {"structured": "forced_tool",
                                                             "forced_tools": True,
                                                             "reasoning": "budget"}}}
        with self.assertRaisesRegex(ValueError, "old cannot combine native tools"):
            anthropic._answer_format({}, JSON["format"], [self.TOOL], "old",
                                     anthropic._mechanisms(data, "old"))

    def test_a_forced_tool_model_that_refuses_forcing_gets_auto_strict_and_a_rule(self):
        from yait_aichain.clients._families import anthropic
        body = {}
        anthropic._answer_format(body, JSON["format"], None, "x",
                                 {"structured": "forced_tool", "forced_tools": False})
        self.assertEqual(body["tool_choice"], {"type": "auto"})
        self.assertTrue(body["tools"][0]["strict"])
        self.assertIn("structured_output", body["system"])


class TestTheAnswer(unittest.TestCase):

    def _parse(self, response):
        model = Model("claude-sonnet-5-5", api_key="k")
        return model.from_response(response, JSON)

    def test_read_from_the_text(self):
        self.assertEqual(self._parse({"content": [{"type": "text", "text": '{"city": "Lisbon"}'}],
                                      "stop_reason": "end_turn"}), {"city": "Lisbon"})

    def test_a_refusal_is_said_as_one(self):
        with self.assertRaisesRegex(InvalidStructuredOutputError, "refused"):
            self._parse({"content": [{"type": "text", "text": "I can't help"}], "stop_reason": "refusal"})

    def test_a_cut_off_answer_is_truncated_buffered_and_streamed(self):
        with self.assertRaises(TruncatedResponseError):
            self._parse({"content": [{"type": "text", "text": '{"city": "Lis'}], "stop_reason": "max_tokens",
                         "usage": {"output_tokens": 10}})
        from yait_aichain.skills._skill import Skill as _S
        from yait_aichain.models._base import StreamOutcome
        skill = _S(Model("claude-sonnet-5-5", api_key="k"), prompt="x",
                   output={"format": {"type": "json_schema", "schema": SCHEMA}})
        with self.assertRaises(TruncatedResponseError):
            skill._streamed_result(StreamOutcome(None, None, [], True), skill._output, ['{"city": "Li'], None)


class TestClaudeMechanismsLive(unittest.TestCase):
    """One structured call and one reasoning call per generation."""

    MODELS = ("claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-haiku-5-5",
              "claude-sonnet-5-5", "claude-opus-5-5")

    @unittest.skipUnless(os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set")
    def test_structured_and_reasoning_on_each(self):
        schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        for name in self.MODELS:
            with self.subTest(model=name), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = Skill(Model(name), prompt="City in: I live in Lisbon",
                            output={"format": {"type": "json_schema", "schema": schema}}).run()
                self.assertEqual(got["city"], "Lisbon")
                said = Skill(Model(name, options={"reasoning": "low"}), prompt="Say hi in two words.").run()
                self.assertTrue(said.strip())


if __name__ == "__main__":
    unittest.main()
