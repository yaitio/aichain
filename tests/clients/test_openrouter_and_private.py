"""
OpenRouter is a provider of its own, a data file with no model list; private
takes reasoning in the one spelling the servers behind it share and says the
server decides; a provider with no reasoning control declines a level rather
than refusing the model.
"""

import json
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain import Model, Skill
from yait_aichain.models._calls import ToolCallRequest

MSG = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
TEXT = {"modalities": ["text"], "format": {"type": "text"}}


def _build(name, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, body, made = Model(name, options=options, api_key="k").build(MSG, TEXT)
    return body, {a.option: a for a in made}


class TestOpenRouter(unittest.TestCase):

    def test_the_name_after_the_prefix_goes_as_written(self):
        model = Model("openrouter/anthropic/claude-sonnet-5.5", api_key="k")
        path, body = model.to_request(MSG, TEXT)
        self.assertEqual((model._provider, body["model"], path),
                         ("openrouter", "anthropic/claude-sonnet-5.5", "/v1/chat/completions"))

    def test_reasoning_goes_as_its_effort_and_none_turns_it_off(self):
        body, said = _build("openrouter/qwen/qwen3.8-flash", reasoning="high")
        self.assertEqual(body["reasoning"], {"effort": "high"})
        self.assertEqual(said["reasoning"].kind, "adapted")
        self.assertEqual(_build("openrouter/x", reasoning="none")[0]["reasoning"], {"effort": "none"})

    def test_top_k_goes_where_the_data_claims_it(self):
        self.assertEqual(_build("openrouter/x", top_k=5)[0]["top_k"], 5)
        self.assertNotIn("top_k", _build("private/x", top_k=5)[0])

    def test_the_cost_is_the_bill_in_the_response(self):
        from yait_aichain.models._usage import attach_cost, extract_usage
        reply = {"usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.0000123}}
        usage = attach_cost(extract_usage(reply), "anthropic/claude-sonnet-5.5")
        self.assertAlmostEqual(usage.cost, 0.0000123)

    def test_the_thinking_beside_a_call_is_kept_from_its_new_field(self):
        model = Model("private/qwen3", api_key="k")
        got = model.from_response({"choices": [{"finish_reason": "tool_calls", "message": {
            "content": None, "reasoning": "need the weather",
            "tool_calls": [{"id": "c1", "function": {"name": "w", "arguments": "{}"}}]}}]}, TEXT)
        self.assertIsInstance(got, ToolCallRequest)
        self.assertEqual(got.opaque, {"reasoning_content": "need the weather"})


class TestPrivate(unittest.TestCase):

    def test_reasoning_goes_as_reasoning_effort_and_says_the_server_decides(self):
        body, said = _build("private/qwen3", reasoning="low")
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertIn("extra=", said["reasoning"].why)
        self.assertEqual(_build("private/qwen3", reasoning="none")[0]["reasoning_effort"], "none")

    def test_the_servers_own_field_goes_through_extra(self):
        body, _ = _build("private/deepseek-v3.1",
                         extra={"chat_template_kwargs": {"thinking": True}})
        self.assertEqual(body["chat_template_kwargs"], {"thinking": True})


class TestNoReasoningControl(unittest.TestCase):

    def test_a_level_is_declined_not_refused(self):
        for name in ("sonar", "mimo-v2.6-pro"):
            with self.subTest(model=name):
                body, said = _build(name, reasoning="high")
                self.assertNotIn("reasoning_effort", body)
                self.assertEqual(said["reasoning"].kind, "declined")

    def test_a_typo_is_still_refused(self):
        with self.assertRaisesRegex(ValueError, "none, low, medium, high"):
            Model("sonar", options={"reasoning": "hgh"}, api_key="k")


class TestOpenRouterLive(unittest.TestCase):
    """Four calls on two vendors behind OpenRouter: about $0.001."""

    @unittest.skipUnless(os.getenv("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY not set")
    def test_a_schema_and_reasoning_with_a_cost(self):
        schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        for name in ("openrouter/qwen/qwen3.8-flash", "openrouter/anthropic/claude-haiku-5.5"):
            with self.subTest(model=name), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skill = Skill(Model(name, options={"reasoning": "low"}),
                              prompt="City in: I live in Lisbon",
                              output={"format": {"type": "json_schema", "schema": schema}})
                self.assertEqual(skill.run()["city"], "Lisbon")
                self.assertGreater(skill.last_usage.cost, 0)


if __name__ == "__main__":
    unittest.main()
