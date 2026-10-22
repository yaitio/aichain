"""
Z.ai, Meta, Mistral, Xiaomi, MiniMax, StepFun (4.0) and Inception (4.12): providers described by data,
on the OpenAI family.

What is pinned here is what each provider's documentation says the request
must be, read 2026-10-11: the URL, the token field, how reasoning is asked
for, and what happens to a JSON Schema where the provider takes none. Mistral
and Xiaomi were called live the same day, and what that found is pinned too:
Mistral's reasoning models answer with a list of parts, thinking among them,
and two of its models refuse a reasoning setting outright. Inception was
read and called live on 2026-10-21: reasoning as instant/low/medium/high,
temperature only within 0.5–1.0, no top_p.
"""

import os
import unittest
import warnings
from unittest.mock import patch

from yait_aichain import Model

MSGS = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
SCHEMA = {"type": "json_schema", "schema": {"type": "object", "properties": {"a": {"type": "string"}},
                                            "required": ["a"]}}
KEYS = {"ZAI_API_KEY": "k", "META_API_KEY": "k", "MISTRAL_API_KEY": "k", "MIMO_API_KEY": "k",
        "MINIMAX_API_KEY": "k", "STEPFUN_API_KEY": "k", "INCEPTION_API_KEY": "k"}


def _request(name, options=None, fmt=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Model(name, options=options)
        path, body = m.to_request(MSGS, {"format": fmt or {"type": "text"}})
    return m, path, body


@patch.dict(os.environ, KEYS)
class TestTheNameIsTheProvider(unittest.TestCase):

    def test_by_prefix(self):
        for name, provider in (("glm-5.3", "zai"), ("muse-spark-1.3", "meta"),
                               ("mistral-small-2603", "mistral"), ("ministral-14b-2512", "mistral"),
                               ("codestral-2508", "mistral"), ("mimo-v2.6-pro", "xiaomi"),
                               ("MiniMax-M3", "minimax"), ("step-5-preview", "stepfun"),
                               ("mercury-3", "inception")):
            with self.subTest(name):
                self.assertEqual(Model(name)._provider, provider)


@patch.dict(os.environ, KEYS)
class TestZai(unittest.TestCase):

    def test_the_path_is_its_own(self):
        m, path, body = _request("glm-5.3")
        self.assertEqual((m.client._base_url, path), ("https://api.z.ai/api/paas/v4", "/chat/completions"))
        self.assertIn("max_tokens", body)

    def test_a_level_turns_thinking_on_and_sets_its_depth(self):
        _, _, body = _request("glm-5.3", {"reasoning": "high"})
        self.assertEqual((body["thinking"], body["reasoning_effort"]), ({"type": "enabled"}, "high"))

    def test_none_turns_thinking_off(self):
        # glm-5.2: the 5.3 family always thinks — see TestZaiModelsThatAlwaysThink
        _, _, body = _request("glm-5.2", {"reasoning": "none"})
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", body)

    def test_a_schema_rides_in_the_prompt_and_is_said(self):
        m, _, body = _request("glm-5.3", fmt=SCHEMA)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertIn("JSON Schema", body["messages"][0]["content"])
        self.assertIn(("adapted", "format"), [(a.kind, a.option) for a in m.last_adaptations])


@patch.dict(os.environ, KEYS)
class TestZaiModelsThatAlwaysThink(unittest.TestCase):
    """GLM-5.3 and its Flash models answer 400 to thinking switched off
    ("always engages in thinking and cannot be disabled", 2026-10-11)."""

    def test_none_is_declined_not_sent(self):
        from yait_aichain.models._adaptation import DECLINED
        for name in ("glm-5.3", "glm-5.3-flash", "glm-5.3-flashx"):
            with self.subTest(name):
                m, _, body = _request(name, {"reasoning": "none"})
                self.assertNotIn("thinking", body)
                self.assertNotIn("reasoning_effort", body)
                self.assertEqual([a.kind for a in m.last_adaptations if a.option == "reasoning"],
                                 [DECLINED])

    def test_a_level_still_goes(self):
        _, _, body = _request("glm-5.3", {"reasoning": "high"})
        self.assertEqual((body["thinking"], body["reasoning_effort"]), ({"type": "enabled"}, "high"))

    def test_older_models_still_switch_off(self):
        _, _, body = _request("glm-5.2", {"reasoning": "none"})
        self.assertEqual(body["thinking"], {"type": "disabled"})


@patch.dict(os.environ, KEYS)
class TestMeta(unittest.TestCase):

    def test_the_token_field_and_reasoning(self):
        _, path, body = _request("muse-spark-1.3", {"reasoning": "medium"})
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual((body["max_completion_tokens"], body["reasoning_effort"]), (16384, "medium"))

    def test_temperature_without_top_p(self):
        _, _, body = _request("muse-spark-1.3")
        self.assertNotIn("top_p", body)

    def test_none_is_refused_before_the_network(self):
        with self.assertRaisesRegex(ValueError, "reasoning for the 'meta' provider"):
            Model("muse-spark-1.3", options={"reasoning": "none"})

    def test_a_schema_goes_as_a_schema(self):
        _, _, body = _request("muse-spark-1.3", fmt=SCHEMA)
        self.assertEqual(body["response_format"]["type"], "json_schema")


@patch.dict(os.environ, KEYS)
class TestMistral(unittest.TestCase):

    def test_reasoning_none_is_a_level(self):
        _, _, body = _request("mistral-small-2603", {"reasoning": "none"})
        self.assertEqual(body["reasoning_effort"], "none")

    def test_the_token_field(self):
        _, _, body = _request("mistral-small-2603")
        self.assertIn("max_tokens", body)

    def test_a_model_that_does_not_reason_is_not_asked_to(self):
        from yait_aichain.models._adaptation import DECLINED
        for name in ("codestral-2508", "ministral-14b-2512"):
            with self.subTest(name):
                m, _, body = _request(name, {"reasoning": "high"})
                self.assertNotIn("reasoning_effort", body)
                notes = [a for a in m.last_adaptations if a.option == "reasoning"]
                self.assertEqual([a.kind for a in notes], [DECLINED])

    def test_thinking_parts_stay_out_of_the_answer(self):
        m = Model("mistral-large-4-0")
        reply = {"choices": [{"message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": [{"type": "text", "text": "Paris, surely."}]},
            {"type": "text", "text": '{"a": "Paris"}'}]}, "finish_reason": "stop"}]}
        self.assertEqual(m.client.parse_response(reply, {"format": {"type": "text"}}), '{"a": "Paris"}')
        self.assertEqual(m.client.parse_response(reply, {"format": SCHEMA}), {"a": "Paris"})
        event = {"choices": [{"delta": {"content": [{"type": "thinking", "thinking": []},
                                                    {"type": "text", "text": "Par"}]}}]}
        self.assertEqual(m.client.parse_stream_event(event, {}), "Par")


@patch.dict(os.environ, KEYS)
class TestMiniMax(unittest.TestCase):

    def test_thinking_stays_out_of_the_answer(self):
        _, _, body = _request("MiniMax-M3")
        self.assertIs(body["reasoning_split"], True)

    def test_a_level_is_adaptive_thinking_and_none_turns_it_off(self):
        _, _, on = _request("MiniMax-M3", {"reasoning": "high"})
        _, _, off = _request("MiniMax-M3", {"reasoning": "none"})
        self.assertEqual((on["thinking"], off["thinking"]), ({"type": "adaptive"}, {"type": "disabled"}))
        self.assertNotIn("reasoning_effort", on)


@patch.dict(os.environ, KEYS)
class TestMiniMaxJSON(unittest.TestCase):
    """MiniMax takes a json_schema response_format and does not follow it â
    extra fields, and the JSON fenced in ```json (2026-10-11)."""

    def test_the_schema_rides_in_the_prompt(self):
        m, _, body = _request("MiniMax-M3", fmt=SCHEMA)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertIn("JSON Schema", body["messages"][0]["content"])

    def test_a_fenced_answer_is_read(self):
        m = Model("MiniMax-M3")
        reply = {"choices": [{"message": {"content": '```json\n{"a": "Paris"}\n```'},
                              "finish_reason": "stop"}]}
        self.assertEqual(m.client.parse_response(reply, {"format": SCHEMA}), {"a": "Paris"})

    def test_a_provider_that_takes_a_schema_is_not_loosened(self):
        from yait_aichain.clients._errors import InvalidStructuredOutputError
        m = Model("mistral-small-2603")
        reply = {"choices": [{"message": {"content": '```json\n{"a": "Paris"}\n```'},
                              "finish_reason": "stop"}]}
        with self.assertRaises(InvalidStructuredOutputError):
            m.client.parse_response(reply, {"format": SCHEMA})


@patch.dict(os.environ, KEYS)
class TestXiaomiAndStepFun(unittest.TestCase):

    def test_xiaomi_claims_no_reasoning_control(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, body, made = Model("mimo-v2.6-pro", options={"reasoning": "high"},
                                  api_key="k").build(MSGS, {"format": {"type": "text"}})
        self.assertNotIn("reasoning_effort", body)
        self.assertIn(("declined", "reasoning"), {(a.kind, a.option) for a in made})

    def test_mistral_large_4_sends_low_and_medium_as_none_and_says_so(self):
        for level, sent in (("low", "none"), ("medium", "none"), ("high", "high")):
            with self.subTest(level=level), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, body, made = Model("mistral-large-4-0", options={"reasoning": level},
                                      api_key="k").build(MSGS, {"format": {"type": "text"}})
                self.assertEqual(body["reasoning_effort"], sent)
                note = next(a for a in made if a.option == "reasoning")
                self.assertIn("none or high", note.why)

    def test_stepfun_sends_a_level(self):
        _, _, body = _request("step-5-preview", {"reasoning": "low"})
        self.assertEqual(body["reasoning_effort"], "low")


@patch.dict(os.environ, KEYS)
class TestInception(unittest.TestCase):

    def test_the_path_and_the_token_field(self):
        m, path, body = _request("mercury-3")
        self.assertEqual((m.client._base_url, path), ("https://api.inceptionlabs.ai", "/v1/chat/completions"))
        self.assertIn("max_completion_tokens", body)
        self.assertNotIn("top_p", body)

    def test_none_is_instant_and_a_level_is_itself(self):
        for level, sent in (("none", "instant"), ("low", "low"), ("high", "high")):
            with self.subTest(level=level):
                _, _, body = _request("mercury-3", {"reasoning": level})
                self.assertEqual(body["reasoning_effort"], sent)

    def test_temperature_outside_its_range_is_brought_in_and_said(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, body, made = Model("mercury-3", options={"temperature": 0.2},
                                  api_key="k").build(MSGS, {"format": {"type": "text"}})
        self.assertEqual(body["temperature"], 0.5)
        self.assertIn(("adapted", "temperature"), {(a.kind, a.option) for a in made})

    def test_top_p_is_declined_not_sent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, body, made = Model("mercury-3", options={"top_p": 0.9},
                                  api_key="k").build(MSGS, {"format": {"type": "text"}})
        self.assertNotIn("top_p", body)
        self.assertIn(("declined", "top_p"), {(a.kind, a.option) for a in made})



class TestInceptionLive(unittest.TestCase):
    """Mercury 3 called for real: text, a schema, a tool, and every reasoning level."""

    def setUp(self):
        if os.environ.get("INCEPTION_API_KEY", "") in ("", "k"):
            self.skipTest("INCEPTION_API_KEY not set to a real key")

    def test_text_and_its_cost(self):
        from yait_aichain import Skill
        s = Skill(Model("mercury-3"), prompt="Name the capital of Norway in one word.")
        self.assertIn("oslo", s.run().lower())
        self.assertGreater(s.last_usage.cost, 0)

    def test_a_schema_is_followed(self):
        from yait_aichain import Skill
        out = Skill(Model("mercury-3"), prompt="Give two planets with their order from the sun.",
                    output={"format": {"type": "json_schema", "schema": {
                        "type": "object", "additionalProperties": False, "required": ["planets"],
                        "properties": {"planets": {"type": "array", "items": {
                            "type": "object", "additionalProperties": False, "required": ["name", "order"],
                            "properties": {"name": {"type": "string"}, "order": {"type": "integer"}}}}}}}}).run()
        self.assertTrue(all(isinstance(p["order"], int) for p in out["planets"]))

    def test_an_agent_calls_a_tool(self):
        from yait_aichain import Agent
        from yait_aichain.tools import tool
        asked = []

        @tool(risk="read")
        def weather(city: str) -> str:
            """The weather in a city.

            city: the city.
            """
            asked.append(city)
            return f"{city}: 14°C, light rain"

        r = Agent(Model("mercury-3"), tools=[weather]).run("What is the weather in Kazan? One line.")
        self.assertEqual([c.lower() for c in asked], ["kazan"])
        self.assertIn("14", r.output)

    def test_every_reasoning_level_is_taken(self):
        from yait_aichain import Skill
        for level in ("none", "low", "medium", "high"):
            with self.subTest(level=level), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = Skill(Model("mercury-3", options={"reasoning": level}), prompt="2 + 2 = ? Digits only.")
                self.assertIn("4", s.run())


if __name__ == "__main__":
    unittest.main()
