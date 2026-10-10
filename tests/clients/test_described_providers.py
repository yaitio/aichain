"""
Z.ai, Meta, Mistral, Xiaomi, MiniMax and StepFun (4.1): providers described by data, on the OpenAI family.

None has been called live yet — the keys are still to come — so what is pinned
here is what each provider's documentation says the request must be, read
2026-10-11: the URL, the token field, how reasoning is asked for, and what
happens to a JSON Schema where the provider takes none. The live calls belong
beside these once there are keys.
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
        "MINIMAX_API_KEY": "k", "STEPFUN_API_KEY": "k"}


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
                               ("MiniMax-M3", "minimax"), ("step-5-preview", "stepfun")):
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
        _, _, body = _request("glm-5.3", {"reasoning": "none"})
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", body)

    def test_a_schema_rides_in_the_prompt_and_is_said(self):
        m, _, body = _request("glm-5.3", fmt=SCHEMA)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertIn("JSON Schema", body["messages"][0]["content"])
        self.assertIn(("adapted", "format"), [(a.kind, a.option) for a in m.last_adaptations])


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
class TestXiaomiAndStepFun(unittest.TestCase):

    def test_xiaomi_claims_no_reasoning_control(self):
        with self.assertRaisesRegex(ValueError, "reasoning for the 'xiaomi' provider"):
            Model("mimo-v2.6-pro", options={"reasoning": "high"})

    def test_stepfun_sends_a_level(self):
        _, _, body = _request("step-5-preview", {"reasoning": "low"})
        self.assertEqual(body["reasoning_effort"], "low")


if __name__ == "__main__":
    unittest.main()
