"""
tests.models.test_models_features
====================================

Tests for cross-cutting model features:
  1. Reasoning parameter translation per provider
  2. JSON / json_schema output format in request body
  3. Cache control (Anthropic)
  4. Registry query helpers
  5. _detect_image_mime  helper
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_TEST_KEYS = {
    "OPENAI_API_KEY":    "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "GOOGLE_AI_API_KEY": "test-google-key",
    "XAI_API_KEY":       "test-xai-key",
    "PERPLEXITY_API_KEY":"test-perplexity-key",
    "DEEPSEEK_API_KEY":  "test-deepseek-key",
    "MOONSHOT_API_KEY":  "test-kimi-key",
    "DASHSCOPE_API_KEY": "test-qwen-key",
}
for _k, _v in _TEST_KEYS.items():
    if not os.environ.get(_k):
        os.environ[_k] = _v

from models import (
    OpenAIModel, AnthropicModel, GoogleAIModel, XAIModel,
    PerplexityModel, DeepSeekModel, KimiModel, QwenModel,
)
import models._registry as registry

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_USER_MSG    = [{"role": "user", "parts": [{"type": "text", "text": "Hello"}]}]
_TEXT_OUTPUT = {"modalities": ["text"], "format": {"type": "text"}}

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name":  {"type": "string"},
        "score": {"type": "integer"},
    },
    "required": ["name", "score"],
}


# ---------------------------------------------------------------------------
# 1. Reasoning parameter translation
# ---------------------------------------------------------------------------

class TestOpenAIReasoning(unittest.TestCase):

    def test_reasoning_high_adds_reasoning_effort(self):
        model = OpenAIModel("gpt-4o", options={"reasoning": "high"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body.get("reasoning_effort"), "high")

    def test_reasoning_low_adds_reasoning_effort_low(self):
        model = OpenAIModel("gpt-4o", options={"reasoning": "low"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body.get("reasoning_effort"), "low")

    def test_reasoning_none_omits_reasoning_effort(self):
        model = OpenAIModel("gpt-4o")
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertNotIn("reasoning_effort", body)


class TestAnthropicReasoning(unittest.TestCase):

    def test_reasoning_high_adds_thinking_block(self):
        model = AnthropicModel("claude-opus-4-6", options={"reasoning": "high"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertIn("thinking", body)
        self.assertEqual(body["thinking"]["type"], "enabled")
        self.assertEqual(body["thinking"]["budget_tokens"], 20000)

    def test_reasoning_medium_budget(self):
        model = AnthropicModel("claude-opus-4-6", options={"reasoning": "medium"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["thinking"]["budget_tokens"], 10000)

    def test_reasoning_low_budget(self):
        model = AnthropicModel("claude-opus-4-6", options={"reasoning": "low"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["thinking"]["budget_tokens"], 4000)

    def test_reasoning_forces_temperature_to_1(self):
        model = AnthropicModel("claude-opus-4-6", options={"reasoning": "high", "temperature": 0.3})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["temperature"], 1.0)

    def test_no_reasoning_omits_thinking(self):
        model = AnthropicModel("claude-opus-4-6")
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertNotIn("thinking", body)


class TestGoogleReasoning(unittest.TestCase):

    def test_reasoning_high_adds_thinking_config(self):
        model = GoogleAIModel("gemini-2.5-pro", options={"reasoning": "high"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        tc = body["generationConfig"].get("thinkingConfig", {})
        self.assertEqual(tc.get("thinkingBudget"), 24576)

    def test_reasoning_medium_budget(self):
        model = GoogleAIModel("gemini-2.5-pro", options={"reasoning": "medium"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        tc = body["generationConfig"].get("thinkingConfig", {})
        self.assertEqual(tc.get("thinkingBudget"), 8192)

    def test_reasoning_low_budget(self):
        model = GoogleAIModel("gemini-2.5-pro", options={"reasoning": "low"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        tc = body["generationConfig"].get("thinkingConfig", {})
        self.assertEqual(tc.get("thinkingBudget"), 2048)

    def test_no_reasoning_omits_thinking_config(self):
        model = GoogleAIModel("gemini-2.5-flash")
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertNotIn("thinkingConfig", body.get("generationConfig", {}))


class TestXAIReasoning(unittest.TestCase):

    def test_reasoning_high_maps_to_high(self):
        model = XAIModel("grok-3-mini", options={"reasoning": "high"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body.get("reasoning_effort"), "high")

    def test_reasoning_medium_maps_to_high(self):
        # xAI has no "medium"; it maps to "high"
        model = XAIModel("grok-3-mini", options={"reasoning": "medium"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body.get("reasoning_effort"), "high")

    def test_reasoning_low_maps_to_low(self):
        model = XAIModel("grok-3-mini", options={"reasoning": "low"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body.get("reasoning_effort"), "low")


class TestDeepSeekReasoning(unittest.TestCase):

    def test_reasoning_high_switches_model_to_reasoner(self):
        model = DeepSeekModel("deepseek-chat", options={"reasoning": "high"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["model"], "deepseek-reasoner")

    def test_reasoning_low_keeps_chat_model(self):
        model = DeepSeekModel("deepseek-chat", options={"reasoning": "low"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["model"], "deepseek-chat")

    def test_reasoning_medium_keeps_chat_model(self):
        model = DeepSeekModel("deepseek-chat", options={"reasoning": "medium"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["model"], "deepseek-chat")


class TestKimiReasoning(unittest.TestCase):

    def test_any_reasoning_level_enables_thinking(self):
        for level in ("low", "medium", "high"):
            with self.subTest(level=level):
                model = KimiModel("kimi-k2-0905-preview", options={"reasoning": level})
                _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
                self.assertEqual(body.get("thinking"), {"type": "enabled"})

    def test_reasoning_forces_temperature_to_1(self):
        model = KimiModel("kimi-k2-0905-preview", options={"reasoning": "high", "temperature": 0.6})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertEqual(body["temperature"], 1.0)

    def test_no_reasoning_omits_thinking(self):
        model = KimiModel("kimi-k2-0905-preview")
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertNotIn("thinking", body)


class TestQwenReasoning(unittest.TestCase):

    def test_qwq_always_has_enable_thinking(self):
        model = QwenModel("QwQ-32B")
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertTrue(body.get("enable_thinking"))

    def test_qwen3_with_reasoning_has_enable_thinking(self):
        model = QwenModel("qwen3-72b", options={"reasoning": "high"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertTrue(body.get("enable_thinking"))

    def test_qwen3_without_reasoning_no_enable_thinking(self):
        model = QwenModel("qwen3-72b")
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertNotIn("enable_thinking", body)

    def test_qwen_max_never_has_enable_thinking(self):
        model = QwenModel("qwen-max")
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertNotIn("enable_thinking", body)


class TestPerplexityReasoning(unittest.TestCase):

    def test_reasoning_silently_ignored(self):
        # Perplexity has no reasoning parameter; setting it must not crash.
        model = PerplexityModel("sonar", options={"reasoning": "high"})
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertNotIn("reasoning_effort", body)
        self.assertNotIn("thinking", body)


# ---------------------------------------------------------------------------
# 2. JSON / json_schema output format
# ---------------------------------------------------------------------------

class TestOpenAIJsonOutput(unittest.TestCase):

    def test_json_format_sets_response_format(self):
        model  = OpenAIModel("gpt-4o")
        output = {"modalities": ["text"], "format": {"type": "json"}}
        _, body = model.to_request(_USER_MSG, output)
        self.assertEqual(body["response_format"]["type"], "json_object")

    def test_json_schema_format_sets_schema(self):
        model  = OpenAIModel("gpt-4o")
        output = {"modalities": ["text"], "format": {"type": "json_schema", "schema": _JSON_SCHEMA}}
        _, body = model.to_request(_USER_MSG, output)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertIn("json_schema", body["response_format"])

    def test_text_format_omits_response_format(self):
        model = OpenAIModel("gpt-4o")
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertNotIn("response_format", body)

    def test_json_from_response_parses_to_dict(self):
        model  = OpenAIModel("gpt-4o")
        output = {"modalities": ["text"], "format": {"type": "json"}}
        resp   = {"choices": [{"message": {"content": '{"name": "Alice", "score": 9}'}}]}
        result = model.from_response(resp, output)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], "Alice")
        self.assertEqual(result["score"], 9)


class TestAnthropicJsonOutput(unittest.TestCase):

    def test_json_schema_adds_tool_and_tool_choice(self):
        model  = AnthropicModel("claude-opus-4-6")
        output = {"modalities": ["text"], "format": {"type": "json_schema", "schema": _JSON_SCHEMA}}
        _, body = model.to_request(_USER_MSG, output)
        self.assertIn("tools", body)
        self.assertIn("tool_choice", body)
        self.assertEqual(body["tool_choice"]["type"], "tool")

    def test_json_schema_from_response_returns_input_dict(self):
        model  = AnthropicModel("claude-opus-4-6")
        output = {"modalities": ["text"], "format": {"type": "json_schema", "schema": _JSON_SCHEMA}}
        resp   = {
            "content": [
                {"type": "tool_use", "name": "structured_output", "input": {"name": "Bob", "score": 7}},
            ]
        }
        result = model.from_response(resp, output)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], "Bob")

    def test_text_format_no_tools_in_body(self):
        model = AnthropicModel("claude-opus-4-6")
        _, body = model.to_request(_USER_MSG, _TEXT_OUTPUT)
        self.assertNotIn("tools", body)


class TestGoogleJsonOutput(unittest.TestCase):

    def test_json_sets_response_mime_type(self):
        model  = GoogleAIModel("gemini-2.5-flash")
        output = {"modalities": ["text"], "format": {"type": "json"}}
        _, body = model.to_request(_USER_MSG, output)
        self.assertEqual(
            body["generationConfig"].get("responseMimeType"),
            "application/json",
        )

    def test_json_schema_sets_response_schema(self):
        model  = GoogleAIModel("gemini-2.5-flash")
        output = {"modalities": ["text"], "format": {"type": "json_schema", "schema": _JSON_SCHEMA}}
        _, body = model.to_request(_USER_MSG, output)
        gc = body["generationConfig"]
        self.assertEqual(gc.get("responseMimeType"), "application/json")
        self.assertIn("responseSchema", gc)

    def test_google_schema_sanitisation_strips_additional_properties(self):
        from models._google import _sanitize_google_schema
        schema = {"type": "object", "additionalProperties": False, "properties": {}}
        result = _sanitize_google_schema(schema)
        self.assertNotIn("additionalProperties", result)

    def test_google_schema_sanitisation_handles_nullable_types(self):
        from models._google import _sanitize_google_schema
        schema = {"type": ["string", "null"]}
        result = _sanitize_google_schema(schema)
        self.assertEqual(result["type"], "string")
        self.assertTrue(result.get("nullable"))


class TestKimiJsonOutput(unittest.TestCase):

    def test_json_schema_adds_response_format(self):
        model  = KimiModel("kimi-k2-0905-preview")
        output = {"modalities": ["text"], "format": {"type": "json_schema", "schema": _JSON_SCHEMA}}
        _, body = model.to_request(_USER_MSG, output)
        self.assertIn("response_format", body)
        self.assertEqual(body["response_format"]["type"], "json_schema")

    def test_json_format_sets_json_object(self):
        model  = KimiModel("kimi-k2-0905-preview")
        output = {"modalities": ["text"], "format": {"type": "json"}}
        _, body = model.to_request(_USER_MSG, output)
        self.assertEqual(body["response_format"]["type"], "json_object")


class TestDeepSeekJsonOutput(unittest.TestCase):

    def test_json_sets_response_format_json_object(self):
        model  = DeepSeekModel("deepseek-chat")
        output = {"modalities": ["text"], "format": {"type": "json"}}
        _, body = model.to_request(_USER_MSG, output)
        self.assertEqual(body["response_format"]["type"], "json_object")

    def test_json_schema_falls_back_to_json_object(self):
        # DeepSeek does not support json_schema natively; falls back to json_object.
        model  = DeepSeekModel("deepseek-chat")
        output = {"modalities": ["text"], "format": {"type": "json_schema", "schema": _JSON_SCHEMA}}
        _, body = model.to_request(_USER_MSG, output)
        self.assertEqual(body["response_format"]["type"], "json_object")


# ---------------------------------------------------------------------------
# 3. Registry helpers
# ---------------------------------------------------------------------------

class TestRegistry(unittest.TestCase):

    def test_providers_returns_all_eight(self):
        all_providers = registry.providers()
        for p in ("openai", "anthropic", "google", "xai", "perplexity", "kimi", "deepseek", "qwen"):
            self.assertIn(p, all_providers)

    def test_providers_text_to_image_excludes_anthropic_perplexity_deepseek(self):
        image_providers = registry.providers(task="text-to-image")
        self.assertNotIn("anthropic",  image_providers)
        self.assertNotIn("perplexity", image_providers)
        self.assertNotIn("deepseek",   image_providers)

    def test_models_anthropic_text_to_text(self):
        names = registry.models(provider="anthropic", task="text-to-text")
        self.assertGreater(len(names), 0)
        for n in names:
            self.assertTrue(n.startswith("claude-"))

    def test_models_openai_text_to_image(self):
        names = registry.models(provider="openai", task="text-to-image")
        self.assertGreater(len(names), 0)

    def test_models_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            registry.models(provider="unknown-provider")

    def test_models_unknown_task_raises(self):
        with self.assertRaises(ValueError):
            registry.models(task="text-to-audio")

    def test_tasks_gpt4o_returns_text_and_vision(self):
        t = registry.tasks("gpt-4o")
        self.assertIn("text-to-text",  t)
        self.assertIn("image-to-text", t)

    def test_tasks_gpt_image_returns_only_image_generation(self):
        t = registry.tasks("gpt-image-1")
        self.assertIn("text-to-image", t)
        self.assertNotIn("text-to-text", t)

    def test_tasks_unknown_model_returns_empty_list(self):
        self.assertEqual(registry.tasks("no-such-model-xyz"), [])

    def test_is_supported_known_model(self):
        self.assertTrue(registry.is_supported("gpt-4o"))

    def test_is_supported_with_matching_task(self):
        self.assertTrue(registry.is_supported("gpt-4o", "text-to-text"))

    def test_is_supported_with_wrong_task(self):
        self.assertFalse(registry.is_supported("gpt-image-1", "text-to-text"))

    def test_is_supported_unknown_model(self):
        self.assertFalse(registry.is_supported("no-such-model-xyz"))

    def test_is_supported_invalid_task_raises(self):
        with self.assertRaises(ValueError):
            registry.is_supported("gpt-4o", "text-to-audio")

    def test_tasks_constant_has_three_entries(self):
        self.assertEqual(len(registry.TASKS), 3)

    def test_providers_constant_has_eight_entries(self):
        self.assertEqual(len(registry.PROVIDERS), 8)


# ---------------------------------------------------------------------------
# 4. _detect_image_mime helper
# ---------------------------------------------------------------------------

class TestDetectImageMime(unittest.TestCase):

    def setUp(self):
        from models._openai import _detect_image_mime
        self._detect = _detect_image_mime

    def _b64(self, raw: bytes) -> str:
        import base64
        return base64.b64encode(raw).decode()

    def test_png_detected(self):
        header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 4
        self.assertEqual(self._detect(self._b64(header)), "image/png")

    def test_jpeg_detected(self):
        header = b"\xff\xd8\xff\xe0" + b"\x00" * 8
        self.assertEqual(self._detect(self._b64(header)), "image/jpeg")

    def test_webp_detected(self):
        header = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 4
        self.assertEqual(self._detect(self._b64(header)), "image/webp")

    def test_gif_detected(self):
        header = b"GIF89a" + b"\x00" * 10
        self.assertEqual(self._detect(self._b64(header)), "image/gif")

    def test_unknown_fallback_is_png(self):
        header = b"\x00\x01\x02\x03" * 4
        self.assertEqual(self._detect(self._b64(header)), "image/png")

    def test_none_fallback_is_png(self):
        self.assertEqual(self._detect(None), "image/png")

    def test_empty_string_fallback_is_png(self):
        self.assertEqual(self._detect(""), "image/png")


if __name__ == "__main__":
    unittest.main()
