"""
tests.skills.test_adapters
===========================

Unit tests for the format utilities in ``skills._adapters``
(``validate_input``, ``validate_output``, ``substitute``) and for the
provider-specific ``to_request`` / ``from_response`` methods that now live
directly on each model class.

All tests are pure — no network calls, no real API keys.
"""

import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Set fake keys so model constructors don't raise
_TEST_KEYS = {
    "OPENAI_API_KEY":     "test-openai-key",
    "ANTHROPIC_API_KEY":  "test-anthropic-key",
    "GOOGLE_AI_API_KEY":  "test-google-key",
    "XAI_API_KEY":        "test-xai-key",
    "PERPLEXITY_API_KEY": "test-perplexity-key",
}
for _k, _v in _TEST_KEYS.items():
    if not os.environ.get(_k):
        os.environ[_k] = _v

from skills._adapters import validate_input, validate_output, substitute
from models import OpenAIModel, AnthropicModel, GoogleAIModel, XAIModel, PerplexityModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_output():
    return {"modalities": ["text"], "format": {"type": "text"}}

def _json_output():
    return {"modalities": ["text"], "format": {"type": "json"}}


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------

class TestValidateInput(unittest.TestCase):

    def _ok(self):
        return {
            "messages": [
                {"role": "user", "parts": [{"type": "text", "text": "hello"}]}
            ]
        }

    def test_valid_text_message(self):
        validate_input(self._ok())

    def test_valid_multipart_message(self):
        validate_input({
            "messages": [
                {
                    "role": "user",
                    "parts": [
                        {"type": "text", "text": "look at this"},
                        {"type": "image", "source": {"kind": "url", "url": "https://x.com/img.png"}},
                    ],
                }
            ]
        })

    def test_valid_all_roles(self):
        for role in ("system", "user", "assistant"):
            validate_input({
                "messages": [{"role": role, "parts": [{"type": "text", "text": "hi"}]}]
            })

    def test_valid_audio_base64(self):
        validate_input({
            "messages": [{
                "role": "user",
                "parts": [
                    {"type": "audio", "source": {"kind": "base64", "mime": "audio/wav", "data": "abc"}},
                ]
            }]
        })

    def test_valid_video_url(self):
        validate_input({
            "messages": [{
                "role": "user",
                "parts": [
                    {"type": "video", "source": {"kind": "url", "url": "https://example.com/v.mp4"}},
                ]
            }]
        })

    def test_not_a_dict(self):
        with self.assertRaises(ValueError):
            validate_input([])

    def test_missing_messages(self):
        with self.assertRaises(ValueError):
            validate_input({})

    def test_empty_messages(self):
        with self.assertRaises(ValueError):
            validate_input({"messages": []})

    def test_invalid_role(self):
        with self.assertRaises(ValueError):
            validate_input({
                "messages": [{"role": "bot", "parts": [{"type": "text", "text": "hi"}]}]
            })

    def test_missing_parts(self):
        with self.assertRaises(ValueError):
            validate_input({"messages": [{"role": "user"}]})

    def test_empty_parts(self):
        with self.assertRaises(ValueError):
            validate_input({"messages": [{"role": "user", "parts": []}]})

    def test_invalid_part_type(self):
        with self.assertRaises(ValueError):
            validate_input({
                "messages": [{"role": "user", "parts": [{"type": "file", "text": "x"}]}]
            })

    def test_text_part_missing_text(self):
        with self.assertRaises(ValueError):
            validate_input({
                "messages": [{"role": "user", "parts": [{"type": "text"}]}]
            })

    def test_text_part_non_string(self):
        with self.assertRaises(ValueError):
            validate_input({
                "messages": [{"role": "user", "parts": [{"type": "text", "text": 42}]}]
            })

    def test_media_part_missing_source(self):
        with self.assertRaises(ValueError):
            validate_input({
                "messages": [{"role": "user", "parts": [{"type": "image"}]}]
            })

    def test_media_part_invalid_kind(self):
        with self.assertRaises(ValueError):
            validate_input({
                "messages": [{
                    "role": "user",
                    "parts": [{"type": "image", "source": {"kind": "s3"}}]
                }]
            })


# ---------------------------------------------------------------------------
# validate_output
# ---------------------------------------------------------------------------

class TestValidateOutput(unittest.TestCase):

    def test_valid_text(self):
        validate_output({"modalities": ["text"], "format": {"type": "text"}})

    def test_valid_json(self):
        validate_output({"modalities": ["text"], "format": {"type": "json"}})

    def test_valid_json_schema(self):
        validate_output({
            "modalities": ["text"],
            "format": {
                "type":   "json_schema",
                "name":   "result",
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
            },
        })

    def test_not_a_dict(self):
        with self.assertRaises(ValueError):
            validate_output("text")

    def test_missing_modalities(self):
        with self.assertRaises(ValueError):
            validate_output({"format": {"type": "text"}})

    def test_empty_modalities(self):
        with self.assertRaises(ValueError):
            validate_output({"modalities": [], "format": {"type": "text"}})

    def test_missing_format(self):
        with self.assertRaises(ValueError):
            validate_output({"modalities": ["text"]})

    def test_invalid_format_type(self):
        with self.assertRaises(ValueError):
            validate_output({"modalities": ["text"], "format": {"type": "xml"}})

    def test_json_schema_missing_schema(self):
        with self.assertRaises(ValueError):
            validate_output({
                "modalities": ["text"],
                "format": {"type": "json_schema"},
            })

    def test_json_schema_non_dict_schema(self):
        with self.assertRaises(ValueError):
            validate_output({
                "modalities": ["text"],
                "format": {"type": "json_schema", "schema": "string"},
            })


# ---------------------------------------------------------------------------
# substitute
# ---------------------------------------------------------------------------

class TestSubstitute(unittest.TestCase):

    def _msgs(self):
        return [
            {"role": "system", "parts": [{"type": "text", "text": "You are {persona}."}]},
            {"role": "user",   "parts": [{"type": "text", "text": "Tell me about {topic}."}]},
        ]

    def test_basic_substitution(self):
        result = substitute(self._msgs(), {"persona": "Alice", "topic": "gravity"})
        self.assertEqual(result[0]["parts"][0]["text"], "You are Alice.")
        self.assertEqual(result[1]["parts"][0]["text"], "Tell me about gravity.")

    def test_original_unchanged(self):
        original = self._msgs()
        substitute(original, {"persona": "Bob", "topic": "light"})
        self.assertIn("{persona}", original[0]["parts"][0]["text"])

    def test_empty_variables(self):
        msgs    = self._msgs()
        result  = substitute(msgs, {})
        self.assertEqual(result[0]["parts"][0]["text"], "You are {persona}.")

    def test_none_variables_treated_as_empty(self):
        msgs   = self._msgs()
        result = substitute(msgs, {})
        self.assertEqual(result[1]["parts"][0]["text"], "Tell me about {topic}.")

    def test_partial_substitution(self):
        result = substitute(self._msgs(), {"topic": "mars"})
        self.assertIn("{persona}", result[0]["parts"][0]["text"])
        self.assertEqual(result[1]["parts"][0]["text"], "Tell me about mars.")

    def test_media_parts_deep_copied(self):
        msgs = [{"role": "user", "parts": [
            {"type": "image", "source": {"kind": "url", "url": "https://x.com/img.png"}},
        ]}]
        result = substitute(msgs, {"x": "y"})
        self.assertEqual(result[0]["parts"][0]["source"]["url"], "https://x.com/img.png")
        result[0]["parts"][0]["source"]["url"] = "changed"
        self.assertEqual(msgs[0]["parts"][0]["source"]["url"], "https://x.com/img.png")

    def test_multiple_parts_mixed(self):
        msgs = [{"role": "user", "parts": [
            {"type": "text",  "text": "Look at {thing}:"},
            {"type": "image", "source": {"kind": "url", "url": "https://x.com/img.png"}},
        ]}]
        result = substitute(msgs, {"thing": "this image"})
        self.assertEqual(result[0]["parts"][0]["text"], "Look at this image:")
        self.assertEqual(result[0]["parts"][1]["type"], "image")


# ---------------------------------------------------------------------------
# OpenAIModel.to_request / from_response
# ---------------------------------------------------------------------------

class TestOpenAIToRequest(unittest.TestCase):

    def _model(self, **kw):
        m = OpenAIModel("gpt-4o", options=kw if kw else None)
        return m

    def test_path(self):
        path, _ = self._model().to_request([], _text_output())
        self.assertEqual(path, "/v1/chat/completions")

    def test_model_name_in_body(self):
        _, body = self._model().to_request([], _text_output())
        self.assertEqual(body["model"], "gpt-4o")

    def test_max_completion_tokens(self):
        _, body = self._model(max_tokens=2048).to_request([], _text_output())
        self.assertEqual(body["max_completion_tokens"], 2048)
        self.assertNotIn("max_tokens", body)

    def test_temperature_included(self):
        _, body = self._model(temperature=0.5).to_request([], _text_output())
        self.assertEqual(body["temperature"], 0.5)

    def test_top_p_included_when_set(self):
        _, body = self._model(top_p=0.9).to_request([], _text_output())
        self.assertEqual(body["top_p"], 0.9)

    def test_top_p_omitted_when_none(self):
        m = OpenAIModel("gpt-4o")
        m.top_p = None
        _, body = m.to_request([], _text_output())
        self.assertNotIn("top_p", body)

    def test_reasoning_effort_included(self):
        _, body = self._model(reasoning="high").to_request([], _text_output())
        self.assertEqual(body["reasoning_effort"], "high")

    def test_reasoning_effort_omitted_when_none(self):
        _, body = self._model().to_request([], _text_output())
        self.assertNotIn("reasoning_effort", body)

    def test_text_part_converted(self):
        msgs = [{"role": "user", "parts": [{"type": "text", "text": "hello"}]}]
        _, body = self._model().to_request(msgs, _text_output())
        self.assertEqual(body["messages"][0]["content"], "hello")

    def test_system_role_preserved(self):
        msgs = [
            {"role": "system", "parts": [{"type": "text", "text": "Be brief."}]},
            {"role": "user",   "parts": [{"type": "text", "text": "hi"}]},
        ]
        _, body = self._model().to_request(msgs, _text_output())
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][0]["content"], "Be brief.")

    def test_image_url_part_converted(self):
        msgs = [{"role": "user", "parts": [
            {"type": "image", "source": {"kind": "url", "url": "https://x.com/img.png"}},
        ]}]
        _, body = self._model().to_request(msgs, _text_output())
        content = body["messages"][0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[0]["image_url"]["url"], "https://x.com/img.png")

    def test_image_base64_part_converted(self):
        msgs = [{"role": "user", "parts": [
            {"type": "image", "source": {"kind": "base64", "mime": "image/jpeg", "data": "abc123"}},
        ]}]
        _, body = self._model().to_request(msgs, _text_output())
        content = body["messages"][0]["content"]
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_video_part_skipped(self):
        msgs = [{"role": "user", "parts": [
            {"type": "video", "source": {"kind": "url", "url": "https://x.com/v.mp4"}},
        ]}]
        _, body = self._model().to_request(msgs, _text_output())
        self.assertEqual(body["messages"], [])

    def test_json_output_format(self):
        _, body = self._model().to_request([], _json_output())
        self.assertEqual(body["response_format"]["type"], "json_object")

    def test_json_schema_output_format(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        output = {
            "modalities": ["text"],
            "format": {"type": "json_schema", "name": "res", "schema": schema, "strict": True},
        }
        _, body = self._model().to_request([], output)
        rf = body["response_format"]
        self.assertEqual(rf["type"], "json_schema")
        self.assertEqual(rf["json_schema"]["name"],   "res")
        self.assertEqual(rf["json_schema"]["schema"], schema)
        self.assertTrue(rf["json_schema"]["strict"])

    def test_text_output_no_response_format(self):
        _, body = self._model().to_request([], _text_output())
        self.assertNotIn("response_format", body)


class TestOpenAIFromResponse(unittest.TestCase):

    def _resp(self, content):
        return {"choices": [{"message": {"content": content}}]}

    def test_returns_string_for_text_output(self):
        result = OpenAIModel("gpt-4o").from_response(self._resp("Hello!"), _text_output())
        self.assertEqual(result, "Hello!")

    def test_returns_dict_for_json_output(self):
        result = OpenAIModel("gpt-4o").from_response(self._resp('{"key": "value"}'), _json_output())
        self.assertIsInstance(result, dict)
        self.assertEqual(result["key"], "value")

    def test_returns_dict_for_json_schema_output(self):
        output = {"modalities": ["text"], "format": {"type": "json_schema", "schema": {}}}
        result = OpenAIModel("gpt-4o").from_response(self._resp('{"answer": 42}'), output)
        self.assertEqual(result["answer"], 42)


# ---------------------------------------------------------------------------
# XAIModel.to_request / from_response
# ---------------------------------------------------------------------------

class TestXAIToRequest(unittest.TestCase):

    def test_path_is_v1_chat_completions(self):
        model = XAIModel("grok-3")
        path, _ = model.to_request([], _text_output())
        self.assertEqual(path, "/v1/chat/completions")

    def test_model_name_in_body(self):
        model = XAIModel("grok-3")
        _, body = model.to_request([], _text_output())
        self.assertEqual(body["model"], "grok-3")

    def test_from_response_returns_string(self):
        model = XAIModel("grok-3")
        resp  = {"choices": [{"message": {"content": "Grok answer"}}]}
        self.assertEqual(model.from_response(resp, _text_output()), "Grok answer")


# ---------------------------------------------------------------------------
# PerplexityModel.to_request / from_response
# ---------------------------------------------------------------------------

class TestPerplexityToRequest(unittest.TestCase):

    def test_path_is_chat_completions(self):
        model = PerplexityModel("sonar")
        path, _ = model.to_request([], _text_output())
        self.assertEqual(path, "/chat/completions")

    def test_model_name_in_body(self):
        model = PerplexityModel("sonar-pro")
        _, body = model.to_request([], _text_output())
        self.assertEqual(body["model"], "sonar-pro")

    def test_from_response_returns_string(self):
        model = PerplexityModel("sonar")
        resp  = {"choices": [{"message": {"content": "Sonar answer"}}]}
        self.assertEqual(model.from_response(resp, _text_output()), "Sonar answer")


# ---------------------------------------------------------------------------
# AnthropicModel.to_request / from_response
# ---------------------------------------------------------------------------

class TestAnthropicToRequest(unittest.TestCase):

    def _model(self, **kw):
        return AnthropicModel(
            "claude-sonnet-4-5",
            options={"max_tokens": 8192, **kw},
            api_key="test-anthropic-key",
        )

    def test_path(self):
        path, _ = self._model().to_request([], _text_output())
        self.assertEqual(path, "/v1/messages")

    def test_model_name_and_max_tokens(self):
        _, body = self._model().to_request([], _text_output())
        self.assertEqual(body["model"],      "claude-sonnet-4-5")
        self.assertEqual(body["max_tokens"], 8192)

    def test_system_extracted_to_top_level(self):
        msgs = [
            {"role": "system", "parts": [{"type": "text", "text": "Be concise."}]},
            {"role": "user",   "parts": [{"type": "text", "text": "hi"}]},
        ]
        _, body = self._model().to_request(msgs, _text_output())
        self.assertEqual(body["system"], "Be concise.")
        for m in body["messages"]:
            self.assertNotEqual(m["role"], "system")

    def test_multiple_system_messages_joined(self):
        msgs = [
            {"role": "system", "parts": [{"type": "text", "text": "Part A."}]},
            {"role": "system", "parts": [{"type": "text", "text": "Part B."}]},
            {"role": "user",   "parts": [{"type": "text", "text": "hi"}]},
        ]
        _, body = self._model().to_request(msgs, _text_output())
        self.assertIn("Part A.", body["system"])
        self.assertIn("Part B.", body["system"])

    def test_text_part_converted(self):
        msgs = [{"role": "user", "parts": [{"type": "text", "text": "hello"}]}]
        _, body = self._model().to_request(msgs, _text_output())
        block = body["messages"][0]["content"][0]
        self.assertEqual(block["type"], "text")
        self.assertEqual(block["text"], "hello")

    def test_image_url_part_converted(self):
        msgs = [{"role": "user", "parts": [
            {"type": "image", "source": {"kind": "url", "url": "https://x.com/img.png"}},
        ]}]
        _, body = self._model().to_request(msgs, _text_output())
        block = body["messages"][0]["content"][0]
        self.assertEqual(block["type"],           "image")
        self.assertEqual(block["source"]["type"], "url")
        self.assertEqual(block["source"]["url"],  "https://x.com/img.png")

    def test_image_base64_part_converted(self):
        msgs = [{"role": "user", "parts": [
            {"type": "image", "source": {"kind": "base64", "mime": "image/png", "data": "abc"}},
        ]}]
        _, body = self._model().to_request(msgs, _text_output())
        block = body["messages"][0]["content"][0]
        self.assertEqual(block["source"]["type"],       "base64")
        self.assertEqual(block["source"]["media_type"], "image/png")

    def test_video_part_skipped(self):
        msgs = [{"role": "user", "parts": [
            {"type": "video", "source": {"kind": "url", "url": "https://x.com/v.mp4"}},
        ]}]
        _, body = self._model().to_request(msgs, _text_output())
        self.assertEqual(body["messages"], [])

    def test_top_p_included_when_set(self):
        _, body = self._model(top_p=0.8).to_request([], _text_output())
        self.assertEqual(body["top_p"], 0.8)

    def test_top_k_included_when_set(self):
        _, body = self._model(top_k=40).to_request([], _text_output())
        self.assertEqual(body["top_k"], 40)

    def test_thinking_included_when_reasoning_set(self):
        _, body = self._model(reasoning="medium").to_request([], _text_output())
        self.assertEqual(body["thinking"]["type"],         "enabled")
        self.assertEqual(body["thinking"]["budget_tokens"], 10000)

    def test_temperature_forced_to_1_when_reasoning_set(self):
        # Anthropic requires temperature=1.0 when extended thinking is active.
        _, body = self._model(reasoning="high").to_request([], _text_output())
        self.assertEqual(body["temperature"], 1.0)

    def test_thinking_omitted_when_none(self):
        _, body = self._model().to_request([], _text_output())
        self.assertNotIn("thinking", body)


class TestAnthropicFromResponse(unittest.TestCase):

    def _resp(self, text):
        return {"content": [{"type": "text", "text": text}]}

    def _model(self):
        return AnthropicModel("claude-sonnet-4-5", api_key="test-anthropic-key")

    def test_returns_string_for_text_output(self):
        self.assertEqual(self._model().from_response(self._resp("world"), _text_output()), "world")

    def test_returns_dict_for_json_output(self):
        result = self._model().from_response(self._resp('{"k": 1}'), _json_output())
        self.assertEqual(result["k"], 1)

    def test_empty_content_returns_empty_string(self):
        self.assertEqual(self._model().from_response({"content": []}, _text_output()), "")

    def test_skips_non_text_blocks(self):
        resp = {
            "content": [
                {"type": "thinking", "thinking": "..."},
                {"type": "text",     "text": "answer"},
            ]
        }
        self.assertEqual(self._model().from_response(resp, _text_output()), "answer")


# ---------------------------------------------------------------------------
# GoogleAIModel.to_request / from_response
# ---------------------------------------------------------------------------

class TestGoogleToRequest(unittest.TestCase):

    def _model(self, **kw):
        return GoogleAIModel(
            "gemini-2.0-flash",
            options=kw if kw else None,
            api_key="test-google-key",
        )

    def test_path_contains_model_and_key(self):
        path, _ = self._model().to_request([], _text_output())
        self.assertIn("gemini-2.0-flash", path)
        self.assertIn("generateContent",  path)
        self.assertIn("key=test-google-key", path)

    def test_generation_config_temperature(self):
        _, body = self._model(temperature=0.7).to_request([], _text_output())
        self.assertEqual(body["generationConfig"]["temperature"], 0.7)

    def test_generation_config_max_output_tokens(self):
        _, body = self._model(max_tokens=2048).to_request([], _text_output())
        self.assertEqual(body["generationConfig"]["maxOutputTokens"], 2048)

    def test_top_p_and_top_k_included(self):
        _, body = self._model(top_p=0.9, top_k=32).to_request([], _text_output())
        self.assertEqual(body["generationConfig"]["topP"], 0.9)
        self.assertEqual(body["generationConfig"]["topK"], 32)

    def test_system_goes_into_system_instruction(self):
        msgs = [
            {"role": "system", "parts": [{"type": "text", "text": "Be precise."}]},
            {"role": "user",   "parts": [{"type": "text", "text": "hi"}]},
        ]
        _, body = self._model().to_request(msgs, _text_output())
        self.assertIn("system_instruction", body)
        self.assertEqual(body["system_instruction"]["parts"][0]["text"], "Be precise.")
        for item in body["contents"]:
            self.assertNotEqual(item["role"], "system")

    def test_assistant_role_mapped_to_model(self):
        msgs = [
            {"role": "user",      "parts": [{"type": "text", "text": "ping"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "pong"}]},
        ]
        _, body = self._model().to_request(msgs, _text_output())
        roles = [c["role"] for c in body["contents"]]
        self.assertNotIn("assistant", roles)
        self.assertIn("model", roles)

    def test_text_part_converted(self):
        msgs = [{"role": "user", "parts": [{"type": "text", "text": "hello"}]}]
        _, body = self._model().to_request(msgs, _text_output())
        part = body["contents"][0]["parts"][0]
        self.assertEqual(part["text"], "hello")

    def test_image_url_part_converted(self):
        msgs = [{"role": "user", "parts": [
            {"type": "image",
             "source": {"kind": "url", "url": "https://x.com/img.png", "mime": "image/png"}},
        ]}]
        _, body = self._model().to_request(msgs, _text_output())
        part = body["contents"][0]["parts"][0]
        self.assertIn("fileData", part)
        self.assertEqual(part["fileData"]["fileUri"], "https://x.com/img.png")

    def test_image_base64_part_converted(self):
        msgs = [{"role": "user", "parts": [
            {"type": "image", "source": {"kind": "base64", "mime": "image/jpeg", "data": "xyz"}},
        ]}]
        _, body = self._model().to_request(msgs, _text_output())
        part = body["contents"][0]["parts"][0]
        self.assertIn("inlineData", part)
        self.assertEqual(part["inlineData"]["mimeType"], "image/jpeg")

    def test_video_url_part_converted(self):
        msgs = [{"role": "user", "parts": [
            {"type": "video",
             "source": {"kind": "url", "url": "https://x.com/v.mp4", "mime": "video/mp4"}},
        ]}]
        _, body = self._model().to_request(msgs, _text_output())
        part = body["contents"][0]["parts"][0]
        self.assertIn("fileData", part)
        self.assertEqual(part["fileData"]["mimeType"], "video/mp4")

    def test_audio_base64_part_converted(self):
        msgs = [{"role": "user", "parts": [
            {"type": "audio", "source": {"kind": "base64", "mime": "audio/wav", "data": "abc"}},
        ]}]
        _, body = self._model().to_request(msgs, _text_output())
        part = body["contents"][0]["parts"][0]
        self.assertIn("inlineData", part)
        self.assertEqual(part["inlineData"]["mimeType"], "audio/wav")

    def test_thinking_config_included_when_reasoning_set(self):
        _, body = self._model(reasoning="medium").to_request([], _text_output())
        tc = body["generationConfig"]["thinkingConfig"]
        self.assertEqual(tc["thinkingBudget"], 8192)

    def test_json_output_sets_mime_type(self):
        _, body = self._model().to_request([], _json_output())
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")

    def test_json_schema_sets_mime_and_schema(self):
        schema = {"type": "object"}
        output = {
            "modalities": ["text"],
            "format": {"type": "json_schema", "schema": schema},
        }
        _, body = self._model().to_request([], output)
        gc = body["generationConfig"]
        self.assertEqual(gc["responseMimeType"], "application/json")
        self.assertEqual(gc["responseSchema"],   schema)


class TestGoogleFromResponse(unittest.TestCase):

    def _resp(self, text):
        return {"candidates": [{"content": {"parts": [{"text": text}]}}]}

    def _model(self):
        return GoogleAIModel("gemini-2.0-flash", api_key="test-google-key")

    def test_returns_string_for_text_output(self):
        self.assertEqual(self._model().from_response(self._resp("hi there"), _text_output()), "hi there")

    def test_returns_dict_for_json_output(self):
        result = self._model().from_response(self._resp('{"x": 7}'), _json_output())
        self.assertEqual(result["x"], 7)

    def test_empty_response_returns_empty_string(self):
        self.assertEqual(self._model().from_response({}, _text_output()), "")


if __name__ == "__main__":
    unittest.main()
