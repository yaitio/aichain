"""
tests.models.test_models_text_to_image
========================================

Tests for text-to-image models:
  - OpenAI   : dall-e-3, gpt-image-1
  - xAI      : grok-imagine-image
  - Google   : gemini-3.1-flash-image-preview  (responseModalities)
  - Qwen     : wanx2.1-t2i-turbo

Covers:
  1. Factory routing    — correct provider subclass
  2. to_request()       — correct endpoint path + body structure
  3. from_response()    — returns dict with url / base64 / mime_type / revised_prompt
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_TEST_KEYS = {
    "OPENAI_API_KEY":    "test-openai-key",
    "XAI_API_KEY":       "test-xai-key",
    "GOOGLE_AI_API_KEY": "test-google-key",
    "DASHSCOPE_API_KEY": "test-qwen-key",
}
for _k, _v in _TEST_KEYS.items():
    if not os.environ.get(_k):
        os.environ[_k] = _v

from models import Model, OpenAIModel, XAIModel, GoogleAIModel, QwenModel

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_PROMPT_MSG = [
    {"role": "user", "parts": [{"type": "text", "text": "A red apple on a table"}]},
]

_IMAGE_OUTPUT = {"modalities": ["image"], "format": {"type": "image"}}
_TEXT_OUTPUT  = {"modalities": ["text"],  "format": {"type": "text"}}

# Minimal 1×1 PNG base64 (used to construct synthetic responses)
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _openai_image_response(b64: str = _TINY_PNG_B64) -> dict:
    return {"data": [{"b64_json": b64, "revised_prompt": "A red apple"}]}


def _google_image_response(b64: str = _TINY_PNG_B64) -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data":     b64,
                            }
                        }
                    ]
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# 1. Factory routing
# ---------------------------------------------------------------------------

class TestImageModelFactory(unittest.TestCase):

    def test_dalle3_routes_to_openai(self):
        self.assertIsInstance(Model("dall-e-3"), OpenAIModel)

    def test_gpt_image_routes_to_openai(self):
        self.assertIsInstance(Model("gpt-image-1"), OpenAIModel)

    def test_grok_imagine_routes_to_xai(self):
        self.assertIsInstance(Model("grok-imagine-image-pro"), XAIModel)

    def test_gemini_image_routes_to_google(self):
        self.assertIsInstance(Model("gemini-3.1-flash-image-preview"), GoogleAIModel)

    def test_wanx_routes_to_qwen(self):
        self.assertIsInstance(Model("wanx2.1-t2i-turbo"), QwenModel)


# ---------------------------------------------------------------------------
# 2a. to_request() — OpenAI image models
# ---------------------------------------------------------------------------

class TestOpenAIImageToRequest(unittest.TestCase):

    def test_dalle3_path(self):
        model = OpenAIModel("dall-e-3")
        path, _ = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertEqual(path, "/v1/images/generations")

    def test_gpt_image_path(self):
        model = OpenAIModel("gpt-image-1")
        path, _ = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertEqual(path, "/v1/images/generations")

    def test_body_has_model_name(self):
        model = OpenAIModel("dall-e-3")
        _, body = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertEqual(body["model"], "dall-e-3")

    def test_body_has_prompt(self):
        model = OpenAIModel("dall-e-3")
        _, body = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertIn("prompt", body)
        self.assertIn("apple", body["prompt"].lower())

    def test_body_has_n_equals_1(self):
        model = OpenAIModel("dall-e-3")
        _, body = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertEqual(body["n"], 1)

    def test_dalle3_requests_b64_json(self):
        model = OpenAIModel("dall-e-3")
        _, body = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertEqual(body.get("response_format"), "b64_json")

    def test_gpt_image_omits_response_format(self):
        # gpt-image-* always returns b64_json natively and rejects the param.
        model = OpenAIModel("gpt-image-1")
        _, body = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertNotIn("response_format", body)

    def test_size_forwarded_when_set(self):
        model = OpenAIModel("dall-e-3")
        output = {"modalities": ["image"], "format": {"type": "image", "size": "1024x1024"}}
        _, body = model.to_request(_PROMPT_MSG, output)
        self.assertEqual(body["size"], "1024x1024")

    def test_size_absent_when_not_set(self):
        model = OpenAIModel("dall-e-3")
        _, body = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertNotIn("size", body)


# ---------------------------------------------------------------------------
# 2b. to_request() — xAI image models
# ---------------------------------------------------------------------------

class TestXAIImageToRequest(unittest.TestCase):

    def test_path(self):
        model = XAIModel("grok-imagine-image")
        path, _ = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertEqual(path, "/v1/images/generations")

    def test_body_has_model_name(self):
        model = XAIModel("grok-imagine-image")
        _, body = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertEqual(body["model"], "grok-imagine-image")

    def test_body_has_prompt(self):
        model = XAIModel("grok-imagine-image")
        _, body = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertIn("prompt", body)

    def test_body_requests_b64_json(self):
        model = XAIModel("grok-imagine-image")
        _, body = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertEqual(body.get("response_format"), "b64_json")

    def test_xai_image_does_not_include_size(self):
        # xAI grok-imagine models do not accept size; should be absent.
        model = XAIModel("grok-imagine-image")
        _, body = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertNotIn("size", body)


# ---------------------------------------------------------------------------
# 2c. to_request() — Google image models
# ---------------------------------------------------------------------------

class TestGoogleImageToRequest(unittest.TestCase):

    def test_path_contains_model_and_generate_content(self):
        model = GoogleAIModel("gemini-3.1-flash-image-preview")
        path, _ = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertIn("gemini-3.1-flash-image-preview", path)
        self.assertIn("generateContent", path)

    def test_body_has_response_modalities_image(self):
        model = GoogleAIModel("gemini-3.1-flash-image-preview")
        _, body = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        mods = body["generationConfig"].get("responseModalities", [])
        self.assertIn("IMAGE", mods)

    def test_text_plus_image_modality_includes_text(self):
        model = GoogleAIModel("gemini-3.1-flash-image-preview")
        output = {"modalities": ["text", "image"], "format": {"type": "image"}}
        _, body = model.to_request(_PROMPT_MSG, output)
        mods = body["generationConfig"].get("responseModalities", [])
        self.assertIn("IMAGE", mods)
        self.assertIn("TEXT", mods)


# ---------------------------------------------------------------------------
# 2d. to_request() — Qwen wanx models
# ---------------------------------------------------------------------------

class TestQwenImageToRequest(unittest.TestCase):

    def test_wanx_path(self):
        model = QwenModel("wanx2.1-t2i-turbo")
        path, _ = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertEqual(path, "/compatible-mode/v1/images/generations")

    def test_body_has_model_name(self):
        model = QwenModel("wanx2.1-t2i-turbo")
        _, body = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertEqual(body["model"], "wanx2.1-t2i-turbo")

    def test_body_has_prompt(self):
        model = QwenModel("wanx2.1-t2i-turbo")
        _, body = model.to_request(_PROMPT_MSG, _IMAGE_OUTPUT)
        self.assertIn("prompt", body)


# ---------------------------------------------------------------------------
# 3a. from_response() — OpenAI / xAI  (data[0].b64_json)
# ---------------------------------------------------------------------------

class TestOpenAIImageFromResponse(unittest.TestCase):

    def _model(self):
        return OpenAIModel("dall-e-3")

    def test_returns_dict(self):
        result = self._model().from_response(_openai_image_response(), _IMAGE_OUTPUT)
        self.assertIsInstance(result, dict)

    def test_dict_has_required_keys(self):
        result = self._model().from_response(_openai_image_response(), _IMAGE_OUTPUT)
        for key in ("url", "base64", "mime_type", "revised_prompt"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_base64_matches_response(self):
        result = self._model().from_response(_openai_image_response(_TINY_PNG_B64), _IMAGE_OUTPUT)
        self.assertEqual(result["base64"], _TINY_PNG_B64)

    def test_revised_prompt_extracted(self):
        result = self._model().from_response(_openai_image_response(), _IMAGE_OUTPUT)
        self.assertEqual(result["revised_prompt"], "A red apple")

    def test_mime_type_detected_as_png(self):
        result = self._model().from_response(_openai_image_response(_TINY_PNG_B64), _IMAGE_OUTPUT)
        self.assertEqual(result["mime_type"], "image/png")

    def test_xai_image_from_response(self):
        model = XAIModel("grok-imagine-image")
        result = model.from_response(_openai_image_response(), _IMAGE_OUTPUT)
        self.assertIsInstance(result, dict)
        self.assertIn("base64", result)


# ---------------------------------------------------------------------------
# 3b. from_response() — Google
# ---------------------------------------------------------------------------

class TestGoogleImageFromResponse(unittest.TestCase):

    def setUp(self):
        self.model = GoogleAIModel("gemini-3.1-flash-image-preview")
        self._output = {"modalities": ["image"], "format": {"type": "image"}}

    def test_returns_dict(self):
        result = self.model.from_response(_google_image_response(), self._output)
        self.assertIsInstance(result, dict)

    def test_dict_has_required_keys(self):
        result = self.model.from_response(_google_image_response(), self._output)
        for key in ("url", "base64", "mime_type", "revised_prompt"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_base64_extracted(self):
        result = self.model.from_response(_google_image_response(_TINY_PNG_B64), self._output)
        self.assertEqual(result["base64"], _TINY_PNG_B64)

    def test_mime_type_from_response(self):
        result = self.model.from_response(_google_image_response(), self._output)
        self.assertEqual(result["mime_type"], "image/png")

    def test_url_is_none_for_google(self):
        result = self.model.from_response(_google_image_response(), self._output)
        self.assertIsNone(result["url"])

    def test_empty_candidates_returns_none_base64(self):
        result = self.model.from_response({"candidates": []}, self._output)
        self.assertIsNone(result["base64"])


# ---------------------------------------------------------------------------
# 3c. from_response() — Qwen wanx
# ---------------------------------------------------------------------------

class TestQwenImageFromResponse(unittest.TestCase):

    def test_returns_dict_with_required_keys(self):
        model = QwenModel("wanx2.1-t2i-turbo")
        resp  = _openai_image_response(_TINY_PNG_B64)
        result = model.from_response(resp, _IMAGE_OUTPUT)
        self.assertIsInstance(result, dict)
        for key in ("url", "base64", "mime_type", "revised_prompt"):
            self.assertIn(key, result)


# ---------------------------------------------------------------------------
# Live integration tests
# ---------------------------------------------------------------------------

_OPENAI_REAL = os.getenv("OPENAI_API_KEY", "").startswith("sk-")
_XAI_REAL    = os.getenv("XAI_API_KEY",    "").startswith("xai-")
_GOOGLE_REAL = os.getenv("GOOGLE_AI_API_KEY", "").startswith("AIza")
_QWEN_REAL   = os.getenv("DASHSCOPE_API_KEY", "").startswith("sk-")


@unittest.skipUnless(_OPENAI_REAL, "Set a real OPENAI_API_KEY to run live image tests")
class TestOpenAIImageLive(unittest.TestCase):
    def test_gpt_image_returns_base64(self):
        from skills import Skill
        skill = Skill(
            model         = Model("gpt-image-1"),
            input         = {"messages": [{"role": "user", "parts": [{"type": "text", "text": "A tiny red dot"}]}]},
            output_format = "image",
        )
        result = skill.run()
        self.assertIsInstance(result, dict)
        self.assertIn("base64", result)
        self.assertIsNotNone(result["base64"])


@unittest.skipUnless(_GOOGLE_REAL, "Set a real GOOGLE_AI_API_KEY to run live image tests")
class TestGoogleImageLive(unittest.TestCase):
    def test_gemini_image_returns_base64(self):
        from skills import Skill
        skill = Skill(
            model      = Model("gemini-3.1-flash-image-preview"),
            input      = {"messages": [{"role": "user", "parts": [{"type": "text", "text": "A tiny red dot"}]}]},
            output     = {"modalities": ["image"], "format": {"type": "image"}},
        )
        result = skill.run()
        self.assertIsInstance(result, dict)
        self.assertIn("base64", result)


if __name__ == "__main__":
    unittest.main()
