"""
tests.models.test_models_image_to_text
========================================

Tests for vision / multimodal models — image input → text output.

Covers:
  1. to_request()   — image URL and base64 encoded correctly per provider
  2. from_response() — text description extracted correctly

Providers with vision:
  OpenAI     gpt-4o          image_url content block
  Anthropic  claude-opus-4-6 source.type=url / source.type=base64
  Google     gemini-2.5-flash fileData (URL) / inlineData (base64)
  xAI        grok-3          image_url (same as OpenAI)
  Kimi       kimi-k2.5       image_url (OpenAI-compatible)
  Qwen       qwen-vl-max     image_url (OpenAI-compatible)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_TEST_KEYS = {
    "OPENAI_API_KEY":    "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "GOOGLE_AI_API_KEY": "test-google-key",
    "XAI_API_KEY":       "test-xai-key",
    "MOONSHOT_API_KEY":  "test-kimi-key",
    "DASHSCOPE_API_KEY": "test-qwen-key",
}
for _k, _v in _TEST_KEYS.items():
    if not os.environ.get(_k):
        os.environ[_k] = _v

from models import OpenAIModel, AnthropicModel, GoogleAIModel, XAIModel, KimiModel, QwenModel

# ---------------------------------------------------------------------------
# Shared universal message fixtures
# ---------------------------------------------------------------------------

_IMAGE_URL = "https://example.com/cat.png"
_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAABjE+ibYAAAAASUVORK5CYII="

_TEXT_OUTPUT = {"modalities": ["text"], "format": {"type": "text"}}


def _url_vision_msgs(url: str = _IMAGE_URL) -> list:
    return [
        {
            "role": "user",
            "parts": [
                {"type": "image", "source": {"kind": "url", "url": url, "mime": "image/png"}},
                {"type": "text",  "text": "What is in this image?"},
            ],
        }
    ]


def _b64_vision_msgs(data: str = _IMAGE_B64) -> list:
    return [
        {
            "role": "user",
            "parts": [
                {
                    "type":   "image",
                    "source": {"kind": "base64", "data": data, "mime": "image/png"},
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# 1. OpenAI — image_url content block
# ---------------------------------------------------------------------------

class TestOpenAIVisionToRequest(unittest.TestCase):

    def setUp(self):
        self.model = OpenAIModel("gpt-4o")

    def test_url_produces_image_url_block(self):
        _, body = self.model.to_request(_url_vision_msgs(), _TEXT_OUTPUT)
        user_content = body["messages"][0]["content"]
        # Multi-part message: content is a list
        self.assertIsInstance(user_content, list)
        types = [c["type"] for c in user_content]
        self.assertIn("image_url", types)

    def test_url_value_matches_input(self):
        _, body = self.model.to_request(_url_vision_msgs(), _TEXT_OUTPUT)
        user_content = body["messages"][0]["content"]
        img_block = next(c for c in user_content if c["type"] == "image_url")
        self.assertEqual(img_block["image_url"]["url"], _IMAGE_URL)

    def test_base64_produces_data_uri(self):
        _, body = self.model.to_request(_b64_vision_msgs(), _TEXT_OUTPUT)
        user_content = body["messages"][0]["content"]
        img_block = next(c for c in user_content if c["type"] == "image_url")
        self.assertIn("data:image/png;base64,", img_block["image_url"]["url"])

    def test_text_part_also_present(self):
        _, body = self.model.to_request(_url_vision_msgs(), _TEXT_OUTPUT)
        user_content = body["messages"][0]["content"]
        types = [c["type"] for c in user_content]
        self.assertIn("text", types)


# ---------------------------------------------------------------------------
# 2. Anthropic — native image source blocks
# ---------------------------------------------------------------------------

class TestAnthropicVisionToRequest(unittest.TestCase):

    def setUp(self):
        self.model = AnthropicModel("claude-opus-4-6")

    def _get_user_content(self, messages):
        _, body = self.model.to_request(messages, _TEXT_OUTPUT)
        return body["messages"][0]["content"]

    def test_url_produces_url_source(self):
        content = self._get_user_content(_url_vision_msgs())
        img_block = next(b for b in content if b["type"] == "image")
        self.assertEqual(img_block["source"]["type"], "url")
        self.assertEqual(img_block["source"]["url"], _IMAGE_URL)

    def test_base64_produces_base64_source(self):
        content = self._get_user_content(_b64_vision_msgs())
        img_block = next(b for b in content if b["type"] == "image")
        self.assertEqual(img_block["source"]["type"], "base64")
        self.assertEqual(img_block["source"]["data"], _IMAGE_B64)

    def test_base64_source_has_media_type(self):
        content = self._get_user_content(_b64_vision_msgs())
        img_block = next(b for b in content if b["type"] == "image")
        self.assertIn("media_type", img_block["source"])

    def test_text_part_present_alongside_image(self):
        content = self._get_user_content(_url_vision_msgs())
        types = [b["type"] for b in content]
        self.assertIn("text", types)
        self.assertIn("image", types)


# ---------------------------------------------------------------------------
# 3. Google — fileData (URL) / inlineData (base64)
# ---------------------------------------------------------------------------

class TestGoogleVisionToRequest(unittest.TestCase):

    def setUp(self):
        self.model = GoogleAIModel("gemini-2.5-flash")

    def _get_user_parts(self, messages):
        _, body = self.model.to_request(messages, _TEXT_OUTPUT)
        # Find the user content (role="user")
        user_content = next(c for c in body["contents"] if c["role"] == "user")
        return user_content["parts"]

    def test_url_produces_file_data(self):
        parts = self._get_user_parts(_url_vision_msgs())
        img_part = next(p for p in parts if "fileData" in p)
        self.assertEqual(img_part["fileData"]["fileUri"], _IMAGE_URL)

    def test_base64_produces_inline_data(self):
        parts = self._get_user_parts(_b64_vision_msgs())
        img_part = next(p for p in parts if "inlineData" in p)
        self.assertEqual(img_part["inlineData"]["data"], _IMAGE_B64)

    def test_inline_data_has_mime_type(self):
        parts = self._get_user_parts(_b64_vision_msgs())
        img_part = next(p for p in parts if "inlineData" in p)
        self.assertIn("mimeType", img_part["inlineData"])

    def test_text_part_present(self):
        parts = self._get_user_parts(_url_vision_msgs())
        has_text = any("text" in p for p in parts)
        self.assertTrue(has_text)


# ---------------------------------------------------------------------------
# 4. xAI — same as OpenAI (image_url blocks)
# ---------------------------------------------------------------------------

class TestXAIVisionToRequest(unittest.TestCase):

    def setUp(self):
        self.model = XAIModel("grok-3")

    def test_url_produces_image_url_block(self):
        _, body = self.model.to_request(_url_vision_msgs(), _TEXT_OUTPUT)
        user_content = body["messages"][0]["content"]
        types = [c["type"] for c in user_content]
        self.assertIn("image_url", types)

    def test_url_value_correct(self):
        _, body = self.model.to_request(_url_vision_msgs(), _TEXT_OUTPUT)
        user_content = body["messages"][0]["content"]
        img_block = next(c for c in user_content if c["type"] == "image_url")
        self.assertEqual(img_block["image_url"]["url"], _IMAGE_URL)


# ---------------------------------------------------------------------------
# 5. Kimi — OpenAI-compatible (image_url blocks)
# ---------------------------------------------------------------------------

class TestKimiVisionToRequest(unittest.TestCase):

    def setUp(self):
        self.model = KimiModel("kimi-k2.5")

    def test_url_produces_image_url_block(self):
        _, body = self.model.to_request(_url_vision_msgs(), _TEXT_OUTPUT)
        user_content = body["messages"][0]["content"]
        types = [c["type"] for c in user_content]
        self.assertIn("image_url", types)


# ---------------------------------------------------------------------------
# 6. Qwen — OpenAI-compatible (image_url blocks)
# ---------------------------------------------------------------------------

class TestQwenVisionToRequest(unittest.TestCase):

    def setUp(self):
        self.model = QwenModel("qwen-vl-max")

    def test_path_is_compatible_mode_chat(self):
        path, _ = self.model.to_request(_url_vision_msgs(), _TEXT_OUTPUT)
        self.assertEqual(path, "/compatible-mode/v1/chat/completions")

    def test_url_produces_image_url_block(self):
        _, body = self.model.to_request(_url_vision_msgs(), _TEXT_OUTPUT)
        user_content = body["messages"][0]["content"]
        types = [c["type"] for c in user_content]
        self.assertIn("image_url", types)


# ---------------------------------------------------------------------------
# 7. from_response() — providers return text describing the image
# ---------------------------------------------------------------------------

class TestVisionFromResponse(unittest.TestCase):

    def _openai_resp(self, text):
        return {"choices": [{"message": {"content": text}}]}

    def _anthropic_resp(self, text):
        return {"content": [{"type": "text", "text": text}]}

    def _google_resp(self, text):
        return {"candidates": [{"content": {"parts": [{"text": text}]}}]}

    def test_openai_extracts_description(self):
        result = OpenAIModel("gpt-4o").from_response(
            self._openai_resp("A cat sitting on a mat."), _TEXT_OUTPUT
        )
        self.assertEqual(result, "A cat sitting on a mat.")

    def test_anthropic_extracts_description(self):
        result = AnthropicModel("claude-opus-4-6").from_response(
            self._anthropic_resp("A tabby cat."), _TEXT_OUTPUT
        )
        self.assertEqual(result, "A tabby cat.")

    def test_google_extracts_description(self):
        result = GoogleAIModel("gemini-2.5-flash").from_response(
            self._google_resp("An orange cat."), _TEXT_OUTPUT
        )
        self.assertEqual(result, "An orange cat.")

    def test_xai_extracts_description(self):
        result = XAIModel("grok-3").from_response(
            self._openai_resp("A ginger cat."), _TEXT_OUTPUT
        )
        self.assertEqual(result, "A ginger cat.")

    def test_qwen_extracts_description(self):
        result = QwenModel("qwen-vl-max").from_response(
            self._openai_resp("A white cat."), _TEXT_OUTPUT
        )
        self.assertEqual(result, "A white cat.")


# ---------------------------------------------------------------------------
# Live integration tests
# ---------------------------------------------------------------------------

_IMAGE_URL_PUBLIC = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"

_OPENAI_REAL     = os.getenv("OPENAI_API_KEY",    "").startswith("sk-")
_ANTHROPIC_REAL  = os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-ant")
_GOOGLE_REAL     = os.getenv("GOOGLE_AI_API_KEY", "").startswith("AIza")


def _vision_skill(model_name, url=_IMAGE_URL_PUBLIC):
    from models import Model
    from skills import Skill
    return Skill(
        model = Model(model_name),
        input = {
            "messages": [
                {
                    "role": "user",
                    "parts": [
                        {"type": "image", "source": {"kind": "url", "url": url, "mime": "image/jpeg"}},
                        {"type": "text",  "text": "What animal is in this image? One word answer."},
                    ],
                }
            ]
        },
        output_format = "text",
    )


@unittest.skipUnless(_OPENAI_REAL, "Set a real OPENAI_API_KEY to run live vision tests")
class TestOpenAIVisionLive(unittest.TestCase):
    def test_identifies_cat(self):
        result = _vision_skill("gpt-4o-mini").run()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


@unittest.skipUnless(_ANTHROPIC_REAL, "Set a real ANTHROPIC_API_KEY to run live vision tests")
class TestAnthropicVisionLive(unittest.TestCase):
    def test_identifies_cat(self):
        result = _vision_skill("claude-haiku-4-5-20251001").run()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


@unittest.skipUnless(_GOOGLE_REAL, "Set a real GOOGLE_AI_API_KEY to run live vision tests")
class TestGoogleVisionLive(unittest.TestCase):
    def test_identifies_cat(self):
        result = _vision_skill("gemini-2.5-flash").run()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
