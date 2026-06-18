"""
Tests for image-to-image (image editing) across all four image providers.

Detection rule: an input image part + an image-output model/modality ⇒ edit.
- OpenAI : multipart POST /v1/images/edits   (image[] file fields)
- xAI    : JSON     POST /v1/images/edits     (image:{url,type})
- Qwen   : sync     POST multimodal-generation (content[].image data-URI)
- Google : same     :generateContent          (inlineData + IMAGE modality)

Transport is mocked — no network is touched.
"""

import base64
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from clients._families._openai_compat import (
    _messages_have_image,
    _is_openai_editable_image_model,
    _build_image_edits_request,
    _build_xai_image_edit_request,
    _parse_image_generations_response,
)
from clients._families.openai import OpenAIClient
from clients._families.qwen import (
    QwenClient,
    _MULTIMODAL_GEN_PATH,
    _build_qwen_image_edit_request,
    _is_qwen_image_edit,
)
from models._data import PROVIDERS as _PROVIDERS
from models import Model, registry
from skills import Skill
from skills._adapters import normalize_input

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
_B64 = base64.b64encode(_PNG).decode("ascii")
_OUT = {"modalities": ["image"], "format": {"type": "image", "size": "1024x1024"}}


def _img(kind="base64", **kw):
    return {"type": "image", "source": {"kind": kind, **kw}}


def _edit_msgs(src=None):
    return [{"role": "user", "parts": [
        src or _img(data=_B64, mime="image/png"),
        {"type": "text", "text": "Put it on a marble counter, soft light"},
    ]}]


def _wrap(name):
    return SimpleNamespace(name=name)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestDetection(unittest.TestCase):

    def test_has_image_true_with_image_part(self):
        self.assertTrue(_messages_have_image(_edit_msgs()))

    def test_has_image_false_text_only(self):
        self.assertFalse(_messages_have_image(
            [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]))

    def test_openai_editable_models(self):
        self.assertTrue(_is_openai_editable_image_model("gpt-image-1.5"))
        self.assertTrue(_is_openai_editable_image_model("chatgpt-image-latest"))
        self.assertTrue(_is_openai_editable_image_model("dall-e-2"))
        self.assertFalse(_is_openai_editable_image_model("dall-e-3"))
        self.assertFalse(_is_openai_editable_image_model("gpt-4o"))

    def test_qwen_edit_gate(self):
        self.assertTrue(_is_qwen_image_edit("qwen-image-edit"))
        self.assertTrue(_is_qwen_image_edit("qwen-image-edit-plus"))
        self.assertFalse(_is_qwen_image_edit("wan2.2-t2i-flash"))


# ---------------------------------------------------------------------------
# OpenAI — multipart edits, and the generate regression
# ---------------------------------------------------------------------------

class TestOpenAIEdit(unittest.TestCase):

    def test_edit_routes_to_multipart(self):
        m = Model("gpt-image-1.5", api_key="k")
        path, body = m.to_request(_edit_msgs(), _OUT)
        self.assertEqual(path, "/v1/images/edits")
        self.assertTrue(body["_multipart"])
        names = [f[0] for f in body["fields"]]
        self.assertIn("model", names)
        self.assertIn("prompt", names)
        self.assertIn("image[]", names)        # gpt-image uses the array key
        self.assertNotIn("response_format", names)   # gpt-image returns b64 natively
        # the image[] field is a file tuple (filename, bytes, mime)
        file_field = next(f for f in body["fields"] if f[0] == "image[]")
        self.assertEqual(file_field[1][1], _PNG)
        self.assertEqual(file_field[1][2], "image/png")

    def test_no_image_falls_back_to_generation(self):
        m = Model("gpt-image-1.5", api_key="k")
        path, body = m.to_request(
            [{"role": "user", "parts": [{"type": "text", "text": "a fox"}]}], _OUT)
        self.assertEqual(path, "/v1/images/generations")
        self.assertNotIn("_multipart", body)

    def test_url_source_rejected(self):
        with self.assertRaises(ValueError):
            _build_image_edits_request(
                _wrap("gpt-image-1.5"), _edit_msgs(_img("url", url="https://x/y.png")), _OUT)

    def test_dalle2_uses_single_image_field(self):
        _, body = _build_image_edits_request(_wrap("dall-e-2"), _edit_msgs(), _OUT)
        self.assertIn("image", [f[0] for f in body["fields"]])


class TestOpenAISendSeam(unittest.TestCase):

    def _client(self):
        return OpenAIClient("k", data=_PROVIDERS["openai"])

    def test_multipart_body_goes_to_post_form_without_content_type(self):
        c = self._client()
        c._post_form = MagicMock(return_value=b"{}")
        c._post = MagicMock(return_value=b"{}")
        headers = {"Authorization": "Bearer k", "Content-Type": "application/json"}
        c.send("/v1/images/edits", {"_multipart": True, "fields": [("model", "x")]}, headers)
        c._post_form.assert_called_once()
        sent_headers = c._post_form.call_args[0][2]
        self.assertNotIn("Content-Type", sent_headers)
        self.assertIn("Authorization", sent_headers)
        c._post.assert_not_called()

    def test_plain_body_goes_to_post(self):
        c = self._client()
        c._post = MagicMock(return_value=b"{}")
        c.send("/v1/chat/completions", {"model": "gpt-4o"}, {"Content-Type": "application/json"})
        c._post.assert_called_once()


# ---------------------------------------------------------------------------
# xAI — JSON edits (confirmed by live probe)
# ---------------------------------------------------------------------------

class TestXaiEdit(unittest.TestCase):

    def test_edit_json_shape(self):
        m = Model("grok-imagine-image", api_key="k")
        path, body = m.to_request(_edit_msgs(), _OUT)
        self.assertEqual(path, "/v1/images/edits")
        self.assertEqual(body["model"], "grok-imagine-image")
        self.assertEqual(body["image"]["type"], "image_url")
        self.assertTrue(body["image"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(body["response_format"], "b64_json")

    def test_url_source_passes_through(self):
        _, body = _build_xai_image_edit_request(
            _wrap("grok-imagine-image"), _edit_msgs(_img("url", url="https://x/y.png")), _OUT)
        self.assertEqual(body["image"]["url"], "https://x/y.png")

    def test_multi_image_uses_images_array(self):
        # Two reference images → xAI `images` array (single stays `image`); ≤3 cap.
        msgs = [{"role": "user", "parts": [
            _img(data=_B64, mime="image/png"),
            _img(data=_B64, mime="image/png"),
            {"type": "text", "text": "combine the two references"}]}]
        _, body = _build_xai_image_edit_request(_wrap("grok-imagine-image"), msgs, _OUT)
        self.assertNotIn("image", body)
        self.assertEqual(len(body["images"]), 2)
        self.assertEqual(body["images"][0]["type"], "image_url")

    def test_no_image_falls_back_to_generation(self):
        m = Model("grok-imagine-image", api_key="k")
        path, _ = m.to_request(
            [{"role": "user", "parts": [{"type": "text", "text": "a fox"}]}], _OUT)
        self.assertEqual(path, "/v1/images/generations")


# ---------------------------------------------------------------------------
# Qwen — synchronous multimodal-generation edit
# ---------------------------------------------------------------------------

class TestQwenEdit(unittest.TestCase):

    def test_build_shape(self):
        path, body = _build_qwen_image_edit_request(_wrap("qwen-image-edit"), _edit_msgs(), _OUT)
        self.assertEqual(path, _MULTIMODAL_GEN_PATH)
        content = body["input"]["messages"][0]["content"]
        self.assertEqual(content[0]["image"], f"data:image/png;base64,{_B64}")
        self.assertEqual(content[1]["text"], "Put it on a marble counter, soft light")
        self.assertEqual(body["parameters"]["size"], "1024*1024")   # x → *

    def test_routes_through_build_request(self):
        c = QwenClient("k", data=_PROVIDERS["qwen"])
        path, _ = c.build_request(_edit_msgs(), _OUT, {"name": "qwen-image-edit"})
        self.assertEqual(path, _MULTIMODAL_GEN_PATH)

    def test_provider_resolves_to_qwen(self):
        self.assertEqual(Model("qwen-image-edit", api_key="k")._provider, "qwen")

    def test_sync_send_downloads_and_reshapes(self):
        c = QwenClient("k", data=_PROVIDERS["qwen"])
        c._post = MagicMock(return_value=json.dumps({"output": {"choices": [
            {"message": {"content": [{"image": "https://img/out.png"}]}}]}}).encode())
        c._download = MagicMock(return_value={"data": _PNG, "media_type": "image/png"})
        raw = c.send(_MULTIMODAL_GEN_PATH, {"model": "qwen-image-edit"}, {})
        data = json.loads(raw)
        self.assertEqual(data["data"][0]["b64_json"], _B64)
        c._download.assert_called_once_with("https://img/out.png")

    def test_end_to_end_through_skill(self):
        m = Model("qwen-image-edit", api_key="k")
        m.client._post = MagicMock(return_value=json.dumps({"output": {"choices": [
            {"message": {"content": [{"image": "https://img/out.png"}]}}]}}).encode())
        m.client._download = MagicMock(return_value={"data": _PNG, "media_type": "image/png"})
        img = Skill(model=m, input={"messages": _edit_msgs()}, output=_OUT).run()
        self.assertEqual(img["base64"], _B64)
        self.assertEqual(img["mime_type"], "image/png")


# ---------------------------------------------------------------------------
# Google — same generateContent endpoint, input image + IMAGE modality
# ---------------------------------------------------------------------------

class TestGoogleEdit(unittest.TestCase):

    def test_input_image_and_image_modality(self):
        m = Model("gemini-3.1-flash-image", api_key="k")
        path, body = m.to_request(_edit_msgs(), _OUT)
        self.assertTrue(path.endswith(":generateContent"))
        kinds = [list(p.keys())[0] for p in body["contents"][0]["parts"]]
        self.assertIn("inlineData", kinds)
        self.assertEqual(body["generationConfig"]["responseModalities"], ["IMAGE"])


# ---------------------------------------------------------------------------
# Parser reuse — mime_type honoured when the provider supplies it
# ---------------------------------------------------------------------------

class TestParserReuse(unittest.TestCase):

    def test_provider_mime_type_used(self):
        out = _parse_image_generations_response(
            {"data": [{"b64_json": _B64, "mime_type": "image/jpeg"}]})
        self.assertEqual(out["mime_type"], "image/jpeg")
        self.assertEqual(out["base64"], _B64)

    def test_mime_detected_when_absent(self):
        out = _parse_image_generations_response({"data": [{"b64_json": _B64}]})
        self.assertEqual(out["mime_type"], "image/png")   # from PNG magic bytes


# ---------------------------------------------------------------------------
# file → base64 input helper
# ---------------------------------------------------------------------------

class TestFileSourceHelper(unittest.TestCase):

    def test_file_path_loaded_to_base64_with_mime(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_PNG)
            path = f.name
        try:
            norm = normalize_input({"messages": [{"role": "user", "parts": [
                _img("file", path=path)]}]})
            src = norm["messages"][0]["parts"][0]["source"]
            self.assertEqual(src["data"], _B64)
            self.assertEqual(src["mime"], "image/png")
        finally:
            os.unlink(path)

    def test_existing_data_untouched(self):
        norm = normalize_input({"messages": [{"role": "user", "parts": [
            _img("file", data="ALREADY", mime="image/webp")]}]})
        src = norm["messages"][0]["parts"][0]["source"]
        self.assertEqual(src["data"], "ALREADY")
        self.assertEqual(src["mime"], "image/webp")


# ---------------------------------------------------------------------------
# Registry discovery
# ---------------------------------------------------------------------------

class TestRegistry(unittest.TestCase):

    def test_models_listed(self):
        ms = registry.models(task="image-to-image")
        for name in ("gpt-image-1.5", "gemini-3.1-flash-image",
                     "grok-imagine-image", "qwen-image-edit"):
            self.assertIn(name, ms)

    def test_providers_listed(self):
        self.assertEqual(set(registry.providers(task="image-to-image")),
                         {"openai", "google", "xai", "qwen", "recraft", "bfl"})

    def test_is_supported(self):
        self.assertTrue(registry.is_supported("qwen-image-edit", "image-to-image"))


# ---------------------------------------------------------------------------
# Empty image response → clear error (no silent base64=None)
# ---------------------------------------------------------------------------

class TestEmptyImageResponse(unittest.TestCase):

    _OUT_IMG = {"modalities": ["image"], "format": {"type": "image"}}

    def test_google_no_candidates_raises(self):
        m = Model("gemini-3.1-flash-image", api_key="k")
        with self.assertRaises(ValueError):
            m.from_response({"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}, self._OUT_IMG)

    def test_google_text_instead_of_image_raises(self):
        m = Model("gemini-3.1-flash-image", api_key="k")
        resp = {"candidates": [{"content": {"parts": [{"text": "sorry"}]}, "finishReason": "STOP"}]}
        with self.assertRaises(ValueError):
            m.from_response(resp, self._OUT_IMG)


if __name__ == "__main__":
    unittest.main()
