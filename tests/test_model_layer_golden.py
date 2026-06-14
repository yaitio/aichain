"""
tests.test_model_layer_golden
=============================

Characterisation ("golden master") tests for the model layer, captured on the
clean 1.2.1 baseline *before* the two-tier refactor (data + clients/_families).

They pin the exact wire behaviour of every provider through the Model() facade
— so they keep guarding behaviour after the per-provider classes are dissolved
into data + family clients. Any drift here is a regression, not an improvement.

Covers BOTH text and every non-text modality (image generation, vision, video,
image response), plus capability routing — the refactor must not break any of
them.

Pure: no network, no real keys.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_AI_API_KEY",
           "XAI_API_KEY", "PERPLEXITY_API_KEY", "DEEPSEEK_API_KEY",
           "MOONSHOT_API_KEY", "DASHSCOPE_API_KEY"):
    os.environ.setdefault(_k, "test-key")

from yait_aichain.models import Model

_MSG_USER = [{"role": "user", "parts": [{"type": "text", "text": "Hello"}]}]
_MSG_SYS = [
    {"role": "system", "parts": [{"type": "text", "text": "Be helpful."}]},
    {"role": "user",   "parts": [{"type": "text", "text": "Hello"}]},
]
_OUT_TEXT = {"modalities": ["text"], "format": {"type": "text"}}
_OUT_JSON = {"modalities": ["text"], "format": {"type": "json"}}

# Golden snapshot captured from the 1.2.1 implementation (2026-06-14).
_GOLDEN = {
    "openai": {"name": "gpt-4o", "path": "/v1/chat/completions",
        "scalars": {"model": "gpt-4o", "max_completion_tokens": 16384,
                    "temperature": 1.0, "top_p": 1.0},
        "system_in_messages": True, "json_adds": "response_format"},
    "anthropic": {"name": "claude-sonnet-4-6", "path": "/v1/messages",
        "scalars": {"model": "claude-sonnet-4-6", "max_tokens": 8192,
                    "temperature": 1.0, "system": "Be helpful."},
        "system_in_messages": False, "json_adds": None},
    "google": {"name": "gemini-2.5-flash",
        "path": "/models/gemini-2.5-flash:generateContent",
        "scalars": None, "system_in_messages": False, "json_adds": None},
    "xai": {"name": "grok-3", "path": "/v1/chat/completions",
        "scalars": {"model": "grok-3", "max_completion_tokens": 16384,
                    "temperature": 1.0, "top_p": 1.0},
        "system_in_messages": True, "json_adds": "response_format"},
    "perplexity": {"name": "sonar-pro", "path": "/chat/completions",
        "scalars": {"model": "sonar-pro", "max_completion_tokens": 8192,
                    "temperature": 0.2, "top_p": 0.9},
        "system_in_messages": True, "json_adds": "response_format"},
    "kimi": {"name": "kimi-k2-0905-preview", "path": "/v1/chat/completions",
        "scalars": {"model": "kimi-k2-0905-preview", "max_tokens": 32768,
                    "temperature": 1.0, "top_p": 0.95},
        "system_in_messages": True, "json_adds": "response_format"},
    "deepseek": {"name": "deepseek-chat", "path": "/v1/chat/completions",
        "scalars": {"model": "deepseek-chat", "max_tokens": 4096,
                    "temperature": 0.0, "top_p": 1.0},
        "system_in_messages": True, "json_adds": "response_format"},
    "qwen": {"name": "qwen-max", "path": "/compatible-mode/v1/chat/completions",
        "scalars": {"model": "qwen-max", "max_completion_tokens": 2048,
                    "temperature": 0.7, "top_p": 0.8},
        "system_in_messages": True, "json_adds": "response_format"},
}


class TestModelLayerGolden(unittest.TestCase):

    def _model(self, spec):
        return Model(spec["name"], api_key="k")

    def test_paths(self):
        for prov, spec in _GOLDEN.items():
            path, _ = self._model(spec).to_request(_MSG_USER, _OUT_TEXT)
            self.assertEqual(path, spec["path"], f"{prov} path drifted")

    def test_scalar_params(self):
        for prov, spec in _GOLDEN.items():
            if spec["scalars"] is None:
                continue
            _, body = self._model(spec).to_request(_MSG_SYS, _OUT_TEXT)
            scalars = {k: v for k, v in body.items()
                       if k not in ("messages", "contents")}
            self.assertEqual(scalars, spec["scalars"],
                             f"{prov} scalar params/defaults drifted")

    def test_system_handling(self):
        for prov, spec in _GOLDEN.items():
            _, body = self._model(spec).to_request(_MSG_SYS, _OUT_TEXT)
            in_msgs = any(m.get("role") == "system"
                          for m in body.get("messages", []))
            self.assertEqual(in_msgs, spec["system_in_messages"],
                             f"{prov} system handling drifted")

    def test_json_mode(self):
        for prov, spec in _GOLDEN.items():
            m = self._model(spec)
            _, bt = m.to_request(_MSG_USER, _OUT_TEXT)
            _, bj = m.to_request(_MSG_USER, _OUT_JSON)
            if spec["json_adds"]:
                self.assertIn(spec["json_adds"], bj, f"{prov} json lost")
                self.assertNotIn(spec["json_adds"], bt)

    def test_google_generation_config(self):
        _, body = Model("gemini-2.5-flash", api_key="k").to_request(_MSG_SYS, _OUT_TEXT)
        gc = body["generationConfig"]
        self.assertEqual(gc.get("temperature"), 1.0)
        self.assertEqual(gc.get("maxOutputTokens"), 8192)
        self.assertIn("system_instruction", body)


class TestModalityGolden(unittest.TestCase):
    """Pin non-text modalities — the refactor must not break them."""

    _IMG  = {"modalities": ["image"], "format": {"type": "image"}}
    _TEXT = {"modalities": ["text"],  "format": {"type": "text"}}

    def test_image_generation_routing(self):
        prompt = [{"role": "user", "parts": [{"type": "text", "text": "a cat"}]}]
        expected = {
            "gpt-image-1":                    "/v1/images/generations",
            "grok-imagine-image-pro":         "/v1/images/generations",
            "wanx2.1-t2i-turbo":              "/compatible-mode/v1/images/generations",
            "gemini-3.1-flash-image-preview": "/models/gemini-3.1-flash-image-preview:generateContent",
        }
        for name, path in expected.items():
            p, body = Model(name, api_key="k").to_request(prompt, self._IMG)
            self.assertEqual(p, path, f"{name} image route drifted")
            if "images/generations" in p:
                self.assertIn("prompt", body)

    def test_vision_input_conversion(self):
        msg = [{"role": "user", "parts": [
            {"type": "text", "text": "describe"},
            {"type": "image",
             "source": {"kind": "base64", "data": "AAAA", "mime": "image/png"}}]}]
        _, b = Model("gpt-4o", api_key="k").to_request(msg, self._TEXT)
        self.assertEqual([c.get("type") for c in b["messages"][0]["content"]],
                         ["text", "image_url"])
        _, b = Model("claude-sonnet-4-6", api_key="k").to_request(msg, self._TEXT)
        self.assertEqual([c.get("type") for c in b["messages"][0]["content"]],
                         ["text", "image"])
        _, b = Model("gemini-2.5-flash", api_key="k").to_request(msg, self._TEXT)
        self.assertEqual([list(p.keys())[0] for p in b["contents"][0]["parts"]],
                         ["text", "inlineData"])

    def test_video_input_handling(self):
        vid = [{"role": "user", "parts": [
            {"type": "video",
             "source": {"kind": "url", "url": "http://v.mp4", "mime": "video/mp4"}}]}]
        _, bg = Model("gemini-2.5-flash", api_key="k").to_request(vid, self._TEXT)
        self.assertEqual([list(p.keys())[0] for p in bg["contents"][0]["parts"]],
                         ["fileData"])
        _, bo = Model("gpt-4o", api_key="k").to_request(vid, self._TEXT)
        self.assertEqual(bo["messages"], [])

    def test_image_response_extraction(self):
        r = Model("gpt-image-1", api_key="k").from_response(
            {"data": [{"b64_json": "IMG"}]}, self._IMG)
        self.assertEqual(r["base64"], "IMG")
        self.assertEqual(r["mime_type"], "image/png")
        rg = Model("gemini-3.1-flash-image-preview", api_key="k").from_response(
            {"candidates": [{"content": {"parts": [
                {"inlineData": {"mimeType": "image/png", "data": "IMG"}}]}}]},
            self._IMG)
        self.assertEqual(rg["base64"], "IMG")

    def test_capabilities_preserved(self):
        from yait_aichain.models import registry
        self.assertFalse(registry.is_supported("deepseek-chat", "text-to-image"))
        self.assertFalse(registry.is_supported("sonar-pro", "text-to-image"))
        self.assertTrue(registry.is_supported("gpt-image-1", "text-to-image"))
        self.assertTrue(registry.is_supported("gpt-4o", "image-to-text"))
        self.assertTrue(registry.is_supported("claude-sonnet-4-6", "image-to-text"))


if __name__ == "__main__":
    unittest.main()
