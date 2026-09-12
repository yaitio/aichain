"""
Tests for the Recraft provider (image generation + image-to-image).

Recraft rides the OpenAI Bearer transport: text-to-image on
/v1/images/generations (JSON), image-to-image on /v1/images/imageToImage
(multipart). Written to the Recraft API reference; transport is mocked.
"""

import base64
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.clients._families.recraft import RecraftClient
from yait_aichain.models._data import PROVIDERS as _PROVIDERS
from yait_aichain.models import Model, registry

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
_B64 = base64.b64encode(_PNG).decode("ascii")
_OUT = {"modalities": ["image"], "format": {"type": "image", "size": "1024x1024"}}


def _img(kind="base64", **kw):
    return {"type": "image", "source": {"kind": kind, **kw}}


def _edit_msgs(src=None):
    return [{"role": "user", "parts": [
        src or _img(data=_B64, mime="image/png"),
        {"type": "text", "text": "Make the background a sunset"},
    ]}]


_TEXT_MSGS = [{"role": "user", "parts": [{"type": "text", "text": "a red bicycle"}]}]


class TestRecraftGeneration(unittest.TestCase):

    def test_generation_shape(self):
        m = Model("recraftv4_1", api_key="k")
        path, body = m.to_request(_TEXT_MSGS, _OUT)
        self.assertEqual(path, "/v1/images/generations")
        self.assertEqual(body["model"], "recraftv4_1")
        self.assertEqual(body["prompt"], "a red bicycle")
        self.assertEqual(body["response_format"], "b64_json")
        self.assertEqual(body["size"], "1024x1024")
        self.assertNotIn("_multipart", body)

    def test_style_passthrough(self):
        m = Model("recraftv4_1", api_key="k")
        _, body = m.to_request(_TEXT_MSGS,
                               {"format": {"type": "image", "style": "realistic_image"}})
        self.assertEqual(body["style"], "realistic_image")


class TestRecraftEdit(unittest.TestCase):

    def test_edit_multipart_shape(self):
        m = Model("recraftv3", api_key="k")
        path, body = m.to_request(_edit_msgs(), _OUT)
        self.assertEqual(path, "/v1/images/imageToImage")
        self.assertTrue(body["_multipart"])
        names = [f[0] for f in body["fields"]]
        self.assertIn("image", names)
        self.assertIn("strength", names)
        self.assertIn("prompt", names)
        file_field = next(f for f in body["fields"] if f[0] == "image")
        self.assertEqual(file_field[1][1], _PNG)

    def test_strength_default_and_fidelity_inverted(self):
        """This API asks how much to change; the vocabulary asks how much to
        keep, so `fidelity` arrives here as its complement. Measured
        2026-09-09: strength 0.05 left a blue square blue and 0.95 replaced
        it with the prompt, which is what fixes the direction."""
        m = Model("recraftv3", api_key="k")
        _, body = m.to_request(_edit_msgs(), {"format": {"type": "image"}})
        self.assertEqual(dict(body["fields"])["strength"], "0.2")

        _, kept = m.to_request(_edit_msgs(),
                               {"format": {"type": "image", "fidelity": 0.9}})
        self.assertEqual(dict(kept["fields"])["strength"], "0.1")

        _, changed = m.to_request(_edit_msgs(),
                                  {"format": {"type": "image", "fidelity": 0.1}})
        self.assertEqual(dict(changed["fields"])["strength"], "0.9")

    def test_url_source_rejected(self):
        m = Model("recraftv3", api_key="k")
        with self.assertRaises(ValueError):
            m.to_request(_edit_msgs(_img("url", url="https://x/y.png")), _OUT)

    def test_multipart_send_drops_content_type(self):
        c = RecraftClient("k", data=_PROVIDERS["recraft"])
        c._post_form = MagicMock(return_value=b"{}")
        c.send("/v1/images/imageToImage", {"_multipart": True, "fields": [("model", "recraftv3")]},
               {"Authorization": "Bearer k", "Content-Type": "application/json"})
        c._post_form.assert_called_once()
        self.assertNotIn("Content-Type", c._post_form.call_args[0][2])


class TestRecraftMisc(unittest.TestCase):

    def test_provider_resolves(self):
        self.assertEqual(Model("recraftv3", api_key="k")._provider, "recraft")
        self.assertEqual(Model("recraft/recraftv3", api_key="k")._provider, "recraft")

    def test_parse_response(self):
        c = RecraftClient("k", data=_PROVIDERS["recraft"])
        out = c.parse_response({"data": [{"b64_json": _B64}]}, _OUT)
        self.assertEqual(out["base64"], _B64)

    def test_registry_caps(self):
        self.assertIn("recraftv3", registry.models(task="image-to-image"))
        self.assertIn("recraftv4_1", registry.models(task="text-to-image"))
        self.assertNotIn("recraftv4_1", registry.models(task="image-to-image"))

    def test_v4_1_variants_registered(self):
        # Live-verified V4.1 model names (incl. utility) — text-to-image only.
        t2i = registry.models(provider="recraft", task="text-to-image")
        for name in ("recraftv4_1_utility", "recraftv4_1_utility_pro",
                     "recraftv4_1_pro", "recraftv4_1_utility_vector"):
            self.assertIn(name, t2i)
            self.assertEqual(Model(name, api_key="k")._provider, "recraft")


class TestRecraftVectorize(unittest.TestCase):
    """Raster → SVG conversion (POST /v1/images/vectorize)."""

    def test_provider_resolves(self):
        # The broadened ^recraft prefix resolves the dash-named model.
        self.assertEqual(Model("recraft-vectorize", api_key="k")._provider, "recraft")

    def test_registered_under_image_to_image(self):
        self.assertIn("recraft-vectorize", registry.models(task="image-to-image"))

    def test_routes_to_vectorize_multipart_no_prompt(self):
        m = Model("recraft-vectorize", api_key="k")
        msgs = [{"role": "user", "parts": [_img(data=_B64, mime="image/png")]}]
        path, body = m.to_request(msgs, _OUT)
        self.assertEqual(path, "/v1/images/vectorize")
        self.assertTrue(body["_multipart"])
        names = [f[0] for f in body["fields"]]
        self.assertEqual(names, ["response_format", "image"])   # no model, no prompt

    def test_url_source_rejected(self):
        m = Model("recraft-vectorize", api_key="k")
        msgs = [{"role": "user", "parts": [_img(kind="url", url="https://x/y.png")]}]
        with self.assertRaises(ValueError):
            m.to_request(msgs, _OUT)

    def test_parse_vectorize_response(self):
        c = RecraftClient("k", data=_PROVIDERS["recraft"])
        # live shape: {"image": {"b64_json": ..., "image_id": ...}, "credits": N}
        svg_b64 = base64.b64encode(b'<svg xmlns="..."></svg>').decode()
        out = c.parse_response({"credits": 10,
                                "image": {"b64_json": svg_b64, "image_id": "x"}}, _OUT)
        self.assertEqual(out["base64"], svg_b64)
        self.assertEqual(out["mime_type"], "image/svg+xml")

    def test_generation_response_still_parsed(self):
        # The {"data": [...]} generation shape must still work (no regression).
        c = RecraftClient("k", data=_PROVIDERS["recraft"])
        out = c.parse_response({"data": [{"b64_json": _B64}]}, _OUT)
        self.assertEqual(out["base64"], _B64)


if __name__ == "__main__":
    unittest.main()
