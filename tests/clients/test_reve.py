"""
Tests for the Reve image provider (create / edit / remix).

Request shapes are exercised through the real Model → ReveClient seam; the
response parser is checked directly. No network.
"""

import base64
import json
import unittest
from unittest.mock import MagicMock

from yait_aichain.models import Model
from yait_aichain.clients._families.reve import ReveClient, _parse_reve_response

_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n....fake").decode()
_IMG_OUT = {"modalities": ["image"], "format": {"type": "image", "aspect_ratio": "16:9"}}


def _user(*parts):
    return [{"role": "user", "parts": list(parts)}]


def _text(t):
    return {"type": "text", "text": t}


def _image(b64=_B64, kind="base64"):
    src = {"kind": kind, "data": b64, "mime": "image/png"} if kind != "url" \
        else {"kind": "url", "url": b64}
    return {"type": "image", "source": src}


class TestReveRouting(unittest.TestCase):

    def setUp(self):
        self.m = Model("reve-image", api_key="k")

    def test_provider_resolves(self):
        self.assertEqual(self.m._provider, "reve")
        self.assertEqual(Model("reve/reve-image", api_key="k")._provider, "reve")

    def test_create_no_image(self):
        path, body = self.m.to_request(_user(_text("a stack of pancakes")), _IMG_OUT)
        self.assertEqual(path, "/v1/image/create")
        self.assertEqual(body["prompt"], "a stack of pancakes")
        self.assertEqual(body["version"], "latest")
        self.assertEqual(body["aspect_ratio"], "16:9")
        self.assertNotIn("reference_image", body)

    def test_edit_single_image(self):
        path, body = self.m.to_request(
            _user(_text("make it impasto"), _image()), _IMG_OUT)
        self.assertEqual(path, "/v1/image/edit")
        self.assertEqual(body["edit_instruction"], "make it impasto")
        self.assertEqual(body["reference_image"], _B64)        # raw base64, no data: prefix
        self.assertNotIn("prompt", body)

    def test_remix_multiple_images(self):
        path, body = self.m.to_request(
            _user(_text("style <img>0</img> onto <img>1</img>"), _image(), _image()),
            _IMG_OUT)
        self.assertEqual(path, "/v1/image/remix")
        self.assertEqual(len(body["reference_images"]), 2)
        self.assertEqual(body["reference_images"][0], _B64)
        self.assertIn("<img>0</img>", body["prompt"])

    def test_version_override(self):
        out = {"modalities": ["image"], "format": {"type": "image", "version": "1.0"}}
        _, body = self.m.to_request(_user(_text("x")), out)
        self.assertEqual(body["version"], "1.0")

    def test_url_reference_passed_through(self):
        path, body = self.m.to_request(
            _user(_text("edit"), _image("https://x/y.png", kind="url")), _IMG_OUT)
        self.assertEqual(body["reference_image"], "https://x/y.png")

    def test_optional_params_forwarded(self):
        out = {"modalities": ["image"], "format": {
            "type": "image",
            "postprocessing": [{"process": "upscale", "upscale_factor": 2}],
            "test_time_scaling": 3,
        }}
        _, body = self.m.to_request(_user(_text("x")), out)
        self.assertEqual(body["postprocessing"], [{"process": "upscale", "upscale_factor": 2}])
        self.assertEqual(body["test_time_scaling"], 3)

    def test_optional_params_omitted_by_default(self):
        _, body = self.m.to_request(_user(_text("x")), _IMG_OUT)
        self.assertNotIn("postprocessing", body)
        self.assertNotIn("test_time_scaling", body)


class TestReveAuth(unittest.TestCase):

    def test_bearer_header(self):
        c = ReveClient("secret", data=Model("reve-image", api_key="k").client._data)
        h = c._auth_headers()
        self.assertEqual(h["Authorization"], "Bearer secret")
        self.assertEqual(h["Accept"], "application/json")


class TestReveResponseParser(unittest.TestCase):

    def test_native_image_key(self):
        out = _parse_reve_response({"image": _B64})
        self.assertEqual(out["base64"], _B64)
        self.assertEqual(out["mime_type"], "image/png")

    def test_images_list(self):
        self.assertEqual(_parse_reve_response({"images": [_B64]})["base64"], _B64)

    def test_openai_shaped_data(self):
        self.assertEqual(
            _parse_reve_response({"data": [{"b64_json": _B64}]})["base64"], _B64)

    def test_url_response(self):
        out = _parse_reve_response({"url": "https://x/y.png"})
        self.assertEqual(out["url"], "https://x/y.png")
        self.assertIsNone(out["base64"])

    def test_content_violation_raises(self):
        with self.assertRaises(ValueError):
            _parse_reve_response({"content_violation": True})

    def test_unknown_shape_raises(self):
        with self.assertRaises(ValueError):
            _parse_reve_response({"unexpected": 1})


class TestReveSendSeam(unittest.TestCase):

    def test_send_posts_and_parses(self):
        m = Model("reve-image", api_key="k")
        m.client._post = MagicMock(return_value=json.dumps({"image": _B64}).encode())
        path, body = m.to_request(_user(_text("pancakes")), _IMG_OUT)
        raw = m.client.send(path, body, m.client._auth_headers())
        out = m.from_response(json.loads(raw), _IMG_OUT)
        self.assertEqual(out["base64"], _B64)


if __name__ == "__main__":
    unittest.main()
