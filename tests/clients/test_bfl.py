"""
Tests for the BFL / FLUX provider (async image generation + Kontext editing).

BFL uses an x-key header, the model name as the endpoint path, and an
asynchronous submit → poll → download flow that QwenClient-style reshapes the
result into {"data": [{"b64_json": ...}]}. Written to the BFL API reference;
transport is mocked.
"""

import base64
import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from clients._errors import TaskFailedError
from clients._families.bfl import BFLClient, _flux_image_value
from models._data import PROVIDERS as _PROVIDERS
from models import Model, registry

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
_B64 = base64.b64encode(_PNG).decode("ascii")
_OUT = {"modalities": ["image"], "format": {"type": "image"}}


def _img(kind="base64", **kw):
    return {"type": "image", "source": {"kind": kind, **kw}}


def _edit_msgs(*srcs):
    parts = list(srcs) or [_img(data=_B64, mime="image/png")]
    parts.append({"type": "text", "text": "Make it watercolour"})
    return [{"role": "user", "parts": parts}]


_TEXT_MSGS = [{"role": "user", "parts": [{"type": "text", "text": "a red bicycle"}]}]


def _client():
    return BFLClient("k", data=_PROVIDERS["bfl"])


class TestBuildRequest(unittest.TestCase):

    def test_auth_header_is_x_key(self):
        h = _client()._auth_headers()
        self.assertEqual(h["x-key"], "k")
        self.assertNotIn("Authorization", h)

    def test_generation_path_and_size(self):
        m = Model("flux-pro-1.1", api_key="k")
        path, body = m.to_request(_TEXT_MSGS, {"format": {"type": "image", "size": "1024x768"}})
        self.assertEqual(path, "/v1/flux-pro-1.1")
        self.assertEqual(body["prompt"], "a red bicycle")
        self.assertEqual((body["width"], body["height"]), (1024, 768))
        self.assertNotIn("input_image", body)

    def test_kontext_edit_input_image(self):
        m = Model("flux-kontext-pro", api_key="k")
        path, body = m.to_request(_edit_msgs(), _OUT)
        self.assertEqual(path, "/v1/flux-kontext-pro")
        self.assertEqual(body["input_image"], _B64)        # raw base64, not a data URI
        self.assertNotIn("width", body)                    # kontext uses aspect_ratio, not w/h

    def test_multi_reference(self):
        m = Model("flux-kontext-max", api_key="k")
        _, body = m.to_request(_edit_msgs(_img(data=_B64), _img(data=_B64)), _OUT)
        self.assertIn("input_image", body)
        self.assertIn("input_image_2", body)

    def test_url_source_passthrough(self):
        self.assertEqual(_flux_image_value({"kind": "url", "url": "https://x/y.png"}),
                         "https://x/y.png")


class TestAsyncFlow(unittest.TestCase):

    def _ready_client(self, sample="https://img/out.png"):
        c = _client()
        c._POLL_INTERVAL = 0
        c._post = MagicMock(return_value=json.dumps(
            {"id": "t-1", "polling_url": "https://api.bfl.ai/v1/get_result?id=t-1"}).encode())
        c._download = MagicMock(side_effect=[
            {"data": json.dumps({"status": "Ready", "result": {"sample": sample}}).encode(),
             "media_type": "application/json"},
            {"data": _PNG, "media_type": "image/png"},
        ])
        return c

    def test_submit_poll_download_happy_path(self):
        c = self._ready_client()
        raw = c.send("/v1/flux-kontext-pro", {"prompt": "x"}, c._auth_headers())
        data = json.loads(raw)
        self.assertEqual(data["data"][0]["b64_json"], _B64)
        # polled the polling_url, then downloaded the sample url
        self.assertEqual(c._download.call_args_list[0][0][0],
                         "https://api.bfl.ai/v1/get_result?id=t-1")
        self.assertEqual(c._download.call_args_list[1][0][0], "https://img/out.png")

    def test_no_polling_url_raises(self):
        c = _client()
        c._post = MagicMock(return_value=json.dumps({"id": "t-1"}).encode())
        with self.assertRaises(TaskFailedError):
            c.send("/v1/flux-2-pro", {"prompt": "x"}, {})

    def test_failed_status_raises(self):
        c = _client()
        c._POLL_INTERVAL = 0
        c._post = MagicMock(return_value=json.dumps(
            {"id": "t", "polling_url": "https://api.bfl.ai/v1/get_result?id=t"}).encode())
        c._download = MagicMock(return_value={
            "data": json.dumps({"status": "Content Moderated"}).encode(),
            "media_type": "application/json"})
        with self.assertRaises(TaskFailedError):
            c.send("/v1/flux-kontext-pro", {"prompt": "x"}, {})

    def test_end_to_end_through_skill(self):
        from skills import Skill
        m = Model("flux-kontext-pro", api_key="k")
        m.client = self._ready_client()
        img = Skill(model=m, input={"messages": _edit_msgs()}, output=_OUT).run()
        self.assertEqual(img["base64"], _B64)


class TestBflMisc(unittest.TestCase):

    def test_provider_resolves(self):
        self.assertEqual(Model("flux-kontext-pro", api_key="k")._provider, "bfl")

    def test_registry_caps(self):
        self.assertIn("flux-kontext-pro", registry.models(task="image-to-image"))
        self.assertIn("flux-pro-1.1", registry.models(task="text-to-image"))


if __name__ == "__main__":
    unittest.main()
