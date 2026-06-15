"""
Tests for QwenClient text-to-image (DashScope native async task API).

The wanx models do not run on the OpenAI-compatible endpoint — QwenClient
submits a job, polls until it finishes, then downloads the image and shapes the
result like a synchronous images response.  These tests mock the three transport
primitives (`_post`, `_get`, `_download`) so no network is touched.
"""

import base64
import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from clients._errors import ServerError
from clients._families.qwen import (
    QwenClient as _Family,
    _IMAGE_SYNTHESIS_PATH,
    _TASKS_PATH,
    _build_qwen_image_request,
)
from models._data import PROVIDERS as _PROVIDERS
from models import Model
from skills import Skill

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _client(**kw):
    return _Family("k", data=_PROVIDERS["qwen"], **kw)


def _wrap(name):
    from types import SimpleNamespace
    return SimpleNamespace(name=name)


class TestBuildRequest(unittest.TestCase):

    def test_native_path_and_size_conversion(self):
        path, body = _build_qwen_image_request(
            _wrap("wan2.2-t2i-flash"),
            [{"role": "user", "parts": [{"type": "text", "text": "a cat"}]}],
            {"format": {"type": "image", "size": "1280x720"}},
        )
        self.assertEqual(path, _IMAGE_SYNTHESIS_PATH)
        self.assertEqual(body["model"], "wan2.2-t2i-flash")
        self.assertEqual(body["input"]["prompt"], "a cat")
        self.assertEqual(body["parameters"]["size"], "1280*720")  # x → *

    def test_size_omitted_when_absent(self):
        _, body = _build_qwen_image_request(
            _wrap("wan2.2-t2i-plus"),
            [{"role": "user", "parts": [{"type": "text", "text": "x"}]}],
            {"format": {"type": "image"}},
        )
        self.assertNotIn("size", body["parameters"])

    def test_build_request_routes_wanx_to_native(self):
        c = _client()
        path, _ = c.build_request(
            [{"role": "user", "parts": [{"type": "text", "text": "x"}]}],
            {"modalities": ["image"], "format": {"type": "image"}},
            {"name": "wan2.2-t2i-flash"},
        )
        self.assertEqual(path, _IMAGE_SYNTHESIS_PATH)


class TestSynthesisFlow(unittest.TestCase):

    def _client_with_task(self, *poll_statuses, results=None):
        """A client whose transport returns a submit id, then the given polls."""
        c = _client()
        c._POLL_INTERVAL = 0  # no real sleeping
        c._post = MagicMock(return_value=json.dumps(
            {"output": {"task_id": "t-1", "task_status": "PENDING"}}).encode())
        polls = []
        for st in poll_statuses:
            out = {"task_id": "t-1", "task_status": st}
            if st == "SUCCEEDED":
                out["results"] = results if results is not None else [{"url": "https://img/1.png"}]
            polls.append(json.dumps({"output": out}).encode())
        c._get = MagicMock(side_effect=polls)
        c._download = MagicMock(return_value={"data": _PNG, "media_type": "image/png"})
        c._auth_headers = MagicMock(return_value={"Authorization": "Bearer k"})
        return c

    def test_submit_poll_download_happy_path(self):
        c = self._client_with_task("PENDING", "SUCCEEDED")
        raw = c.send(_IMAGE_SYNTHESIS_PATH, {"model": "wan2.2-t2i-flash"}, {})
        data = json.loads(raw)
        self.assertEqual(data["data"][0]["b64_json"],
                         base64.b64encode(_PNG).decode("ascii"))
        # submit carried the async header
        self.assertEqual(c._post.call_args[0][0], _IMAGE_SYNTHESIS_PATH)
        self.assertEqual(c._post.call_args[0][2]["X-DashScope-Async"], "enable")
        # polled the task endpoint twice; downloaded the result url
        self.assertEqual(c._get.call_count, 2)
        self.assertTrue(c._get.call_args_list[0][0][0].startswith(_TASKS_PATH))
        c._download.assert_called_once_with("https://img/1.png")

    def test_end_to_end_through_skill(self):
        c = self._client_with_task("SUCCEEDED")
        m = Model("wan2.2-t2i-flash", api_key="k")
        m.client = c
        img = Skill(
            model=m,
            input={"messages": [{"role": "user", "parts": ["a fox"]}]},
            output={"modalities": ["image"], "format": {"type": "image", "size": "1024x1024"}},
        ).run()
        self.assertEqual(img["base64"], base64.b64encode(_PNG).decode("ascii"))
        self.assertEqual(img["mime_type"], "image/png")

    def test_failed_task_raises(self):
        c = self._client_with_task("FAILED")
        with self.assertRaises(ServerError) as ctx:
            c.send(_IMAGE_SYNTHESIS_PATH, {"model": "wan2.2-t2i-flash"}, {})
        self.assertIn("FAILED", str(ctx.exception))

    def test_timeout_raises(self):
        c = self._client_with_task("PENDING")
        c._POLL_TIMEOUT = 0  # deadline already passed after the first poll
        with self.assertRaises(ServerError) as ctx:
            c.send(_IMAGE_SYNTHESIS_PATH, {"model": "wan2.2-t2i-flash"}, {})
        self.assertEqual(ctx.exception.status, 504)

    def test_no_results_raises(self):
        c = self._client_with_task("SUCCEEDED", results=[])
        with self.assertRaises(ServerError):
            c.send(_IMAGE_SYNTHESIS_PATH, {"model": "wan2.2-t2i-flash"}, {})


class TestSendRouting(unittest.TestCase):

    def test_non_image_path_is_single_post(self):
        c = _client()
        c._post = MagicMock(return_value=b"{}")
        out = c.send("/compatible-mode/v1/chat/completions", {"model": "qwen-max"}, {})
        self.assertEqual(out, b"{}")
        c._post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
