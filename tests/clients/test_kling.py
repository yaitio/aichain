"""
Kling: images through the older API — model_name, a task polled at
/v1/images/generations/{id}, an input image as bare base64 — and video through
the newer one, which video() covers in tests/tools/test_video.py.
"""

import base64
import json
import os
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.clients._errors import TaskFailedError
from yait_aichain.tools import image

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def _png():
    path = os.path.join(tempfile.mkdtemp(), "in.png")
    with open(path, "wb") as fh:
        fh.write(PNG)
    return path


def _run(model, polls, images=None, options=None, submitted=None):
    tool = image("kling", model=model, api_key="k")
    client, sent = tool._model.client, []

    def post(path, body, headers):
        sent.append((path, body))
        return json.dumps(submitted or {"code": 0, "data": {"task_id": "t1", "task_status": "submitted"}}).encode()
    answers = iter(polls)
    with patch.object(client, "_post", side_effect=post), \
            patch.object(client, "_get", side_effect=lambda *a, **k: json.dumps(next(answers)).encode()), \
            patch.object(client, "_download", return_value={"data": PNG}), \
            patch("time.sleep"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = tool.run("a teal box", images=images, options=options)
    return result, sent


DONE = {"code": 0, "data": {"task_status": "succeed",
                            "task_result": {"images": [{"index": 0, "url": "https://k/i.png"}]}}}


class TestKlingImages(unittest.TestCase):

    def test_a_task_is_submitted_polled_and_downloaded(self):
        result, sent = _run("kling-v3", [{"code": 0, "data": {"task_status": "processing"}}, DONE],
                            options={"aspect_ratio": "4:5"})
        path, body = sent[0]
        self.assertEqual(path, "/v1/images/generations")
        self.assertEqual((body["model_name"], body["aspect_ratio"], body["n"]), ("kling-v3", "3:4", 1))
        self.assertEqual(result["cost"], 0.028)
        self.assertTrue(any(n.startswith("aspect_ratio: adapted") for n in result["notices"]))

    def test_an_input_image_goes_as_bare_base64_and_costs_twice_on_2_1(self):
        result, sent = _run("kling-v2-1", [DONE], images=[_png()])
        self.assertFalse(sent[0][1]["image"].startswith("data:"))
        self.assertEqual(base64.b64decode(sent[0][1]["image"]), PNG)
        self.assertEqual(result["cost"], 0.028)

    def test_a_refusal_and_a_failure_say_why(self):
        with self.assertRaisesRegex(TaskFailedError, "insufficient balance"):
            _run("kling-v3", [], submitted={"code": 1102, "message": "insufficient balance"})
        with self.assertRaisesRegex(TaskFailedError, "risk control"):
            _run("kling-v3", [{"code": 0, "data": {"task_status": "failed", "task_status_msg": "risk control"}}])


class TestKlingImagesLive(unittest.TestCase):
    """One Kling Image 3.0 picture: $0.028."""

    @unittest.skipUnless(os.getenv("KLING_API_KEY"), "KLING_API_KEY not set")
    def test_a_picture(self):
        result = image("kling", model="kling-v3").run("a red apple on a white table")
        self.assertEqual(result["images"][0]["mime_type"], "image/png")
        self.assertEqual(result["cost"], 0.028)


if __name__ == "__main__":
    unittest.main()
