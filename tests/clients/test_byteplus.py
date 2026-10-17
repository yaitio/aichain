"""
BytePlus ModelArk: Seed text through the OpenAI-compatible family; Seedream
images as a pixel size inside each model's range, priced by the size that came
back. Seedance video is covered in tests/tools/test_video.py.
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

from yait_aichain import Model, Skill
from yait_aichain.models._adaptation import collect
from yait_aichain.tools import image

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def _png():
    path = os.path.join(tempfile.mkdtemp(), "in.png")
    with open(path, "wb") as fh:
        fh.write(PNG)
    return path


def _picture(model, size, images=None, options=None):
    """One image() call on *model*, the API answering with a picture of *size*."""
    tool = image("byteplus", model=model, api_key="k")
    sent = []

    def post(path, body, headers):
        sent.append((path, body))
        return json.dumps({"model": model, "data": [{"b64_json": base64.b64encode(PNG).decode(), "size": size}],
                           "usage": {"generated_images": 1, "output_tokens": 3600}}).encode()
    with patch.object(tool._model.client, "_post", side_effect=post), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = tool.run("a teal box", images=images, options=options)
    return result, sent


class TestSeedText(unittest.TestCase):

    def _request(self, options=None, fmt=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Model("seed-2-0-mini-260428", api_key="k", options=options)
            return m, *m.to_request([{"role": "user", "parts": [{"type": "text", "text": "hi"}]}],
                                    {"format": fmt or {"type": "text"}})

    def test_the_seed_models_route_here_and_speak_chat_completions(self):
        m, path, body = self._request()
        self.assertEqual((m._provider, path), ("byteplus", "/api/v3/chat/completions"))
        self.assertIn("max_completion_tokens", body)

    def test_no_reasoning_goes_as_minimal(self):
        # "none" turns the reasoning off and the answer runs long; minimal answers directly.
        self.assertEqual(self._request({"reasoning": "none"})[2]["reasoning_effort"], "minimal")
        self.assertEqual(self._request({"reasoning": "high"})[2]["reasoning_effort"], "high")

    def test_a_schema_goes_as_one(self):
        schema = {"type": "json_schema", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}
        self.assertEqual(self._request(fmt=schema)[2]["response_format"]["type"], "json_schema")


class TestSeedream(unittest.TestCase):

    def test_no_shape_asked_sends_a_tier_and_turns_the_watermark_off(self):
        result, sent = _picture("dola-seedream-5-0-pro-260628", "1536x1536")
        path, body = sent[0]
        self.assertEqual(path, "/api/v3/images/generations")
        self.assertEqual((body["size"], body["watermark"], body["response_format"]), ("1.5K", False, "b64_json"))
        self.assertNotIn("sequential_image_generation", body)              # 5.0 pro has none
        self.assertEqual(result["cost"], 0.045)

    def test_a_ratio_is_drawn_at_the_base_size(self):
        _, sent = _picture("seedream-4-0-250828", "1360x768", options={"aspect_ratio": "16:9"})
        self.assertEqual(sent[0][1]["size"], "1360x768")
        self.assertEqual(sent[0][1]["sequential_image_generation"], "disabled")

    def test_a_size_outside_the_range_is_scaled_into_it_and_said(self):
        with collect() as made:
            result, sent = _picture("seedream-4-5-251128", "1920x1920", options={"size": "1024x1024"})
        w, h = map(int, sent[0][1]["size"].split("x"))
        self.assertEqual(w, h)
        self.assertGreaterEqual(w * h, 3686400)
        self.assertTrue(any(n.startswith("size: adapted") for n in result["notices"]))

    def test_the_price_follows_the_size_that_came_back(self):
        result, _ = _picture("dola-seedream-5-0-pro-260628", "2048x2048", options={"size": "2048x2048"})
        self.assertEqual(result["cost"], 0.09)                   # above 2.61 million pixels

    def test_input_images_go_as_data_uris_and_the_second_is_charged(self):
        result, sent = _picture("dola-seedream-5-0-pro-260628", "1536x1536", images=[_png(), _png()])
        self.assertTrue(all(i.startswith("data:image/png;base64,") for i in sent[0][1]["image"]))
        self.assertEqual(result["cost"], 0.048)


class TestByteplusLive(unittest.TestCase):
    """Seed 2.0 Mini, a fraction of a cent; one Seedream 4.0 picture, $0.03."""

    @unittest.skipUnless(os.getenv("BYTEPLUS_API_KEY"), "BYTEPLUS_API_KEY not set")
    def test_text(self):
        skill = Skill(Model("seed-2-0-mini-260428", options={"reasoning": "none"}),
                      prompt="What is 17*23? Answer with the number only.")
        self.assertIn("391", skill.run())

    @unittest.skipUnless(os.getenv("BYTEPLUS_API_KEY"), "BYTEPLUS_API_KEY not set")
    def test_a_picture(self):
        result = image("byteplus", model="seedream-4-0-250828").run("a red apple on a white table")
        self.assertEqual(result["cost"], 0.03)


if __name__ == "__main__":
    unittest.main()
