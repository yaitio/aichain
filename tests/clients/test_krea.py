"""
Krea: a request is built from the model's own path, ratios and tiers; a job is
submitted, polled and downloaded; its price is read from Krea's list, since a
job reports none; Krea 2 refuses an edit it would not follow.
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

from yait_aichain import Model
from yait_aichain.clients._errors import TaskFailedError
from yait_aichain.tools import image

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
MSG = [{"role": "user", "parts": [{"type": "text", "text": "a cat"}]}]


def _built(name, **fmt):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Model(name, api_key="k").build(MSG, {"modalities": ["image"],
                                                    "format": {"type": "image", **fmt}})


def _png_file():
    path = os.path.join(tempfile.mkdtemp(), "in.png")
    with open(path, "wb") as fh:
        fh.write(PNG_1X1)
    return path


class TestTheRequest(unittest.TestCase):

    def test_each_model_has_its_path(self):
        self.assertEqual(_built("krea-2-turbo")[0], "/generate/image/krea/krea-2/medium-turbo")
        self.assertEqual(_built("krea-3")[0], "/generate/image/krea/krea-3")

    def test_a_ratio_it_takes_goes_as_it_is(self):
        _, body, made = _built("krea-2-medium", aspect_ratio="4:5", seed=7)
        self.assertEqual((body["aspect_ratio"], body["seed"], body["resolution"]), ("4:5", 7, "1K"))
        self.assertEqual(made, [])

    def test_a_ratio_it_lacks_goes_as_the_nearest_and_is_said(self):
        _, body, made = _built("krea-2-medium", aspect_ratio="21:9")
        self.assertEqual(body["aspect_ratio"], "2.35:1")
        self.assertEqual([(a.kind, a.option) for a in made], [("adapted", "aspect_ratio")])

    def test_a_size_becomes_a_ratio_and_a_tier(self):
        _, body, made = _built("krea-3", size="2048x1152")
        self.assertEqual((body["aspect_ratio"], body["resolution"]), ("16:9", "2K"))
        self.assertEqual([(a.kind, a.option) for a in made], [("adapted", "size")])

    def test_krea_3_frames_by_itself_when_no_shape_is_asked(self):
        self.assertNotIn("aspect_ratio", _built("krea-3")[1])
        self.assertEqual(_built("krea-2-medium")[1]["aspect_ratio"], "1:1")

    def test_a_ratio_that_is_not_one_is_refused(self):
        with self.assertRaises(ValueError):
            _built("krea-2-medium", aspect_ratio="wide")


class TestAJob(unittest.TestCase):

    def _run(self, model, jobs, images=None, **options):
        tool = image("krea", model=model, api_key="k")
        client = tool._model.client
        sent = []

        def post(path, body, headers):
            sent.append((path, body))
            return json.dumps({"job_id": "j1", "status": "queued"}).encode()

        polls = iter(jobs)
        with patch.object(client, "_post", side_effect=post), \
                patch.object(client, "_get", side_effect=lambda *a, **k: json.dumps(next(polls)).encode()), \
                patch.object(client, "_download", return_value={"data": PNG_1X1}), \
                patch("time.sleep"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = tool.run("a cat", images=images, options=options or None)
        return result, sent

    def test_polled_until_completed_then_downloaded(self):
        result, sent = self._run("krea-2-medium", [
            {"job_id": "j1", "status": "processing"},
            {"job_id": "j1", "status": "completed", "result": {"urls": ["https://gen.krea.ai/a.png"]}}])
        self.assertEqual(result["images"][0]["mime_type"], "image/png")
        self.assertEqual(sent[0][0], "/generate/image/krea/krea-2/medium")
        self.assertEqual(result["cost"], 0.03)

    def test_a_failed_job_says_why(self):
        with self.assertRaisesRegex(TaskFailedError, "content moderated"):
            self._run("krea-2-medium", [{"job_id": "j1", "status": "failed",
                                         "error": {"code": "nsfw", "message": "content moderated"}}])

    def test_an_edit_on_krea_3_sends_the_image_and_is_priced_by_its_tiers(self):
        result, sent = self._run("krea-3", [
            {"job_id": "j1", "status": "completed", "result": {"urls": ["https://gen.krea.ai/a.png"]}}],
            images=[_png_file()], size="2048x2048")
        body = sent[0][1]
        self.assertTrue(body["image_urls"][0].startswith("data:image/png;base64,"))
        self.assertEqual(body["resolution"], "2K")
        self.assertAlmostEqual(result["cost"], 0.06 + 0.02)    # a 1x1 reference is priced at 1K

    def test_krea_2_refuses_an_edit(self):
        with self.assertRaisesRegex(ValueError, "does not edit.*krea-3"):
            self._run("krea-2-medium", [], images=[_png_file()])


class TestKreaLive(unittest.TestCase):
    """One generation against the real API: about $0.015."""

    @unittest.skipUnless(os.getenv("KREA_API_KEY"), "KREA_API_KEY not set")
    def test_turbo_makes_an_image(self):
        result = image("krea", model="krea-2-turbo").run("a red apple", options={"aspect_ratio": "4:5"})
        self.assertEqual(result["images"][0]["mime_type"], "image/png")
        self.assertEqual(result["cost"], 0.015)


if __name__ == "__main__":
    unittest.main()
