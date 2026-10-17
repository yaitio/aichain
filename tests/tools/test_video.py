"""
video(): a job per call — submitted, polled, downloaded — with the image tool's
manners: one option vocabulary, every adaptation said, the cost on the result,
the bytes never shown to a model. The request shape of each provider is pinned
to what it accepted live on 2026-10-17.
"""

import json
import os
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain import Budget
from yait_aichain._budget import BudgetExceeded, lend
from yait_aichain.clients._errors import TaskFailedError
from yait_aichain.models import registry
from yait_aichain.models._adaptation import collect
from yait_aichain.models._usage import price_per_second
from yait_aichain.tools import video
from yait_aichain.tools.video import Video

PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


def _frame():
    path = os.path.join(tempfile.mkdtemp(), "frame.png")
    with open(path, "wb") as fh:
        fh.write(PNG)
    return path


def _run(tool, polls, images=None, options=None, budget=None):
    """Run *tool* against canned answers: the submit, then *polls* in order."""
    sent, client = [], tool._model.client

    def post(path, body, headers):
        sent.append((path, body, headers))
        return json.dumps({"name": "models/m/operations/op1", "request_id": "r1", "task_id": "t1",
                           "output": {"task_id": "t1"}}).encode()
    answers = iter(polls)
    with patch.object(client, "_post", side_effect=post), \
            patch.object(client, "_get", side_effect=lambda *a, **k: json.dumps(next(answers)).encode()), \
            patch.object(client, "_download", return_value={"data": MP4, "media_type": "video/mp4"}), \
            patch("time.sleep"), warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        if budget is not None:
            with lend(budget):
                result = tool.run("a paper boat", images=images, options=options)
        else:
            result = tool.run("a paper boat", images=images, options=options)
    return result, sent, [str(w.message) for w in warned]


def _plan(model, **options):
    tool = Video(model, api_key="k")
    with collect() as made, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = tool._plan(options, False)
    return plan, {(a.option, a.kind): a for a in made}


class TestThePlan(unittest.TestCase):

    def test_veo_gives_1080p_at_eight_seconds_and_says_so(self):
        plan, said = _plan("veo-3.1-lite-generate-preview", duration=5, resolution="1080p")
        self.assertEqual((plan["duration"], plan["resolution"]), (8, "1080p"))
        self.assertIn("8 seconds only", said["duration", "adapted"].why)

    def test_the_nearest_length_ratio_and_resolution(self):
        plan, said = _plan("grok-imagine-video-1.5-lite", duration=20, aspect_ratio="2:1", resolution="4k")
        self.assertEqual((plan["duration"], plan["aspect_ratio"], plan["resolution"]), (15, "16:9", "1080p"))
        self.assertEqual({k for k, _ in said}, {"duration", "aspect_ratio", "resolution"})

    def test_a_size_is_a_ratio_and_a_resolution_with_nothing_said_when_exact(self):
        plan, said = _plan("grok-imagine-video-1.5", size="1920x1080")
        self.assertEqual((plan["aspect_ratio"], plan["resolution"]), ("16:9", "1080p"))
        self.assertEqual(said, {})

    def test_what_a_model_cannot_do_is_declined(self):
        _, said = _plan("veo-3.1-fast-generate-preview", audio=False)
        self.assertIn("always makes sound", said["audio", "declined"].why)
        _, said = _plan("grok-imagine-video-1.5", seed=7, negative_prompt="blur")
        self.assertEqual({k for k, _ in said}, {"seed", "negative_prompt"})

    def test_defaults_are_a_short_720p_landscape(self):
        plan, said = _plan("wan2.7-t2v")
        self.assertEqual((plan["duration"], plan["resolution"], plan["aspect_ratio"]), (5, "720p", "16:9"))
        self.assertEqual(said, {})


class TestRefusedBeforeAnything(unittest.TestCase):

    def test_what_cannot_be_asked(self):
        with self.assertRaisesRegex(ValueError, "Sora"):
            video("openai", model="sora-2")
        with self.assertRaisesRegex(ValueError, "not a video model"):
            video("google", model="gemini-3.8-flash", api_key="k")
        with self.assertRaisesRegex(ValueError, "is qwen's"):
            video("google", model="wan2.7-t2v", api_key="k")
        tool = video("qwen", model="wan2.7-i2v", api_key="k")
        with self.assertRaisesRegex(ValueError, "from an image"):
            tool.submit("a boat")
        with self.assertRaisesRegex(ValueError, "unknown option"):
            tool.submit("a boat", [_frame()], {"fps": 24})
        with self.assertRaisesRegex(ValueError, "one image"):
            tool.submit("a boat", [_frame(), _frame()])


class TestEachProvidersRequest(unittest.TestCase):

    def test_google_sends_a_long_running_prediction_and_downloads_with_the_key(self):
        tool = video("google", model="veo-3.1-lite-generate-preview", api_key="k")
        done = {"done": True, "response": {"generateVideoResponse": {
            "generatedSamples": [{"video": {"uri": "https://g/files/v:download"}}]}}}
        result, sent, _ = _run(tool, [{"done": False}, done], images=[_frame()], options={"duration": 4})
        path, body, _ = sent[0]
        self.assertEqual(path, "/models/veo-3.1-lite-generate-preview:predictLongRunning")
        self.assertEqual(body["parameters"], {"aspectRatio": "16:9", "durationSeconds": 4, "resolution": "720p"})
        self.assertEqual(set(body["instances"][0]["image"]), {"bytesBase64Encoded", "mimeType"})
        self.assertEqual((result["cost"], result["videos"][0]["data"]), (0.2, MP4))

    def test_xai_reports_its_own_cost(self):
        tool = video("xai", model="grok-imagine-video-1.5-lite", api_key="k")
        done = {"status": "done", "video": {"url": "https://x/v.mp4", "duration": 1},
                "usage": {"cost_in_usd_ticks": 200_000_000}}
        result, sent, _ = _run(tool, [{"status": "pending", "progress": 40}, done],
                               images=[_frame()], options={"duration": 1, "resolution": "480p"})
        body = sent[0][1]
        self.assertEqual((body["duration"], body["resolution"]), (1, "480p"))
        self.assertTrue(body["image"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual((result["cost"], result["cost_basis"]), (0.02, "xai reported it"))

    def test_qwen_wan_2_2_takes_a_pixel_size_and_wan_2_7_a_resolution(self):
        done = {"output": {"task_status": "SUCCEEDED", "video_url": "https://q/v.mp4"},
                "usage": {"video_duration": 5, "video_ratio": "480*832"}}
        result, sent, _ = _run(video("qwen", model="wan2.2-t2v-plus", api_key="k"),
                               [{"output": {"task_status": "RUNNING"}}, done],
                               options={"resolution": "480p", "aspect_ratio": "9:16"})
        path, body, headers = sent[0]
        self.assertEqual(headers["X-DashScope-Async"], "enable")
        self.assertEqual(body["parameters"], {"size": "480*832"})
        self.assertEqual((result["videos"][0]["resolution"], result["cost"]), ("480p", 0.1))
        done = {"output": {"task_status": "SUCCEEDED", "video_url": "https://q/v.mp4"},
                "usage": {"output_video_duration": 2, "SR": 720}}
        _, sent, _ = _run(video("qwen", model="wan2.7-i2v", api_key="k"), [done],
                          images=[_frame()], options={"duration": 2})
        body = sent[0][1]
        self.assertEqual(body["parameters"], {"resolution": "720P", "duration": 2})   # the shape follows the frame
        self.assertEqual(body["input"]["media"][0]["type"], "first_frame")


class TestTheNewerModels(unittest.TestCase):

    def test_minimax_h3_sends_a_content_list_and_its_shape_follows_a_first_frame(self):
        done = {"task": {"id": "t1", "status": "succeeded", "content": {"url": "https://m/v.mp4"},
                         "usage": {"output_seconds": 4}}}
        result, sent, _ = _run(video("minimax", model="MiniMax-H3", api_key="k"),
                               [{"task": {"status": "running"}}, done], images=[_frame()],
                               options={"duration": 4, "resolution": "1080p"})
        path, body, _ = sent[0]
        self.assertEqual(path, "/v2/video_generation")
        self.assertEqual([c["type"] for c in body["content"]], ["text", "image_url"])
        self.assertEqual(body["content"][1]["role"], "first_frame")
        self.assertEqual((body["resolution"], body["duration"], body["ratio"]), ("768P", 4, "adaptive"))
        self.assertEqual((result["videos"][0]["resolution"], result["cost"]), ("768p", 0.32))
        self.assertTrue(any(n.startswith("resolution: adapted") for n in result["notices"]))

    def test_minimax_from_text_takes_a_ratio(self):
        done = {"task": {"status": "succeeded", "content": {"url": "https://m/v.mp4"}}}
        _, sent, _ = _run(video("minimax", model="MiniMax-H3-Max", api_key="k"), [done],
                          options={"aspect_ratio": "9:16"})
        self.assertEqual((sent[0][1]["ratio"], sent[0][1]["resolution"]), ("9:16", "768P"))

    def test_wan_3_takes_media_sound_and_no_ratio_beside_a_first_frame(self):
        done = {"output": {"task_status": "SUCCEEDED", "video_url": "https://q/v.mp4"},
                "usage": {"output_video_duration": 2, "SR": 480}}
        _, sent, _ = _run(video("qwen", model="wan3.0-video", api_key="k"), [done],
                          options={"resolution": "480p", "duration": 2, "audio": False})
        self.assertEqual(sent[0][1]["parameters"], {"resolution": "480P", "ratio": "16:9", "duration": 2,
                                                    "audio": False})
        result, sent, _ = _run(video("qwen", model="wan3.0-video", api_key="k"), [done], images=[_frame()],
                               options={"resolution": "480p", "duration": 2, "audio": True})
        self.assertEqual(sent[0][1]["parameters"], {"resolution": "480P", "duration": 2})
        self.assertEqual(sent[0][1]["input"]["media"][0]["type"], "first_frame")
        self.assertTrue(any("no sound from a first frame" in n for n in result["notices"]))


class TestTheJob(unittest.TestCase):

    def test_a_job_crosses_a_process_and_is_collected_by_another_tool(self):
        tool = video("xai", model="grok-imagine-video-1.5", api_key="k")
        with patch.object(tool._model.client, "_post", return_value=b'{"request_id": "r9"}'):
            job = tool.submit("a boat", options={"duration": 3})
        job = json.loads(json.dumps(job))
        other = video("xai", model="grok-imagine-video-1.5", api_key="k",
                      output_dir=tempfile.mkdtemp())
        done = {"status": "done", "video": {"url": "https://x/v.mp4", "duration": 3}}
        with patch.object(other._model.client, "_get", return_value=json.dumps(done).encode()), \
                patch.object(other._model.client, "_download", return_value={"data": MP4}):
            result = other.collect(job)
        self.assertTrue(os.path.isfile(result["videos"][0]["path"]))
        self.assertIsNone(result["videos"][0]["data"])
        self.assertEqual(result["cost"], 0.24)                     # 3 s at the registry's $0.08
        self.assertIn("saved to", other.render(result))

    def test_a_failed_job_says_why(self):
        tool = video("qwen", model="wan2.7-t2v", api_key="k")
        with self.assertRaisesRegex(TaskFailedError, "content moderation"):
            _run(tool, [{"output": {"task_status": "FAILED", "message": "content moderation"}}])

    def test_a_wait_that_runs_out_raises_with_the_job(self):
        tool = video("google", model="veo-3.1-lite-generate-preview", api_key="k", timeout=0)
        with self.assertRaises(TaskFailedError) as caught:
            _run(tool, [{"done": False}] * 3)
        self.assertEqual(caught.exception.job["id"], "models/m/operations/op1")
        self.assertIn("collect(job) again", str(caught.exception))


class TestTheBudget(unittest.TestCase):

    def test_an_estimate_above_what_is_left_warns_and_goes_ahead(self):
        tool = video("google", model="veo-3.1-generate-preview", api_key="k")
        done = {"done": True, "response": {"generateVideoResponse": {
            "generatedSamples": [{"video": {"uri": "https://g/v"}}]}}}
        budget = Budget(1.0)
        result, _, warned = _run(tool, [done], options={"duration": 8}, budget=budget)
        self.assertTrue(any("estimated at $3.20" in w for w in warned))
        self.assertTrue(any(n.startswith("budget:") for n in result["notices"]))
        self.assertAlmostEqual(budget.spent, 3.2)

    def test_a_budget_with_nothing_left_stops_the_job(self):
        tool = video("xai", model="grok-imagine-video-1.5-lite", api_key="k")
        with patch.object(tool._model.client, "_post") as post, lend(Budget(0.5, spent=0.5)):
            with self.assertRaises(BudgetExceeded):
                tool.submit("a boat")
        post.assert_not_called()


class TestTheRegistry(unittest.TestCase):

    def test_video_models_are_a_task_and_priced_by_the_second(self):
        listed = registry.models(task="text-to-video")
        self.assertIn("veo-3.1-fast-generate-preview", listed)
        self.assertIn("wan2.7-t2v", listed)
        self.assertNotIn("wan2.7-t2v", registry.models(task="image-to-video"))
        self.assertEqual(price_per_second("veo-3.1-generate-preview", "4k"), 0.6)
        self.assertEqual(price_per_second("grok-imagine-video-1.5", "1080p"), 0.08)

    def test_saved_without_the_key_and_refused_from_an_untrusted_file(self):
        tool = video("google", model="veo-3.1-lite-generate-preview", api_key="secret", output_dir="out")
        self.assertEqual(tool._serialise_init_args(),
                         {"model": "veo-3.1-lite-generate-preview", "timeout": 600.0, "output_dir": "out"})
        self.assertIn("trusted=True", Video._untrusted_load_refusal({"output_dir": "out"}))


class TestVideoLive(unittest.TestCase):
    """The cheapest clip on each provider: about $0.57 for the four."""

    def _live(self, service, model, key, options):
        if not os.getenv(key):
            self.skipTest(f"{key} not set")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = video(service, model=model).run("a paper boat on a puddle", options=options)
        self.assertGreater(len(result["videos"][0]["data"]), 10_000)
        self.assertGreater(result["cost"], 0)

    def test_xai(self):
        self._live("xai", "grok-imagine-video-1.5-lite", "XAI_API_KEY", {"duration": 1, "resolution": "480p"})

    def test_google(self):
        self._live("google", "veo-3.1-lite-generate-preview", "GOOGLE_AI_API_KEY", {"duration": 4})

    def test_minimax(self):
        self._live("minimax", "MiniMax-H3-Max", "MINIMAX_API_KEY", {"resolution": "480p", "duration": 5})

    def test_qwen(self):
        self._live("qwen", "wan2.2-t2v-plus", "DASHSCOPE_API_KEY", {"resolution": "480p"})


if __name__ == "__main__":
    unittest.main()
