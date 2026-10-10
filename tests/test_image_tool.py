"""
image(service, model=…) — an image model as a tool (3.9.0).

A model is what can be an agent's mind; an image model cannot, so it is
reached as a tool. Only the face changed: the request is still built by Skill
and Model from the provider data. What is pinned, offline: the factory and its
refusals; how a call becomes the Skill's request (images as parts, options as
the format); the Record and what a model reads of it — never the bytes; where
an image is saved; which notices reach the reader; and that the old way, a
Skill with an image model, warns while the tool's own Skill does not.

Measured live 2026-10-10: gemini-2.5-flash-image at aspect_ratio 16:9 returned
1344x768, and an edit of it kept the size; gpt-image-1-mini at 1024x1024, low
quality, returned 1024x1024 for $0.0022 from its reported tokens.
"""

import base64
import os
import sys
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Agent, Chain, Model, Skill
from yait_aichain.tools import Image, image

KEYS = {"OPENAI_API_KEY": "k", "GOOGLE_AI_API_KEY": "k", "XAI_API_KEY": "k", "BFL_API_KEY": "k",
        "DASHSCOPE_API_KEY": "k", "RECRAFT_API_KEY": "k", "REVE_API_KEY": "k"}
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode()


def _run_with(result, adaptations=(), usage=None):
    """Patch the Skill an Image builds, capturing what it was asked."""
    asked = {}

    def fake_run(self, *a, **k):
        asked["input"], asked["output"] = self._input, self._output
        self.last_adaptations = list(adaptations)
        self.last_usage = usage
        return result
    return patch.object(Skill, "run", fake_run), asked


@patch.dict(os.environ, KEYS)
class TestFactory(unittest.TestCase):

    def test_by_service_and_model(self):
        t = image("xai", model="grok-imagine-image")
        self.assertIsInstance(t, Image)
        self.assertEqual((t.service, t.model, t.name, t.risk), ("xai", "grok-imagine-image", "imageXAI", "draft"))
        self.assertEqual(image("openai", model="gpt-image-1.5").name, "imageOpenai")

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "grok-imagine-image is xai's, not openai's"):
            image("openai", model="grok-imagine-image")
        with self.assertRaisesRegex(ValueError, "unknown image service 'midjourney'"):
            image("midjourney", model="x")
        with self.assertRaises(TypeError):
            image("openai")                                   # the model is required

    def test_an_agent_takes_it(self):
        Agent(Model("gpt-5.4-mini", api_key="k"), tools=[image("google", model="gemini-2.5-flash-image")])


@patch.dict(os.environ, KEYS)
class TestACall(unittest.TestCase):

    def test_a_prompt_and_options_become_the_skills_request(self):
        fake, asked = _run_with({"base64": PNG, "mime_type": "image/png"})
        with fake:
            image("google", model="gemini-2.5-flash-image").run("a lighthouse", options={"aspect_ratio": "16:9"})
        self.assertEqual(asked["input"]["messages"][0]["parts"], [{"type": "text", "text": "a lighthouse"}])
        self.assertEqual(asked["output"]["format"], {"type": "image", "aspect_ratio": "16:9"})

    def test_images_make_it_an_edit(self):
        path = os.path.join(tempfile.mkdtemp(), "in.png")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(PNG))
        fake, asked = _run_with({"base64": PNG, "mime_type": "image/png"})
        with fake:
            image("google", model="gemini-2.5-flash-image").run("stormy sky", images=[path, "https://e.com/r.png"])
        parts = asked["input"]["messages"][0]["parts"]
        self.assertEqual(parts[0]["source"], {"kind": "file", "path": path})
        self.assertEqual(parts[1]["source"], {"kind": "url", "url": "https://e.com/r.png"})
        self.assertEqual(parts[2], {"type": "text", "text": "stormy sky"})

    def test_mistakes_are_said_before_a_request(self):
        t = image("google", model="gemini-2.5-flash-image")
        with self.assertRaisesRegex(ValueError, "unknown option"):
            t.run("x", options={"aspect": "16:9"})
        with self.assertRaisesRegex(ValueError, "needs a prompt"):
            t.run("  ")
        with self.assertRaises(FileNotFoundError):
            t.run("x", images=["/no/such/file.png"])


@patch.dict(os.environ, KEYS)
class TestTheResult(unittest.TestCase):

    def test_a_record_the_model_reads_without_the_bytes(self):
        usage = SimpleNamespace(cost=0.0022, input_tokens=11, output_tokens=272, total_tokens=283,
                                image_input_tokens=0)
        fake, _ = _run_with({"base64": PNG, "mime_type": "image/png", "revised_prompt": "a blue circle"}, usage=usage)
        with fake:
            out = image("openai", model="gpt-image-1-mini").run("a blue circle")
        self.assertEqual(out["images"][0]["base64"], PNG)
        self.assertEqual((out["cost"], out["usage"]["output_tokens"]), (0.0022, 272))
        text = str(out)
        self.assertIn("went back to the code", text)
        self.assertIn("The model drew from this prompt: a blue circle", text)
        self.assertNotIn(PNG[:20], text)

    def test_saved_where_the_code_says(self):
        d = tempfile.mkdtemp()
        fake, _ = _run_with({"base64": PNG, "mime_type": "image/png"})
        with fake:
            out = image("google", model="gemini-2.5-flash-image", output_dir=d).run("x")
        path = out["images"][0]["path"]
        self.assertTrue(path.startswith(d) and path.endswith(".png") and os.path.isfile(path))
        self.assertIn(f"saved to {path}", str(out))
        self.assertNotIn("output_dir", image("google", model="gemini-2.5-flash-image").parameters["properties"])

    def test_an_unknown_price_is_said(self):
        fake, _ = _run_with({"base64": PNG, "mime_type": "image/png"})
        with fake:
            out = image("bfl", model="flux-dev").run("x")
        self.assertIsNone(out["cost"])
        self.assertIn("not known", out["cost_basis"])

    def test_a_price_per_picture_when_no_usage_is_reported(self):
        # xAI reports no tokens for a picture; the registry's $0.02 stands in,
        # and the basis says it is the base size's rate, not a measurement.
        fake, _ = _run_with({"base64": PNG, "mime_type": "image/png"})
        with fake:
            out = image("xai", model="grok-imagine-image").run("x")
        self.assertEqual(out["cost"], 0.02)
        self.assertIn("price per image", out["cost_basis"])

    def test_only_notices_that_change_what_was_asked(self):
        a = [SimpleNamespace(kind="translated", option="aspect_ratio", why="this provider's word"),
             SimpleNamespace(kind="declined", option="seed", why="this provider has no seed")]
        fake, _ = _run_with({"base64": PNG, "mime_type": "image/png"}, adaptations=a)
        with fake:
            out = image("xai", model="grok-imagine-image").run("x", options={"seed": 1, "aspect_ratio": "1:1"})
        self.assertEqual(out["notices"], ["seed: declined — this provider has no seed"])


@patch.dict(os.environ, KEYS)
class TestTheOldWay(unittest.TestCase):

    def test_a_skill_with_an_image_model_warns(self):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            Skill(Model("grok-imagine-image"), prompt="x",
                  output={"modalities": ["image"], "format": {"type": "image"}})
        mine = [w for w in seen if "raises in 4.0" in str(w.message)]
        self.assertTrue(mine)
        # Shown by default — a DeprecationWarning from library code is not —
        # and pointing at the caller's line, not at the library's wrapper.
        self.assertTrue(issubclass(mine[0].category, FutureWarning))
        self.assertEqual(os.path.basename(mine[0].filename), "test_image_tool.py")

    def test_the_tools_own_skill_does_not(self):
        fake, _ = _run_with({"base64": PNG, "mime_type": "image/png"})
        with warnings.catch_warnings(record=True) as seen, fake:
            warnings.simplefilter("always")
            image("xai", model="grok-imagine-image").run("x")
        self.assertFalse([w for w in seen if "raises in 4.0" in str(w.message)])

    def test_a_saved_chain_keeps_the_model_and_the_directory(self):
        path = os.path.join(tempfile.mkdtemp(), "c.yaml")
        Chain(steps=[(image("xai", model="grok-imagine-image", output_dir="images/"), "picture")]).save(path)
        back = Chain.load(path)._steps[0][0]
        self.assertEqual((back.model, back.output_dir, back.service), ("grok-imagine-image", "images/", "xai"))


if __name__ == "__main__":
    unittest.main()
