"""
An edit given more images than it takes sends the newest and says so.

Recraft took the first image, xAI the first three, BFL the first four — the
oldest in the conversation — and dropped the rest in silence.
"""

import base64
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.models import Model
from yait_aichain.models._adaptation import DECLINED

OUT = {"modalities": ["image"], "format": {"type": "image"}}


def _images(n):
    parts = [{"type": "image", "source": {"kind": "base64", "mime": "image/png",
                                          "data": base64.b64encode(f"img{i}".encode()).decode()}}
             for i in range(n)]
    return [{"role": "user", "parts": [{"type": "text", "text": "edit"}, *parts]}]


class TestLimits(unittest.TestCase):

    def _build(self, name, n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Model(name, api_key="k").build(_images(n), OUT)

    def test_bfl_sends_the_newest_four_and_declines_the_rest(self):
        _, body, made = self._build("flux-kontext-pro", 6)
        sent = [body.get("input_image")] + [body.get(f"input_image_{i}") for i in range(2, 5)]
        self.assertEqual([base64.b64decode(s) for s in sent], [b"img2", b"img3", b"img4", b"img5"])
        self.assertNotIn("input_image_5", body)
        self.assertIn(("images", DECLINED), [(a.option, a.kind) for a in made])

    def test_within_the_limit_nothing_is_said(self):
        _, _, made = self._build("flux-kontext-pro", 2)
        self.assertNotIn("images", [a.option for a in made])


if __name__ == "__main__":
    unittest.main()
