"""
A part a wire cannot carry is left out and said; one it can is sent right.

Video, URL audio and PDFs were dropped in silence and the model answered about
an attachment it never saw; a tool's follow-up held `null` (a 400); mp3 went as
`format: "mpeg"`; Claude got audio as a `document` (a 400).
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.models import Model
from yait_aichain.models._adaptation import DECLINED

OUT = {"format": {"type": "text"}}


def _msg(*parts):
    return [{"role": "user", "parts": [{"type": "text", "text": "look"}, *parts]}]


def _part(kind, mime, data="AAAA", source="base64"):
    return {"type": kind, "source": {"kind": source, "mime": mime, "data": data}}


class TestMedia(unittest.TestCase):

    def _build(self, name, parts):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Model(name, api_key="k").build(_msg(*parts), OUT)

    def test_mp3_goes_as_mp3(self):
        _, body, _ = self._build("gpt-4o", [_part("audio", "audio/mpeg")])
        self.assertEqual(body["messages"][-1]["content"][1]["input_audio"]["format"], "mp3")

    def test_a_pdf_goes_as_a_file_on_chat(self):
        _, body, _ = self._build("gpt-4o", [_part("document", "application/pdf")])
        self.assertEqual(body["messages"][-1]["content"][1]["type"], "file")

    def test_video_is_left_out_and_said(self):
        _, body, made = self._build("gpt-4o", [_part("video", "video/mp4")])
        self.assertEqual(body["messages"][-1]["content"], "look")
        self.assertIn(("video input", DECLINED), [(a.option, a.kind) for a in made])

    def test_claude_gets_no_audio_and_a_pdf_as_a_document(self):
        _, body, made = self._build("claude-sonnet-4-5", [_part("audio", "audio/wav"),
                                                         _part("document", "application/pdf")])
        types = [b["type"] for b in body["messages"][-1]["content"]]
        self.assertEqual(types, ["text", "document"])
        self.assertIn("audio input", [a.option for a in made])


if __name__ == "__main__":
    unittest.main()
