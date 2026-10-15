"""
What the OpenAI-shaped transcription tools send.

They called the OpenAI SDK for one multipart POST; they make that request
themselves now. Pinned per provider — where it goes and which fields it
carries — because the first version of the change sent sttOpenAI to
``None/audio/transcriptions``: the class still said ``_BASE_URL = None``,
which the SDK had read as "the official endpoint", and no offline test looked.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from yait_aichain.tools.transcribe._stt import sttOpenAI

KEYS = {"OPENAI_API_KEY": "k", "XAI_API_KEY": "k", "DASHSCOPE_API_KEY": "k"}


def _audio():
    path = os.path.join(tempfile.mkdtemp(), "a.mp3")
    with open(path, "wb") as fh:
        fh.write(b"\xff\xf3audio")
    return path


def _sent(cls, options=None, reply=b"hello"):
    tool = cls()
    with patch("yait_aichain.tools.transcribe._stt.request",
               return_value=MagicMock(data=reply)) as req:
        out = tool.run(_audio(), options)
    (_, method, url), kwargs = req.call_args
    return method, url, dict((k, v) for k, v in kwargs["fields"]), kwargs["fields"], out


@patch.dict(os.environ, KEYS)
class TestTheWire(unittest.TestCase):

    def test_it_goes_to_its_endpoint(self):
        method, url, _, _, _ = _sent(sttOpenAI)
        self.assertEqual((method, url), ("POST", "https://api.openai.com/v1/audio/transcriptions"))

    def test_gpt_4o_transcribe_asks_for_json_not_verbose_json(self):
        # verbose_json is whisper-1's alone: gpt-4o-transcribe answered 400,
        # checked live 2026-10-13.
        def make():
            return sttOpenAI(model="gpt-4o-transcribe")
        reply = json.dumps({"text": "hi"}).encode()
        _, _, fields, raw, out = _sent(make, {"format": "json"}, reply)
        self.assertEqual(fields["response_format"], "json")
        self.assertNotIn(("timestamp_granularities[]", "word"), raw)
        self.assertEqual(json.loads(out)["text"], "hi")
        with self.assertRaisesRegex(ValueError, "whisper-1"):
            _sent(make, {"timestamps": True})

    def test_a_file_over_the_limit_is_refused_and_told_how_to_cut_it(self):
        tool = sttOpenAI()
        tool._MAX_FILE_MB = 0.0
        with self.assertRaisesRegex(ValueError, "the provider's limit.*Long audio"):
            tool.run(_audio())

    def test_text_asks_for_text(self):
        _, _, fields, _, out = _sent(sttOpenAI, {"language": "en"})
        self.assertEqual((fields["model"], fields["response_format"], fields["language"]),
                         ("whisper-1", "text", "en"))
        self.assertEqual(fields["file"][0], "a.mp3")
        self.assertNotIn("prompt", fields)
        self.assertEqual(out, "hello")

    def test_json_asks_for_words_and_reads_them(self):
        reply = json.dumps({"text": "hi", "words": [{"word": "hi", "start": 0.0, "end": 0.2}]}).encode()
        _, _, fields, raw, out = _sent(sttOpenAI, {"format": "json"}, reply)
        self.assertEqual(fields["response_format"], "verbose_json")
        self.assertIn(("timestamp_granularities[]", "word"), raw)
        self.assertEqual(json.loads(out), {"text": "hi", "words": [{"word": "hi", "start": 0.0, "end": 0.2}],
                                           "segments": []})


if __name__ == "__main__":
    unittest.main()
