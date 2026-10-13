"""
Google's speech tools are two HTTPS requests, not two SDKs.

`speech("google")` and `transcribe("google")` called google-cloud-texttospeech
and google-cloud-speech — grpc, protobuf and their own errors past `APIError`.
They post JSON now, as every other tool does; pinned here: where each goes,
what it carries, and how the answer is read.
"""

import base64
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from yait_aichain.tools import speech, transcribe

KEYS = {"GOOGLE_AI_API_KEY": "g-key"}


@patch.dict(os.environ, KEYS)
class TestGoogleSpeechWire(unittest.TestCase):

    def test_synthesis_posts_json_and_decodes_the_audio(self):
        tool = speech("google", output_dir=tempfile.mkdtemp())
        with patch("yait_aichain.tools.speech._tts.request_json",
                   return_value={"audioContent": base64.b64encode(b"ID3audio").decode()}) as req:
            path = tool.run("Hello.", {"voice": "en-US-Neural2-C", "speed": 1.2, "format": "wav"})
        (_, method, url), kw = req.call_args
        self.assertEqual((method, url), ("POST", "https://texttospeech.googleapis.com/v1/text:synthesize"))
        self.assertEqual(kw["headers"]["x-goog-api-key"], "g-key")
        self.assertEqual(kw["json"], {"input": {"text": "Hello."},
                                      "voice": {"languageCode": "en-US", "name": "en-US-Neural2-C"},
                                      "audioConfig": {"audioEncoding": "LINEAR16", "speakingRate": 1.2}})
        with open(path, "rb") as fh:
            self.assertEqual(fh.read(), b"ID3audio")
        self.assertTrue(path.endswith(".wav"))

    def test_recognition_posts_the_audio_and_reads_words(self):
        tool = transcribe("google")
        path = os.path.join(tempfile.mkdtemp(), "a.flac")
        with open(path, "wb") as fh:
            fh.write(b"fLaCdata")
        answer = {"results": [{"alternatives": [{"transcript": "hello there", "words": [
            {"word": "hello", "startTime": "0s", "endTime": "0.400s"},
            {"word": "there", "startTime": "0.400s", "endTime": "0.900s"}]}]}]}
        with patch("yait_aichain.tools.transcribe._stt.request_json", return_value=answer) as req, \
                patch.object(type(tool), "_probe_duration_seconds", staticmethod(lambda p: 2.0)):
            out = json.loads(tool.run(path, {"timestamps": True, "language": "en-GB"}))
        (_, method, url), kw = req.call_args
        self.assertEqual((method, url), ("POST", "https://speech.googleapis.com/v1/speech:recognize"))
        body = kw["json"]
        self.assertEqual((body["config"]["encoding"], body["config"]["languageCode"],
                          body["config"]["enableWordTimeOffsets"]), ("FLAC", "en-GB", True))
        self.assertEqual(base64.b64decode(body["audio"]["content"]), b"fLaCdata")
        self.assertEqual(out["text"], "hello there")
        self.assertEqual(out["words"][1], {"word": "there", "start": 0.4, "end": 0.9})

    def test_no_key_is_said_at_construction(self):
        with patch.dict(os.environ, {}, clear=True):
            for make in (lambda: speech("google"), lambda: transcribe("google")):
                with self.assertRaisesRegex(ValueError, "Google API key"):
                    make()


if __name__ == "__main__":
    unittest.main()
