"""
A speech tool sends the options that are set, in a format it writes.

``voice=None`` went into the request and ``speed=None`` raised; an unknown
format became MP3 on Google; ttsQwen's default model did not exist on its
endpoint and its answer is WAV whatever was asked; an invalid region passed
the constructor.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from yait_aichain.tools import speech

KEYS = {"OPENAI_API_KEY": "k", "DASHSCOPE_API_KEY": "k"}


@patch.dict(os.environ, KEYS)
class TestSpeechOptions(unittest.TestCase):

    def test_unset_options_are_not_sent(self):
        tool = speech("openai")
        with patch("yait_aichain.tools.speech._tts.request", return_value=MagicMock(data=b"ID3")) as req, \
                patch.object(type(tool), "_save", lambda self, opts, fmt, produce: produce()):
            tool.run("hi", {"voice": None, "speed": None})
        body = req.call_args.kwargs["json"]
        self.assertEqual((body["voice"], body["speed"]), ("alloy", 1.0))

    def test_a_format_the_provider_does_not_write_is_refused(self):
        with self.assertRaisesRegex(ValueError, "writes"):
            speech("openai").run("hi", {"format": "ogg"})
        with self.assertRaisesRegex(ValueError, "wav"):
            speech("qwen").run("hi", {"format": "mp3"})

    def test_qwen_defaults_to_a_model_that_exists(self):
        self.assertEqual(speech("qwen").model, "qwen3-tts-flash")

    def test_an_unknown_region_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            speech("qwen", region="mars")


if __name__ == "__main__":
    unittest.main()
