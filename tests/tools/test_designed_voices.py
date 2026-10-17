"""
Voices made from a description — ElevenLabs and MiniMax: design() returns candidates
with a sample each, keep() makes one permanent where the provider asks for that, run()
speaks with a voice id. What is pinned is the wire each provider documents.
"""

import base64
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.tools import speech

KEYS = {"ELEVENLABS_API_KEY": "el", "MINIMAX_API_KEY": "mm"}
TTS = "yait_aichain.tools.speech._tts"


def _tool(service, **kw):
    with patch.dict(os.environ, KEYS):
        return speech(service, output_dir=tempfile.mkdtemp(), **kw)


class TestElevenLabs(unittest.TestCase):

    def test_design_returns_three_candidates_with_their_samples(self):
        previews = {"previews": [{"generated_voice_id": f"g{i}", "audio_base_64": base64.b64encode(b"ID3" * 9).decode(),
                                  "duration_secs": 4.2} for i in range(3)]}
        tool = _tool("elevenlabs")
        with patch(f"{TTS}.request_json", return_value=previews) as req:
            made = tool.design("a young bright voice", "x" * 120, seed=7)
        url, body = req.call_args.args[2], req.call_args.kwargs["json"]
        self.assertIn("/v1/text-to-voice/design", url)
        self.assertEqual((body["seed"], body["model_id"], len(body["text"])), (7, "eleven_multilingual_ttv_v2", 120))
        self.assertEqual(req.call_args.kwargs["headers"], {"xi-api-key": "el"})
        self.assertEqual([c["voice_id"] for c in made], ["g0", "g1", "g2"])
        self.assertTrue(all(open(c["path"], "rb").read() == b"ID3" * 9 for c in made))

    def test_a_design_text_outside_its_range_is_refused_and_none_is_written_for_it(self):
        tool = _tool("elevenlabs")
        with self.assertRaisesRegex(ValueError, "100–1000"):
            tool.design("a voice", "too short")
        with patch(f"{TTS}.request_json", return_value={"previews": []}) as req:
            tool.design("a voice")
        self.assertTrue(req.call_args.kwargs["json"]["auto_generate_text"])

    def test_keep_turns_a_preview_into_a_voice_of_the_account(self):
        tool = _tool("elevenlabs")
        with patch(f"{TTS}.request_json", return_value={"voice_id": "v42"}) as req:
            self.assertEqual(tool.keep("g1", "Linara", "a bright voice"), "v42")
        self.assertEqual(req.call_args.kwargs["json"],
                         {"voice_name": "Linara", "voice_description": "a bright voice", "generated_voice_id": "g1"})

    def test_run_speaks_with_a_voice_id_on_the_model_of_the_tool(self):
        tool = _tool("elevenlabs", model="eleven_v4")
        with patch(f"{TTS}.request", return_value=MagicMock(data=b"ID3")) as req:
            path = tool.run("[excited] Hello!", {"voice": "v42", "speed": 1.12})
        self.assertIn("/v1/text-to-speech/v42", req.call_args.args[2])
        self.assertEqual(req.call_args.kwargs["json"],
                         {"text": "[excited] Hello!", "model_id": "eleven_v4", "voice_settings": {"speed": 1.12}})
        self.assertTrue(path.endswith(".mp3"))

    def test_no_voice_and_a_speed_out_of_range_are_said(self):
        tool = _tool("elevenlabs")
        with self.assertRaisesRegex(ValueError, "voice id"):
            tool.run("hi")
        with self.assertRaisesRegex(ValueError, "0.7–1.2"):
            tool.run("hi", {"voice": "v", "speed": 1.5})


class TestMiniMax(unittest.TestCase):

    def test_design_is_one_voice_with_its_sample(self):
        tool = _tool("minimax")
        with patch(f"{TTS}.request_json", return_value={"voice_id": "ttv-1", "trial_audio": b"ID3".hex(),
                                                        "base_resp": {"status_code": 0}}) as req:
            (made,) = tool.design("a calm baritone", "A sample line.")
        self.assertIn("/v1/voice_design", req.call_args.args[2])
        self.assertEqual(req.call_args.kwargs["json"], {"prompt": "a calm baritone", "preview_text": "A sample line."})
        self.assertEqual(made["voice_id"], "ttv-1")
        self.assertEqual(open(made["path"], "rb").read(), b"ID3")
        self.assertEqual(tool.keep("ttv-1", "x", "y"), "ttv-1")        # the voice is the account's already

    def test_run_decodes_hex_audio_and_a_refusal_says_why(self):
        tool = _tool("minimax")
        ok = {"data": {"audio": b"ID3".hex()}, "base_resp": {"status_code": 0}}
        with patch(f"{TTS}.request_json", return_value=ok) as req:
            path = tool.run("Hello", {"voice": "ttv-1", "format": "wav"})
        body = req.call_args.kwargs["json"]
        self.assertEqual((body["model"], body["voice_setting"]["voice_id"], body["audio_setting"]["format"]),
                         ("speech-2.8-hd", "ttv-1", "wav"))
        self.assertEqual(open(path, "rb").read(), b"ID3")
        refused = {"base_resp": {"status_code": 1008, "status_msg": "insufficient balance"}}
        with patch(f"{TTS}.request_json", return_value=refused), self.assertRaisesRegex(RuntimeError, "insufficient"):
            tool.run("Hello", {"voice": "ttv-1"})


class TestDesignedVoicesLive(unittest.TestCase):
    """One MiniMax design (about $0.005) and one ElevenLabs line with a stock voice."""

    @unittest.skipUnless(os.getenv("MINIMAX_API_KEY"), "MINIMAX_API_KEY not set")
    def test_minimax_design(self):
        (made,) = speech("minimax", output_dir=tempfile.mkdtemp()).design(
            "A calm adult voice, even and clear.", "This is a short sample of a designed voice.")
        self.assertGreater(os.path.getsize(made["path"]), 5_000)


if __name__ == "__main__":
    unittest.main()
