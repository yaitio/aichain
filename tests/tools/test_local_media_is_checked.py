"""
A local file a model or a variable names is read only if it is what it claims.

`images=["/proc/self/environ"]` uploaded the environment — the AWS keys on
Lambda — to an image provider, `transcribe("~/.aws/credentials")` sent the
credentials file, and `/dev/zero` in a Skill's media variable exhausted the
memory. A string passed as `images` was split into characters.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from yait_aichain.tools._security import check_input_file


def _file(name, content):
    path = os.path.join(tempfile.mkdtemp(), name)
    with open(path, "wb") as fh:
        fh.write(content)
    return path


class TestCheckInputFile(unittest.TestCase):

    def test_a_real_image_and_a_real_audio_pass(self):
        check_input_file(_file("a.png", b"\x89PNG\r\n\x1a\n" + b"0" * 32), "image")
        check_input_file(_file("a.mp3", b"ID3\x03" + b"0" * 32), "audio")

    def test_a_secret_named_as_media_is_refused(self):
        secret = _file("environ", b"AWS_SECRET_ACCESS_KEY=abc\n")
        for kind in ("image", "audio"):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "does not look like"):
                    check_input_file(secret, kind)

    def test_a_device_is_not_a_regular_file(self):
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            check_input_file("/dev/zero", "image")

    def test_the_input_root_confines(self):
        inside = _file("a.png", b"\x89PNG\r\n\x1a\n")
        with patch.dict(os.environ, {"AICHAIN_INPUT_ROOT": tempfile.mkdtemp()}):
            with self.assertRaises(PermissionError):
                check_input_file(inside, "image")

    def test_too_large_is_refused(self):
        with patch.dict("yait_aichain.tools._security.MAX_INPUT_BYTES", {"image": 8}):
            with self.assertRaisesRegex(ValueError, "at most"):
                check_input_file(_file("a.png", b"\x89PNG\r\n\x1a\n" + b"0" * 32), "image")


class TestTheToolsUseIt(unittest.TestCase):

    def test_transcription_refuses_a_secret(self):
        from yait_aichain.tools import transcribe
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            with self.assertRaisesRegex(ValueError, "does not look like audio"):
                transcribe("openai").run(_file("credentials", b"[default]\naws_secret=1\n"))

    def test_a_skill_refuses_a_secret_as_an_image(self):
        from yait_aichain.skills._adapters import resolve_media
        path = _file("environ", b"KEY=1\n")
        with self.assertRaises(ValueError):
            resolve_media([{"role": "user", "parts": [{"type": "image",
                                                       "source": {"kind": "file", "path": path}}]}])


if __name__ == "__main__":
    unittest.main()
