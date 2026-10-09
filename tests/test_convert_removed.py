"""
3.8.0 removed the convert tools — convertToMD, convertToHTML, convertToPDF.

They were one project's case, brought heavy and undeclared dependencies
(MarkItDown's onnxruntime and numpy, 114 MB; mistletoe and WeasyPrint, which no
extra installed, WeasyPrint with system libraries pip cannot install), and their
example lives in the cookbooks now. What is pinned: every way of reaching a
removed name says what replaced it, rather than ``ModuleNotFoundError``; speech
and transcription, which lived under ``tools.convert`` until 3.5.0, still import
from there; and nothing in the library imports the removed code.
"""

import importlib
import os
import pathlib
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Chain


class TestARemovedNameSaysWhatReplacedIt(unittest.TestCase):

    def test_from_the_tools_package(self):
        for name in ("convertToMD", "convertToHTML", "convertToPDF", "MarkItDownTool", "WeasyprintTool"):
            with self.subTest(name), self.assertRaisesRegex(ImportError, f"{name} was removed .* in 3.8.0"):
                exec(f"from yait_aichain.tools import {name}")

    def test_the_url_case_points_at_fetch(self):
        with self.assertRaisesRegex(ImportError, r'fetch\("local"\)'):
            from yait_aichain.tools import convertToMD  # noqa: F401

    def test_from_the_old_modules(self):
        for path in ("yait_aichain.tools.convert.to_md", "yait_aichain.tools.convert.to_html",
                     "yait_aichain.tools.convert.to_pdf"):
            with self.subTest(path), self.assertRaisesRegex(ImportError, "was removed .* in 3.8.0"):
                importlib.import_module(path)
        with self.assertRaisesRegex(ImportError, "convertToHTML was removed"):
            from yait_aichain.tools.convert import convertToHTML  # noqa: F401

    def test_a_chain_saved_with_one_says_so_on_load(self):
        path = os.path.join(tempfile.mkdtemp(), "old.yaml")
        with open(path, "w") as fh:
            fh.write("steps:\n- kind: tool\n  output_key: page\n  tool:\n"
                     "    class: yait_aichain.tools.convert.to_md.convertToMD\n    init_args: {}\n")
        with self.assertRaisesRegex(ImportError, "convertToMD was removed"):
            Chain.load(path)


class TestSpeechStillImportsFromConvert(unittest.TestCase):

    def test_the_old_paths(self):
        from yait_aichain.tools import TTS, STT
        from yait_aichain.tools.convert import TTS as old_tts, STT as old_stt
        from yait_aichain.tools.convert.to_speech import ttsOpenAI
        self.assertIs(old_tts, TTS)
        self.assertIs(old_stt, STT)
        self.assertEqual(ttsOpenAI.__module__, "yait_aichain.tools.speech._tts")


class TestNothingImportsTheRemovedCode(unittest.TestCase):

    def test_no_module_left_and_no_reference(self):
        root = pathlib.Path(__file__).resolve().parent.parent / "yait_aichain"
        self.assertFalse((root / "tools" / "convert").exists())
        allowed = {"_convert_removed.py", "_moved.py", "__init__.py"}      # they name it to refuse it
        # A use is an import or a call; a docstring saying where code came from is not.
        use = re.compile(r"import[^\n]*\bconvertTo(MD|HTML|PDF)\b|\bconvertTo(MD|HTML|PDF)\s*\(|tools\.convert\.to_(md|html|pdf)")
        for f in root.rglob("*.py"):
            if f.name in allowed and f.parent.name in ("tools", "yait_aichain"):
                continue
            hit = use.search(f.read_text(encoding="utf-8"))
            self.assertIsNone(hit, f"{f.relative_to(root)} still uses the removed code: {hit and hit.group(0)}")


if __name__ == "__main__":
    unittest.main()
