"""
Where speech writes is the code's choice, and every call gets its own file.

`output_path` was an option in the schema a model sees, and with no
`AICHAIN_OUTPUT_ROOT` nothing confined it: an injected instruction replaced
`~/.zshrc` with an MP3. The automatic name was a timestamp to the second, so
parallel calls in a Pool wrote one file and all reported it; the default
`./audio` is not writable on Lambda; and the path was checked after the paid
call. The folder is a setting of the tool now, files are created new and
exclusively before the request, and the tool is `draft`.
"""

import os
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from yait_aichain.tools import speech

KEYS = {"OPENAI_API_KEY": "k"}


def _tool(**kw):
    with patch.dict(os.environ, KEYS):
        return speech("openai", **kw)


def _answer(data=b"ID3audio"):
    return patch("yait_aichain.tools.speech._tts.request", return_value=MagicMock(data=data))


class TestTheModelDoesNotChooseThePath(unittest.TestCase):

    def test_output_path_is_not_in_the_schema_and_the_tool_is_draft(self):
        tool = _tool()
        self.assertNotIn("output_path", tool.parameters["properties"]["options"]["properties"])
        self.assertEqual(tool.risk, "draft")

    def test_the_folder_is_the_tools_setting(self):
        folder = tempfile.mkdtemp()
        with _answer():
            path = _tool(output_dir=folder).run("hello")
        self.assertEqual(os.path.dirname(path), os.path.realpath(folder))
        self.assertEqual(open(path, "rb").read(), b"ID3audio")

    def test_by_default_it_writes_where_lambda_can(self):
        with _answer():
            path = _tool().run("hello")
        self.assertTrue(path.startswith(os.path.realpath(tempfile.gettempdir())))
        os.unlink(path)

    def test_output_path_from_code_still_works_and_says_it_is_going(self):
        target = os.path.join(tempfile.mkdtemp(), "a.mp3")
        with _answer(), self.assertWarnsRegex(DeprecationWarning, "output_dir"):
            self.assertEqual(_tool().run("hello", {"output_path": target}), os.path.realpath(target))


class TestEveryCallItsOwnFile(unittest.TestCase):

    def test_parallel_calls_never_share_a_file(self):
        tool, paths = _tool(output_dir=tempfile.mkdtemp()), []
        with _answer():
            threads = [threading.Thread(target=lambda: paths.append(tool.run("x"))) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(len(set(paths)), 8)

    def test_a_call_that_failed_leaves_no_file(self):
        folder = tempfile.mkdtemp()
        with patch("yait_aichain.tools.speech._tts.request", side_effect=RuntimeError("402")):
            with self.assertRaises(RuntimeError):
                _tool(output_dir=folder).run("x")
        self.assertEqual(os.listdir(folder), [])

    def test_a_refused_folder_costs_nothing(self):
        with patch.dict(os.environ, {"AICHAIN_OUTPUT_ROOT": tempfile.mkdtemp()}), \
                patch("yait_aichain.tools.speech._tts.request") as req:
            with self.assertRaises(PermissionError):
                _tool(output_dir="/etc").run("x")
        req.assert_not_called()


if __name__ == "__main__":
    unittest.main()
