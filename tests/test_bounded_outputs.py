"""
Nothing a tool returns, or reads, is unbounded.

A 149 KB answer unpacked by MarkItDown's zip converter took 950 MB; a 2 GB
log was read whole for forty lines; `local_run` and REST buffered everything,
and nothing between a tool and the model cut anything — the model got a
context overflow and the Lambda an OOM.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from yait_aichain.tools._security import clip


class TestClip(unittest.TestCase):

    def test_head_and_tail_with_a_marker_and_idempotent(self):
        text = "a" * 500 + "TAIL"
        once = clip(text, 100)
        self.assertLessEqual(len(once), 100)
        self.assertTrue(once.endswith("TAIL"))
        self.assertIn("left out", once)
        self.assertEqual(clip(once, 100), once)


class TestShell(unittest.TestCase):

    def test_read_streams_and_stops_at_the_budget(self):
        from yait_aichain.utilities.shell import _read
        root = tempfile.mkdtemp()
        with open(os.path.join(root, "big.log"), "w") as fh:
            fh.write(("x" * 99 + "\n") * 5_000)
        tool = _read.LocalReadTool(root)
        with patch.object(_read, "MAX_READ_BYTES", 1_000):
            out = tool.run("big.log")
        self.assertIn("of 5000", out)
        self.assertIn("stopped at 1,000", out)
        self.assertLess(len(out), 2_000)

    def test_run_clips_a_flood(self):
        from yait_aichain.utilities.shell import _run
        tool = _run.LocalRunTool(tempfile.mkdtemp())
        out = tool.run(f"{sys.executable} -c \"print('y' * 200000); print('END')\"")
        self.assertLessEqual(len(out["stdout"]), _run.MAX_OUTPUT_CHARS)
        self.assertIn("END", out["stdout"])


class TestRest(unittest.TestCase):

    def test_an_oversized_response_is_refused(self):
        from yait_aichain.utilities.rest import _tool
        t = _tool.RestApiTool("get", "Get.", "GET", "https://api.example.com/x")
        t._http = MagicMock()
        t._http.request.return_value = MagicMock(status=200, headers={},
                                                  read=MagicMock(return_value=b"0" * 11))
        with patch.object(_tool, "MAX_RESPONSE_BYTES", 10), \
                patch("yait_aichain.tools._security.assert_safe_url", lambda u, **k: None):
            with self.assertRaisesRegex(ValueError, "too large"):
                t.run()


class TestFetchLocal(unittest.TestCase):

    def _tool(self, ctype, chunks):
        from yait_aichain.tools import fetchLocal
        tool = fetchLocal(respect_robots=False)
        r = MagicMock(status=200, headers={"content-type": ctype})
        r.stream.return_value = iter(chunks)
        tool._http = MagicMock()
        tool._http.request.return_value = r
        return tool

    def test_an_archive_is_not_converted(self):
        with patch("yait_aichain.tools._security.assert_safe_url", lambda u, **k: None), \
                patch("yait_aichain.tools.fetch.local.assert_safe_url", lambda u: None):
            page = self._tool("application/zip", [b"PK"]).run("https://e.com/a.zip")["pages"][0]
        self.assertIn("not a page", page["error"])

    def test_a_body_over_the_budget_is_not_read_on(self):
        from yait_aichain.tools.fetch import local
        with patch.object(local, "MAX_PAGE_BYTES", 10), \
                patch("yait_aichain.tools._security.assert_safe_url", lambda u, **k: None), \
                patch("yait_aichain.tools.fetch.local.assert_safe_url", lambda u: None):
            page = self._tool("text/html", [b"x" * 8, b"x" * 8, b"never"]).run("https://e.com")["pages"][0]
        self.assertIn("over 10 bytes", page["error"])


if __name__ == "__main__":
    unittest.main()
