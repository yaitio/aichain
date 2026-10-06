"""
convertToMD bounds the wait for a URL.

MarkItDown calls ``requests`` with no timeout, so a server that holds the
connection open and never answers hangs a conversion for as long as it likes —
measured 2026-10-06 on www.mckinsey.com, for hours. The tool hands MarkItDown a
session that applies a timeout to every request; what is pinned here, offline:

  * the default reaches the request, and an explicit one replaces it;
  * a caller's own per-request timeout is not overridden;
  * ``timeout=None`` leaves MarkItDown's own session in place;
  * the ``Accept`` header MarkItDown sets on its own session survives ours;
  * a timeout becomes a failed call, not a hang.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import markitdown  # noqa: F401
    import requests
except ImportError:  # pragma: no cover - the convert extra is optional
    markitdown = None

from yait_aichain.tools.convert.to_md import DEFAULT_TIMEOUT, convertToMD


@unittest.skipIf(markitdown is None, "markitdown is not installed")
class TestTheTimeoutReachesTheRequest(unittest.TestCase):

    def _timeout_sent(self, tool, **request_kwargs):
        session = tool._get_converter()._requests_session
        with patch.object(requests.Session, "request", return_value="ok") as sent:
            session.request("GET", "https://example.com/", **request_kwargs)
        return sent.call_args.kwargs.get("timeout")

    def test_the_default_is_applied(self):
        self.assertEqual(self._timeout_sent(convertToMD()), DEFAULT_TIMEOUT)

    def test_an_explicit_timeout_replaces_it(self):
        self.assertEqual(self._timeout_sent(convertToMD(timeout=5)), 5)

    def test_a_per_request_timeout_is_not_overridden(self):
        self.assertEqual(self._timeout_sent(convertToMD(), timeout=1), 1)

    def test_none_leaves_markitdowns_own_session(self):
        session = convertToMD(timeout=None)._get_converter()._requests_session
        self.assertIs(type(session), requests.Session)

    def test_markitdowns_accept_header_survives(self):
        session = convertToMD()._get_converter()._requests_session
        self.assertIn("text/markdown", session.headers["Accept"])

    def test_a_timeout_is_a_failed_call_not_a_hang(self):
        tool = convertToMD()
        with patch("yait_aichain.tools.convert.to_md.assert_safe_url"), \
             patch.object(requests.Session, "request",
                          side_effect=requests.exceptions.ReadTimeout("read timed out")):
            result = tool(input="https://example.com/article")
        self.assertFalse(result.success)
        self.assertIn("timed out", result.error)


if __name__ == "__main__":
    unittest.main()
