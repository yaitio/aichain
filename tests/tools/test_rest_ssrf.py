"""
RestApiTool must block SSRF targets (private / loopback / metadata IPs), not
just non-http schemes — unless AICHAIN_ALLOW_PRIVATE_URLS is set (for genuine
internal APIs). The guard runs before any network call.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.rest_api import RestApiTool


def _tool(url):
    return RestApiTool(name="probe", description="probe", method="GET", url=url)


class TestRestApiSSRF(unittest.TestCase):

    def tearDown(self):
        os.environ.pop("AICHAIN_ALLOW_PRIVATE_URLS", None)

    def test_metadata_ip_blocked(self):
        tool = _tool("http://169.254.169.254/latest/meta-data/")
        tool._http.request = MagicMock()           # must never be reached
        with self.assertRaises(ValueError):
            tool.run()
        tool._http.request.assert_not_called()

    def test_loopback_blocked(self):
        tool = _tool("http://127.0.0.1:8080/admin")
        tool._http.request = MagicMock()
        with self.assertRaises(ValueError):
            tool.run()
        tool._http.request.assert_not_called()

    def test_opt_out_allows_private(self):
        os.environ["AICHAIN_ALLOW_PRIVATE_URLS"] = "1"
        tool = _tool("http://127.0.0.1:8080/admin")
        resp = MagicMock(status=200, data=b"{}")
        tool._http.request = MagicMock(return_value=resp)
        tool.run()                                  # no SSRF error → request made
        tool._http.request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
