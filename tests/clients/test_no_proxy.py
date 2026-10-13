"""
The proxy is chosen per request, as the environment says.

`make_http` picked one proxy for the whole manager: `NO_PROXY` was ignored,
so a `private/` model on localhost, a Chroma in the VPC and an internal REST
went through the corporate proxy, got 502 and started a fallback; and
`HTTPS_PROXY` was applied to `http://` too, where the proxy reads the key.
"""

import os
import unittest
from unittest.mock import patch

from yait_aichain.clients._base import make_http


class TestRouting(unittest.TestCase):

    def test_no_proxy_and_the_scheme_decide(self):
        env = {"HTTPS_PROXY": "http://corp:3128", "NO_PROXY": "localhost,.internal"}
        with patch.dict(os.environ, env, clear=True):
            http = make_http()
            direct = {u: http._for(u) is http._direct for u in (
                "https://api.openai.com/v1", "http://localhost:8080/v1",
                "https://chroma.internal/", "http://plain.example/")}
        self.assertEqual(direct, {"https://api.openai.com/v1": False, "http://localhost:8080/v1": True,
                                  "https://chroma.internal/": True, "http://plain.example/": True})

    def test_without_a_proxy_it_is_a_plain_pool(self):
        import urllib3
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsInstance(make_http(), urllib3.PoolManager)


if __name__ == "__main__":
    unittest.main()
