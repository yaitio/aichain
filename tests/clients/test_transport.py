"""
Unified HTTP transport factory (1.3.4 #52): make_http honours an explicit proxy
and the HTTPS_PROXY / HTTP_PROXY env vars, and both model clients and tool
clients build their transport through it.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import urllib3
from clients._base import make_http


class TestMakeHttp(unittest.TestCase):

    def tearDown(self):
        for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            os.environ.pop(k, None)

    def test_no_proxy_is_plain_pool_manager(self):
        http = make_http()
        self.assertNotIsInstance(http, urllib3.ProxyManager)
        self.assertIsInstance(http, urllib3.PoolManager)

    def test_env_proxy_is_honoured(self):
        os.environ["HTTPS_PROXY"] = "http://corp-proxy:3128"
        self.assertIsInstance(make_http(), urllib3.ProxyManager)

    def test_explicit_proxy_overrides_and_adds_basic_auth(self):
        http = make_http({"url": "http://p:3128", "username": "u", "password": "p"})
        self.assertIsInstance(http, urllib3.ProxyManager)
        self.assertIn("Proxy-Authorization", http.proxy_headers)

    def test_model_client_uses_factory(self):
        os.environ["HTTPS_PROXY"] = "http://corp-proxy:3128"
        from models import Model
        m = Model("gpt-4o", api_key="k")
        self.assertIsInstance(m.client._http, urllib3.ProxyManager)

    def test_tool_client_uses_factory(self):
        os.environ["HTTPS_PROXY"] = "http://corp-proxy:3128"
        from tools import PerplexitySearchTool
        tool = PerplexitySearchTool(api_key="k")
        self.assertIsInstance(tool._http, urllib3.ProxyManager)


if __name__ == "__main__":
    unittest.main()
