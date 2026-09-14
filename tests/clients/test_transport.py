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
from yait_aichain.clients._base import make_http


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

    def test_credentials_in_the_env_url_become_proxy_authorization(self):
        # The HTTPS_PROXY convention carries credentials in the URL. urllib3
        # ignores them, and every such proxy answered 407.
        import base64
        os.environ["HTTPS_PROXY"] = "http://us%40er:p%3Ass@corp-proxy:3128"
        http = make_http()
        self.assertIsInstance(http, urllib3.ProxyManager)
        expected = "Basic " + base64.b64encode(b"us@er:p:ss").decode()
        self.assertEqual(http.proxy_headers.get("Proxy-Authorization"), expected)
        self.assertEqual(http.proxy.auth, None)
        self.assertEqual((http.proxy.host, http.proxy.port), ("corp-proxy", 3128))

    def test_explicit_credentials_win_over_the_url(self):
        import base64
        http = make_http({"url": "http://a:b@p:3128", "username": "u", "password": "p"})
        self.assertEqual(http.proxy_headers["Proxy-Authorization"],
                         "Basic " + base64.b64encode(b"u:p").decode())

    def test_model_client_uses_factory(self):
        os.environ["HTTPS_PROXY"] = "http://corp-proxy:3128"
        from yait_aichain.models import Model
        m = Model("gpt-4o", api_key="k")
        self.assertIsInstance(m.client._http, urllib3.ProxyManager)

    def test_tool_client_uses_factory(self):
        os.environ["HTTPS_PROXY"] = "http://corp-proxy:3128"
        from yait_aichain.tools import PerplexitySearchTool
        tool = PerplexitySearchTool(api_key="k")
        self.assertIsInstance(tool._http, urllib3.ProxyManager)


if __name__ == "__main__":
    unittest.main()
