"""
Integration tests for clients._xai — XAIClient.

Two test classes:

  InvalidAuth — always runs; makes a real HTTP request with a fake key
                and verifies the API rejects it (check_auth returns False,
                list_models raises APIError).

  Live        — skipped unless XAI_API_KEY is set; verifies that
                check_auth returns True and list_models returns a
                well-formed list containing known models.

Run live tests:
    XAI_API_KEY=xai-... python3 -m unittest discover -s tests/clients -t . -v
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

from clients._base import APIError
from clients._xai import XAIClient

_KEY  = os.getenv("XAI_API_KEY")
_FAKE = "xai-invalid-key-for-testing-000000000000000000000000000000000000"


# ---------------------------------------------------------------------------
# Unit tests — no network, HTTP layer mocked
# ---------------------------------------------------------------------------

class TestXAIUnit(unittest.TestCase):
    """Auth headers, base URL, and list_models response parsing — all mocked."""

    @patch("urllib3.PoolManager")
    def test_auth_header_uses_bearer_scheme(self, _):
        client = XAIClient("xai-testkey")
        self.assertEqual(client._auth_headers()["Authorization"], "Bearer xai-testkey")

    @patch("urllib3.PoolManager")
    def test_auth_header_includes_content_type(self, _):
        client = XAIClient("xai-testkey")
        self.assertEqual(client._auth_headers()["Content-Type"], "application/json")

    @patch("urllib3.PoolManager")
    def test_auth_header_has_no_extra_keys(self, _):
        client = XAIClient("xai-testkey")
        self.assertEqual(set(client._auth_headers()), {"Authorization", "Content-Type"})

    @patch("urllib3.PoolManager")
    def test_default_base_url(self, _):
        client = XAIClient("key")
        self.assertEqual(client._base_url, "https://api.x.ai")

    @patch("urllib3.PoolManager")
    def test_url_override(self, _):
        client = XAIClient("key", url="https://my-proxy.example.com")
        self.assertEqual(client._base_url, "https://my-proxy.example.com")

    @patch("urllib3.PoolManager")
    def test_list_models_parses_data_array(self, mock_pool):
        fake = {"data": [{"id": "grok-3"}, {"id": "grok-3-mini"}]}
        mock_pool.return_value.request.return_value = MagicMock(
            status=200, data=json.dumps(fake).encode()
        )
        models = XAIClient("key").list_models()
        self.assertEqual(models, ["grok-3", "grok-3-mini"])

    @patch("urllib3.PoolManager")
    def test_list_models_returns_list_of_strings(self, mock_pool):
        fake = {"data": [{"id": "grok-3"}]}
        mock_pool.return_value.request.return_value = MagicMock(
            status=200, data=json.dumps(fake).encode()
        )
        models = XAIClient("key").list_models()
        for m in models:
            self.assertIsInstance(m, str)


# ---------------------------------------------------------------------------
# Invalid-auth tests — always run, no valid key required
# ---------------------------------------------------------------------------

class TestXAIInvalidAuth(unittest.TestCase):
    """Fake key → check_auth must return False; list_models must raise APIError."""

    def setUp(self):
        self.client = XAIClient(_FAKE)

    def test_check_auth_returns_false(self):
        self.assertFalse(self.client.check_auth())

    def test_list_models_raises_api_error(self):
        with self.assertRaises(APIError) as ctx:
            self.client.list_models()
        self.assertGreaterEqual(ctx.exception.status, 400)
        self.assertLess(ctx.exception.status, 500)


# ---------------------------------------------------------------------------
# Live tests — skip unless XAI_API_KEY is set
# ---------------------------------------------------------------------------

@unittest.skipUnless(_KEY, "Set XAI_API_KEY to run live tests")
class TestXAILive(unittest.TestCase):
    """Valid key → check_auth True; list_models returns a usable model list."""

    @classmethod
    def setUpClass(cls):
        cls.client = XAIClient(_KEY)

    def test_check_auth_returns_true(self):
        self.assertTrue(self.client.check_auth())

    def test_list_models_returns_non_empty_list(self):
        models = self.client.list_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)

    def test_list_models_returns_strings(self):
        models = self.client.list_models()
        for m in models:
            self.assertIsInstance(m, str)

    def test_list_models_contains_known_model(self):
        models = self.client.list_models()
        grok_models = [m for m in models if "grok" in m]
        self.assertGreater(len(grok_models), 0)

    def test_url_override_reaches_same_api(self):
        """Explicitly passing the default URL must behave identically."""
        client = XAIClient(_KEY, url="https://api.x.ai")
        self.assertTrue(client.check_auth())


if __name__ == "__main__":
    unittest.main()
