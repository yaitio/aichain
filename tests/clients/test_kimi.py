"""
Tests for clients._kimi — KimiClient.

Three test classes:

  Unit        — no network; mocks urllib3.PoolManager to verify auth headers,
                base URL, and list_models response parsing.

  InvalidAuth — always runs; makes a real HTTP request with a fake key and
                verifies the API rejects it (check_auth returns False,
                list_models raises APIError).

  Live        — skipped unless MOONSHOT_API_KEY is set; verifies that
                check_auth returns True and list_models returns a well-formed
                list containing known models.

Run live tests:
    MOONSHOT_API_KEY=sk-... python3 -m unittest discover -s tests/clients -t . -v
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

from clients._base import APIError
from clients._kimi import KimiClient

_KEY  = os.getenv("MOONSHOT_API_KEY")
_FAKE = "sk-invalid-kimi-key-for-testing-000000000000000000000000"


# ---------------------------------------------------------------------------
# Unit tests — no network, HTTP layer mocked
# ---------------------------------------------------------------------------

class TestKimiUnit(unittest.TestCase):
    """Auth headers, base URL, and list_models response parsing — all mocked."""

    @patch("urllib3.PoolManager")
    def test_auth_header_uses_bearer_scheme(self, _):
        client = KimiClient("sk-testkey")
        self.assertEqual(client._auth_headers()["Authorization"], "Bearer sk-testkey")

    @patch("urllib3.PoolManager")
    def test_auth_header_includes_content_type(self, _):
        client = KimiClient("sk-testkey")
        self.assertEqual(client._auth_headers()["Content-Type"], "application/json")

    @patch("urllib3.PoolManager")
    def test_auth_header_has_no_extra_keys(self, _):
        client = KimiClient("sk-testkey")
        self.assertEqual(set(client._auth_headers()), {"Authorization", "Content-Type"})

    @patch("urllib3.PoolManager")
    def test_default_base_url(self, _):
        client = KimiClient("key")
        self.assertEqual(client._base_url, "https://api.moonshot.ai")

    @patch("urllib3.PoolManager")
    def test_url_override(self, _):
        client = KimiClient("key", url="https://my-proxy.example.com")
        self.assertEqual(client._base_url, "https://my-proxy.example.com")

    @patch("urllib3.PoolManager")
    def test_list_models_parses_data_array(self, mock_pool):
        fake = {
            "data": [
                {"id": "moonshot-v1-8k"},
                {"id": "moonshot-v1-32k"},
                {"id": "kimi-k2"},
            ]
        }
        mock_pool.return_value.request.return_value = MagicMock(
            status=200, data=json.dumps(fake).encode()
        )
        models = KimiClient("key").list_models()
        self.assertEqual(models, ["moonshot-v1-8k", "moonshot-v1-32k", "kimi-k2"])

    @patch("urllib3.PoolManager")
    def test_list_models_returns_list_of_strings(self, mock_pool):
        fake = {"data": [{"id": "moonshot-v1-8k"}]}
        mock_pool.return_value.request.return_value = MagicMock(
            status=200, data=json.dumps(fake).encode()
        )
        models = KimiClient("key").list_models()
        for m in models:
            self.assertIsInstance(m, str)


# ---------------------------------------------------------------------------
# Invalid-auth tests — always run, no valid key required
# ---------------------------------------------------------------------------

class TestKimiInvalidAuth(unittest.TestCase):
    """Fake key → check_auth must return False; list_models must raise APIError."""

    def setUp(self):
        self.client = KimiClient(_FAKE)

    def test_check_auth_returns_false(self):
        self.assertFalse(self.client.check_auth())

    def test_list_models_raises_api_error(self):
        with self.assertRaises(APIError) as ctx:
            self.client.list_models()
        self.assertGreaterEqual(ctx.exception.status, 400)
        self.assertLess(ctx.exception.status, 500)


# ---------------------------------------------------------------------------
# Live tests — skip unless MOONSHOT_API_KEY is set
# ---------------------------------------------------------------------------

@unittest.skipUnless(_KEY, "Set MOONSHOT_API_KEY to run live tests")
class TestKimiLive(unittest.TestCase):
    """Valid key → check_auth True; list_models returns a usable model list."""

    @classmethod
    def setUpClass(cls):
        cls.client = KimiClient(_KEY)

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
        kimi_models = [m for m in models if "moonshot" in m or "kimi" in m]
        self.assertGreater(len(kimi_models), 0)

    def test_url_override_reaches_same_api(self):
        client = KimiClient(_KEY, url="https://api.moonshot.ai")
        self.assertTrue(client.check_auth())


if __name__ == "__main__":
    unittest.main()
