"""
Tests for clients._deepseek — DeepSeekClient.

Three test classes:

  Unit        — no network; mocks urllib3.PoolManager to verify auth headers,
                base URL, and list_models response parsing.

  InvalidAuth — always runs; makes a real HTTP request with a fake key and
                verifies the API rejects it (check_auth returns False,
                list_models raises APIError).

  Live        — skipped unless DEEPSEEK_API_KEY is set; verifies that
                check_auth returns True and list_models returns a well-formed
                list containing known models.

Run live tests:
    DEEPSEEK_API_KEY=... python3 -m unittest discover -s tests/clients -t . -v
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

from clients._base import APIError
from clients._deepseek import DeepSeekClient

_KEY  = os.getenv("DEEPSEEK_API_KEY")
_FAKE = "sk-invalid-deepseek-key-for-testing-0000000000000000"


# ---------------------------------------------------------------------------
# Unit tests — no network, HTTP layer mocked
# ---------------------------------------------------------------------------

class TestDeepSeekUnit(unittest.TestCase):
    """Auth headers, base URL, and list_models response parsing — all mocked."""

    @patch("urllib3.PoolManager")
    def test_auth_header_uses_bearer_scheme(self, _):
        client = DeepSeekClient("sk-testkey")
        self.assertEqual(client._auth_headers()["Authorization"], "Bearer sk-testkey")

    @patch("urllib3.PoolManager")
    def test_auth_header_includes_content_type(self, _):
        client = DeepSeekClient("sk-testkey")
        self.assertEqual(client._auth_headers()["Content-Type"], "application/json")

    @patch("urllib3.PoolManager")
    def test_auth_header_has_no_extra_keys(self, _):
        client = DeepSeekClient("sk-testkey")
        self.assertEqual(set(client._auth_headers()), {"Authorization", "Content-Type"})

    @patch("urllib3.PoolManager")
    def test_default_base_url(self, _):
        client = DeepSeekClient("key")
        self.assertEqual(client._base_url, "https://api.deepseek.com")

    @patch("urllib3.PoolManager")
    def test_url_override(self, _):
        client = DeepSeekClient("key", url="https://api.deepseek.com/beta")
        self.assertEqual(client._base_url, "https://api.deepseek.com/beta")

    @patch("urllib3.PoolManager")
    def test_list_models_parses_data_array(self, mock_pool):
        fake = {"data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]}
        mock_pool.return_value.request.return_value = MagicMock(
            status=200, data=json.dumps(fake).encode()
        )
        models = DeepSeekClient("key").list_models()
        self.assertEqual(models, ["deepseek-chat", "deepseek-reasoner"])

    @patch("urllib3.PoolManager")
    def test_list_models_returns_list_of_strings(self, mock_pool):
        fake = {"data": [{"id": "deepseek-chat"}]}
        mock_pool.return_value.request.return_value = MagicMock(
            status=200, data=json.dumps(fake).encode()
        )
        models = DeepSeekClient("key").list_models()
        for m in models:
            self.assertIsInstance(m, str)


# ---------------------------------------------------------------------------
# Invalid-auth tests — always run, no valid key required
# ---------------------------------------------------------------------------

class TestDeepSeekInvalidAuth(unittest.TestCase):
    """Fake key → check_auth must return False; list_models must raise APIError."""

    def setUp(self):
        self.client = DeepSeekClient(_FAKE)

    def test_check_auth_returns_false(self):
        self.assertFalse(self.client.check_auth())

    def test_list_models_raises_api_error(self):
        with self.assertRaises(APIError) as ctx:
            self.client.list_models()
        self.assertGreaterEqual(ctx.exception.status, 400)
        self.assertLess(ctx.exception.status, 500)


# ---------------------------------------------------------------------------
# Live tests — skip unless DEEPSEEK_API_KEY is set
# ---------------------------------------------------------------------------

@unittest.skipUnless(_KEY, "Set DEEPSEEK_API_KEY to run live tests")
class TestDeepSeekLive(unittest.TestCase):
    """Valid key → check_auth True; list_models returns a usable model list."""

    @classmethod
    def setUpClass(cls):
        cls.client = DeepSeekClient(_KEY)

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
        deepseek_models = [m for m in models if m.startswith("deepseek-")]
        self.assertGreater(len(deepseek_models), 0)

    def test_url_override_reaches_same_api(self):
        client = DeepSeekClient(_KEY, url="https://api.deepseek.com")
        self.assertTrue(client.check_auth())


if __name__ == "__main__":
    unittest.main()
