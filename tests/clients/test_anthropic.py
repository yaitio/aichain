"""
Integration tests for clients._anthropic — AnthropicClient.

Two test classes:

  InvalidAuth — always runs; makes a real HTTP request with a fake key
                and verifies the API rejects it (check_auth returns False,
                list_models raises APIError 401).

  Live        — skipped unless ANTHROPIC_API_KEY is set; verifies that
                check_auth returns True and list_models returns a
                well-formed list containing known models.

Run live tests:
    ANTHROPIC_API_KEY=sk-ant-... python3 -m unittest discover -s tests/clients -t . -v
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

from clients._base import APIError
from clients._anthropic import AnthropicClient

_KEY  = os.getenv("ANTHROPIC_API_KEY")
_FAKE = "sk-ant-invalid-key-for-testing-000000000000000000000000000000000000"


# ---------------------------------------------------------------------------
# Unit tests — no network, HTTP layer mocked
# ---------------------------------------------------------------------------

class TestAnthropicUnit(unittest.TestCase):
    """Auth headers, base URL, and list_models response parsing — all mocked."""

    @patch("urllib3.PoolManager")
    def test_auth_uses_x_api_key_header_not_bearer(self, _):
        client = AnthropicClient("sk-ant-testkey")
        headers = client._auth_headers()
        self.assertEqual(headers["x-api-key"], "sk-ant-testkey")
        self.assertNotIn("Authorization", headers)

    @patch("urllib3.PoolManager")
    def test_auth_header_includes_anthropic_version(self, _):
        client = AnthropicClient("sk-ant-testkey")
        self.assertEqual(client._auth_headers()["anthropic-version"], "2023-06-01")

    @patch("urllib3.PoolManager")
    def test_auth_header_includes_content_type(self, _):
        client = AnthropicClient("sk-ant-testkey")
        self.assertEqual(client._auth_headers()["Content-Type"], "application/json")

    @patch("urllib3.PoolManager")
    def test_default_base_url(self, _):
        client = AnthropicClient("key")
        self.assertEqual(client._base_url, "https://api.anthropic.com")

    @patch("urllib3.PoolManager")
    def test_url_override(self, _):
        client = AnthropicClient("key", url="https://my-proxy.example.com")
        self.assertEqual(client._base_url, "https://my-proxy.example.com")

    @patch("urllib3.PoolManager")
    def test_list_models_parses_data_array(self, mock_pool):
        fake = {"data": [{"id": "claude-opus-4-6"}, {"id": "claude-sonnet-4-6"}]}
        mock_pool.return_value.request.return_value = MagicMock(
            status=200, data=json.dumps(fake).encode()
        )
        models = AnthropicClient("key").list_models()
        self.assertEqual(models, ["claude-opus-4-6", "claude-sonnet-4-6"])

    @patch("urllib3.PoolManager")
    def test_list_models_returns_list_of_strings(self, mock_pool):
        fake = {"data": [{"id": "claude-opus-4-6"}]}
        mock_pool.return_value.request.return_value = MagicMock(
            status=200, data=json.dumps(fake).encode()
        )
        models = AnthropicClient("key").list_models()
        for m in models:
            self.assertIsInstance(m, str)


# ---------------------------------------------------------------------------
# Invalid-auth tests — always run, no valid key required
# ---------------------------------------------------------------------------

class TestAnthropicInvalidAuth(unittest.TestCase):
    """Fake key → check_auth must return False; list_models must raise 401."""

    def setUp(self):
        self.client = AnthropicClient(_FAKE)

    def test_check_auth_returns_false(self):
        self.assertFalse(self.client.check_auth())

    def test_list_models_raises_api_error_401(self):
        with self.assertRaises(APIError) as ctx:
            self.client.list_models()
        self.assertEqual(ctx.exception.status, 401)


# ---------------------------------------------------------------------------
# Live tests — skip unless ANTHROPIC_API_KEY is set
# ---------------------------------------------------------------------------

@unittest.skipUnless(_KEY, "Set ANTHROPIC_API_KEY to run live tests")
class TestAnthropicLive(unittest.TestCase):
    """Valid key → check_auth True; list_models returns a usable model list."""

    @classmethod
    def setUpClass(cls):
        cls.client = AnthropicClient(_KEY)

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
        # Any current Claude model should be present.
        claude_models = [m for m in models if m.startswith("claude-")]
        self.assertGreater(len(claude_models), 0)

    def test_url_override_reaches_same_api(self):
        """Explicitly passing the default URL must behave identically."""
        client = AnthropicClient(_KEY, url="https://api.anthropic.com")
        self.assertTrue(client.check_auth())


if __name__ == "__main__":
    unittest.main()
