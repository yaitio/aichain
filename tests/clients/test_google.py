"""
Integration tests for clients._google — GoogleAIClient.

Two test classes:

  InvalidAuth — always runs; makes a real HTTP request with a fake key
                and verifies the API rejects it (check_auth returns False,
                list_models raises APIError in 4xx range).

                Google returns 400 (INVALID_ARGUMENT) for a structurally
                malformed key and 403 (PERMISSION_DENIED) for a valid-form
                but unauthorised key — both are acceptable here.

  Live        — skipped unless GOOGLE_AI_API_KEY is set; verifies that
                check_auth returns True and list_models returns a
                well-formed list containing known models.

Run live tests:
    GOOGLE_AI_API_KEY=AIza... python3 -m unittest discover -s tests/clients -t . -v
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

from clients._base import APIError
from clients._google import GoogleAIClient

_KEY  = os.getenv("GOOGLE_AI_API_KEY")
_FAKE = "AIzaInvalidKeyForTesting000000000000000"


# ---------------------------------------------------------------------------
# Unit tests — no network, HTTP layer mocked
# ---------------------------------------------------------------------------

class TestGoogleAIUnit(unittest.TestCase):
    """Auth headers, base URL, and list_models response parsing — all mocked."""

    @patch("urllib3.PoolManager")
    def test_auth_does_not_use_authorization_header(self, _):
        # Google authenticates via ?key= query param, not Authorization header.
        client = GoogleAIClient("AIzaTestKey")
        self.assertNotIn("Authorization", client._auth_headers())

    @patch("urllib3.PoolManager")
    def test_auth_header_includes_content_type(self, _):
        client = GoogleAIClient("AIzaTestKey")
        self.assertEqual(client._auth_headers()["Content-Type"], "application/json")

    @patch("urllib3.PoolManager")
    def test_default_base_url(self, _):
        client = GoogleAIClient("key")
        self.assertEqual(
            client._base_url,
            "https://generativelanguage.googleapis.com/v1beta",
        )

    @patch("urllib3.PoolManager")
    def test_url_override(self, _):
        client = GoogleAIClient("key", url="https://my-proxy.example.com/v1beta")
        self.assertEqual(client._base_url, "https://my-proxy.example.com/v1beta")

    @patch("urllib3.PoolManager")
    def test_list_models_strips_models_prefix(self, mock_pool):
        fake = {
            "models": [
                {"name": "models/gemini-2.0-flash"},
                {"name": "models/gemini-1.5-pro"},
            ]
        }
        mock_pool.return_value.request.return_value = MagicMock(
            status=200, data=json.dumps(fake).encode()
        )
        models = GoogleAIClient("key").list_models()
        self.assertEqual(models, ["gemini-2.0-flash", "gemini-1.5-pro"])

    @patch("urllib3.PoolManager")
    def test_list_models_returns_list_of_strings(self, mock_pool):
        fake = {"models": [{"name": "models/gemini-2.0-flash"}]}
        mock_pool.return_value.request.return_value = MagicMock(
            status=200, data=json.dumps(fake).encode()
        )
        models = GoogleAIClient("key").list_models()
        for m in models:
            self.assertIsInstance(m, str)


# ---------------------------------------------------------------------------
# Invalid-auth tests — always run, no valid key required
# ---------------------------------------------------------------------------

class TestGoogleAIInvalidAuth(unittest.TestCase):
    """
    Fake key → check_auth must return False; list_models must raise a 4xx
    APIError (400 or 403 depending on key format).
    """

    def setUp(self):
        self.client = GoogleAIClient(_FAKE)

    def test_check_auth_returns_false(self):
        self.assertFalse(self.client.check_auth())

    def test_list_models_raises_4xx_api_error(self):
        with self.assertRaises(APIError) as ctx:
            self.client.list_models()
        self.assertGreaterEqual(ctx.exception.status, 400)
        self.assertLess(ctx.exception.status, 500)


# ---------------------------------------------------------------------------
# Live tests — skip unless GOOGLE_AI_API_KEY is set
# ---------------------------------------------------------------------------

@unittest.skipUnless(_KEY, "Set GOOGLE_AI_API_KEY to run live tests")
class TestGoogleAILive(unittest.TestCase):
    """Valid key → check_auth True; list_models returns a usable model list."""

    @classmethod
    def setUpClass(cls):
        cls.client = GoogleAIClient(_KEY)

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

    def test_list_models_strips_models_prefix(self):
        """Returned IDs must not contain the 'models/' prefix."""
        models = self.client.list_models()
        for m in models:
            self.assertFalse(m.startswith("models/"), f"Unexpected prefix in: {m!r}")

    def test_list_models_contains_known_model(self):
        models = self.client.list_models()
        gemini_models = [m for m in models if "gemini" in m]
        self.assertGreater(len(gemini_models), 0)

    def test_url_override_reaches_same_api(self):
        """Explicitly passing the default URL must behave identically."""
        client = GoogleAIClient(
            _KEY,
            url="https://generativelanguage.googleapis.com/v1beta",
        )
        self.assertTrue(client.check_auth())


if __name__ == "__main__":
    unittest.main()
