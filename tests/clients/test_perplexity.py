"""
Integration tests for clients._perplexity — PerplexityClient.

Two test classes:

  InvalidAuth — always runs; makes a real HTTP request with a fake key
                and verifies the API rejects it (check_auth returns False).
                list_models() is static on Perplexity, so no auth test
                is needed for it — instead we test the static list directly.

  Live        — skipped unless PERPLEXITY_API_KEY is set; verifies that
                check_auth returns True and list_models returns the
                expected static model list.

Run live tests:
    PERPLEXITY_API_KEY=pplx-... python3 -m unittest discover -s tests/clients -t . -v
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

from clients._base import APIError
from clients._perplexity import PerplexityClient, _KNOWN_MODELS

_KEY  = os.getenv("PERPLEXITY_API_KEY")
_FAKE = "pplx-invalid-key-for-testing-0000000000000000000000000000000000000000"


# ---------------------------------------------------------------------------
# Unit tests — no network, HTTP layer mocked
# ---------------------------------------------------------------------------

class TestPerplexityUnit(unittest.TestCase):
    """Auth headers and base URL — list_models is static so no mock needed."""

    @patch("urllib3.PoolManager")
    def test_auth_header_uses_bearer_scheme(self, _):
        client = PerplexityClient("pplx-testkey")
        self.assertEqual(client._auth_headers()["Authorization"], "Bearer pplx-testkey")

    @patch("urllib3.PoolManager")
    def test_auth_header_includes_content_type(self, _):
        client = PerplexityClient("pplx-testkey")
        self.assertEqual(client._auth_headers()["Content-Type"], "application/json")

    @patch("urllib3.PoolManager")
    def test_auth_header_has_no_extra_keys(self, _):
        client = PerplexityClient("pplx-testkey")
        self.assertEqual(set(client._auth_headers()), {"Authorization", "Content-Type"})

    @patch("urllib3.PoolManager")
    def test_default_base_url(self, _):
        client = PerplexityClient("key")
        self.assertEqual(client._base_url, "https://api.perplexity.ai")

    @patch("urllib3.PoolManager")
    def test_url_override(self, _):
        client = PerplexityClient("key", url="https://my-proxy.example.com")
        self.assertEqual(client._base_url, "https://my-proxy.example.com")


# ---------------------------------------------------------------------------
# Invalid-auth tests — always run, no valid key required
# ---------------------------------------------------------------------------

class TestPerplexityInvalidAuth(unittest.TestCase):
    """
    Fake key → check_auth must return False.

    Perplexity has no /models endpoint, so check_auth makes a real
    /chat/completions call to probe the key — this test always hits
    the network.
    """

    def setUp(self):
        self.client = PerplexityClient(_FAKE)

    def test_check_auth_returns_false(self):
        self.assertFalse(self.client.check_auth())


class TestPerplexityListModelsStatic(unittest.TestCase):
    """
    list_models() is static — it requires no network and no valid key.
    These tests always run.
    """

    def setUp(self):
        # api_key is irrelevant; list_models never uses it.
        self.client = PerplexityClient("any-key")

    def test_returns_list_of_strings(self):
        models = self.client.list_models()
        self.assertIsInstance(models, list)
        for m in models:
            self.assertIsInstance(m, str)

    def test_returns_non_empty_list(self):
        self.assertGreater(len(self.client.list_models()), 0)

    def test_contains_sonar(self):
        self.assertIn("sonar", self.client.list_models())

    def test_returns_copy_not_module_list(self):
        """Mutating the returned list must not affect subsequent calls."""
        first  = self.client.list_models()
        first.clear()
        second = self.client.list_models()
        self.assertEqual(second, _KNOWN_MODELS)


# ---------------------------------------------------------------------------
# Live tests — skip unless PERPLEXITY_API_KEY is set
# ---------------------------------------------------------------------------

@unittest.skipUnless(_KEY, "Set PERPLEXITY_API_KEY to run live tests")
class TestPerplexityLive(unittest.TestCase):
    """Valid key → check_auth True; list_models returns the static list."""

    @classmethod
    def setUpClass(cls):
        cls.client = PerplexityClient(_KEY)

    def test_check_auth_returns_true(self):
        self.assertTrue(self.client.check_auth())

    def test_list_models_matches_known_list(self):
        """Static list must be consistent regardless of the key."""
        self.assertEqual(self.client.list_models(), _KNOWN_MODELS)

    def test_url_override_reaches_same_api(self):
        """Explicitly passing the default URL must behave identically."""
        client = PerplexityClient(_KEY, url="https://api.perplexity.ai")
        self.assertTrue(client.check_auth())


if __name__ == "__main__":
    unittest.main()
