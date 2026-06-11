"""
Tests for clients._qwen — QwenClient and resolve_qwen_base_url.

Three test classes:

  Unit        — no network; mocks urllib3.PoolManager to verify auth headers,
                base URLs per region, and list_models response parsing.

  InvalidAuth — always runs; makes a real HTTP request with a fake key and
                verifies the API rejects it (check_auth returns False).

  Live        — skipped unless DASHSCOPE_API_KEY is set; verifies that
                check_auth returns True and list_models returns a well-formed
                list.

Run live tests:
    DASHSCOPE_API_KEY=sk-... python3 -m unittest discover -s tests/clients -t . -v
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

from clients._base import APIError
from clients._qwen import QwenClient, resolve_qwen_base_url, REGION_URLS

_KEY  = os.getenv("DASHSCOPE_API_KEY")
_FAKE = "sk-invalid-qwen-key-for-testing-000000000000000000000000"


# ---------------------------------------------------------------------------
# Unit tests — no network, HTTP layer mocked
# ---------------------------------------------------------------------------

class TestQwenUnit(unittest.TestCase):
    """Auth headers, region-aware base URLs, and list_models parsing — all mocked."""

    # ── Auth ──────────────────────────────────────────────────────────────

    @patch("urllib3.PoolManager")
    def test_auth_header_uses_bearer_scheme(self, _):
        client = QwenClient("sk-testkey")
        self.assertEqual(client._auth_headers()["Authorization"], "Bearer sk-testkey")

    @patch("urllib3.PoolManager")
    def test_auth_header_includes_content_type(self, _):
        client = QwenClient("sk-testkey")
        self.assertEqual(client._auth_headers()["Content-Type"], "application/json")

    @patch("urllib3.PoolManager")
    def test_auth_header_has_no_extra_keys(self, _):
        client = QwenClient("sk-testkey")
        self.assertEqual(set(client._auth_headers()), {"Authorization", "Content-Type"})

    # ── Region-aware base URLs ─────────────────────────────────────────────

    @patch("urllib3.PoolManager")
    def test_default_region_is_ap(self, _):
        client = QwenClient("key")
        self.assertEqual(client._base_url, REGION_URLS["ap"])

    @patch("urllib3.PoolManager")
    def test_region_ap_uses_intl_endpoint(self, _):
        client = QwenClient("key", region="ap")
        self.assertIn("dashscope-intl.aliyuncs.com", client._base_url)

    @patch("urllib3.PoolManager")
    def test_region_us_uses_us_endpoint(self, _):
        client = QwenClient("key", region="us")
        self.assertIn("dashscope-us.aliyuncs.com", client._base_url)

    @patch("urllib3.PoolManager")
    def test_region_cn_uses_cn_endpoint(self, _):
        client = QwenClient("key", region="cn")
        self.assertIn("dashscope.aliyuncs.com", client._base_url)
        self.assertNotIn("intl", client._base_url)
        self.assertNotIn("hongkong", client._base_url)

    @patch("urllib3.PoolManager")
    def test_region_hk_uses_hongkong_endpoint(self, _):
        client = QwenClient("key", region="hk")
        self.assertIn("cn-hongkong.dashscope.aliyuncs.com", client._base_url)

    @patch("urllib3.PoolManager")
    def test_explicit_url_overrides_region(self, _):
        client = QwenClient("key", url="https://my-proxy.example.com", region="cn")
        self.assertEqual(client._base_url, "https://my-proxy.example.com")

    # ── resolve_qwen_base_url helper ──────────────────────────────────────

    def test_resolve_all_four_regions(self):
        for region, expected_url in REGION_URLS.items():
            with self.subTest(region=region):
                self.assertEqual(resolve_qwen_base_url(region), expected_url)

    def test_resolve_unknown_region_raises(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_qwen_base_url("eu")
        self.assertIn("eu", str(ctx.exception))

    def test_resolve_case_insensitive(self):
        self.assertEqual(resolve_qwen_base_url("AP"), REGION_URLS["ap"])
        self.assertEqual(resolve_qwen_base_url("CN"), REGION_URLS["cn"])

    def test_resolve_env_var_fallback(self):
        with patch.dict(os.environ, {"DASHSCOPE_REGION": "us"}):
            self.assertEqual(resolve_qwen_base_url(None), REGION_URLS["us"])

    def test_resolve_arg_beats_env_var(self):
        with patch.dict(os.environ, {"DASHSCOPE_REGION": "us"}):
            self.assertEqual(resolve_qwen_base_url("hk"), REGION_URLS["hk"])

    def test_resolve_default_when_no_arg_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DASHSCOPE_REGION", None)
            self.assertEqual(resolve_qwen_base_url(None), REGION_URLS["ap"])

    # ── list_models ────────────────────────────────────────────────────────

    @patch("urllib3.PoolManager")
    def test_list_models_parses_data_array(self, mock_pool):
        fake = {"data": [{"id": "qwen-max"}, {"id": "qwen-turbo"}, {"id": "qwen-plus"}]}
        mock_pool.return_value.request.return_value = MagicMock(
            status=200, data=json.dumps(fake).encode()
        )
        models = QwenClient("key").list_models()
        self.assertEqual(models, ["qwen-max", "qwen-turbo", "qwen-plus"])

    @patch("urllib3.PoolManager")
    def test_list_models_returns_list_of_strings(self, mock_pool):
        fake = {"data": [{"id": "qwen-max"}]}
        mock_pool.return_value.request.return_value = MagicMock(
            status=200, data=json.dumps(fake).encode()
        )
        models = QwenClient("key").list_models()
        for m in models:
            self.assertIsInstance(m, str)


# ---------------------------------------------------------------------------
# Invalid-auth tests — always run, no valid key required
# ---------------------------------------------------------------------------

class TestQwenInvalidAuth(unittest.TestCase):
    """Fake key → check_auth must return False."""

    def setUp(self):
        self.client = QwenClient(_FAKE)

    def test_check_auth_returns_false(self):
        self.assertFalse(self.client.check_auth())

    def test_list_models_raises_4xx_api_error(self):
        with self.assertRaises(APIError) as ctx:
            self.client.list_models()
        self.assertGreaterEqual(ctx.exception.status, 400)
        self.assertLess(ctx.exception.status, 500)


# ---------------------------------------------------------------------------
# Live tests — skip unless DASHSCOPE_API_KEY is set
# ---------------------------------------------------------------------------

@unittest.skipUnless(_KEY, "Set DASHSCOPE_API_KEY to run live tests")
class TestQwenLive(unittest.TestCase):
    """Valid key → check_auth True; list_models returns a usable model list."""

    @classmethod
    def setUpClass(cls):
        cls.client = QwenClient(_KEY)

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
        qwen_models = [m for m in models if m.startswith("qwen-")]
        self.assertGreater(len(qwen_models), 0)

    def test_region_ap_default_works(self):
        client = QwenClient(_KEY, region="ap")
        self.assertTrue(client.check_auth())


if __name__ == "__main__":
    unittest.main()
