"""
Unit tests for clients._base — BaseClient and APIError.

These tests cover pure infrastructure logic with no network dependency:
proxy setup, URL normalisation, auth-header contract, check_auth()
behaviour, and APIError structure.
"""

import base64
import unittest
from unittest.mock import patch, MagicMock

from clients._base import BaseClient, APIError


# ---------------------------------------------------------------------------
# Minimal concrete subclass needed to instantiate BaseClient directly
# ---------------------------------------------------------------------------

class ConcreteClient(BaseClient):
    BASE_URL = "https://api.example.com"

    def _auth_headers(self) -> dict:
        return {"Authorization": "Bearer test", "Content-Type": "application/json"}

    def list_models(self) -> list[str]:
        return ["model-a", "model-b"]


# ---------------------------------------------------------------------------
# APIError
# ---------------------------------------------------------------------------

class TestAPIError(unittest.TestCase):

    def test_stores_status_and_message(self):
        err = APIError(404, "not found")
        self.assertEqual(err.status, 404)
        self.assertEqual(err.message, "not found")

    def test_str_contains_status_and_message(self):
        err = APIError(500, "internal server error")
        self.assertIn("500", str(err))
        self.assertIn("internal server error", str(err))

    def test_is_exception(self):
        self.assertTrue(issubclass(APIError, Exception))


# ---------------------------------------------------------------------------
# BaseClient.__init__ — URL and connection pool
# ---------------------------------------------------------------------------

class TestBaseClientInit(unittest.TestCase):

    @patch("urllib3.PoolManager")
    def test_default_url_comes_from_base_url_class_constant(self, _):
        client = ConcreteClient("key")
        self.assertEqual(client._base_url, "https://api.example.com")

    @patch("urllib3.PoolManager")
    def test_url_param_overrides_base_url_class_constant(self, _):
        client = ConcreteClient("key", url="https://custom.example.com")
        self.assertEqual(client._base_url, "https://custom.example.com")

    @patch("urllib3.PoolManager")
    def test_trailing_slash_stripped_from_url_param(self, _):
        client = ConcreteClient("key", url="https://custom.example.com/")
        self.assertEqual(client._base_url, "https://custom.example.com")

    @patch("urllib3.PoolManager")
    def test_api_key_stored(self, _):
        client = ConcreteClient("my-secret")
        self.assertEqual(client._api_key, "my-secret")

    @patch("urllib3.PoolManager")
    def test_no_proxy_uses_pool_manager(self, MockPM):
        client = ConcreteClient("key")
        MockPM.assert_called_once()
        self.assertIs(client._http, MockPM.return_value)

    @patch("urllib3.ProxyManager")
    @patch("urllib3.PoolManager")
    def test_proxy_uses_proxy_manager_not_pool_manager(self, MockPM, MockProxy):
        ConcreteClient("key", proxy={"url": "http://proxy:3128"})
        MockProxy.assert_called_once()
        MockPM.assert_not_called()

    @patch("urllib3.ProxyManager")
    def test_proxy_with_credentials_sets_basic_auth_header(self, MockProxy):
        ConcreteClient(
            "key",
            proxy={"url": "http://proxy:3128", "username": "alice", "password": "secret"},
        )
        _, kwargs = MockProxy.call_args
        expected = "Basic " + base64.b64encode(b"alice:secret").decode()
        self.assertEqual(kwargs["proxy_headers"]["Proxy-Authorization"], expected)

    @patch("urllib3.ProxyManager")
    def test_proxy_without_credentials_omits_auth_header(self, MockProxy):
        ConcreteClient("key", proxy={"url": "http://proxy:3128"})
        _, kwargs = MockProxy.call_args
        self.assertNotIn("Proxy-Authorization", kwargs.get("proxy_headers", {}))


# ---------------------------------------------------------------------------
# BaseClient._auth_headers — abstract contract
# ---------------------------------------------------------------------------

class TestBaseClientAuthHeaders(unittest.TestCase):

    @patch("urllib3.PoolManager")
    def test_base_class_raises_not_implemented(self, _):
        client = object.__new__(BaseClient)
        BaseClient.__init__(client, "key")
        with self.assertRaises(NotImplementedError):
            client._auth_headers()


# ---------------------------------------------------------------------------
# BaseClient.list_models — abstract contract
# ---------------------------------------------------------------------------

class TestBaseClientListModels(unittest.TestCase):

    @patch("urllib3.PoolManager")
    def test_base_class_raises_not_implemented(self, _):
        client = object.__new__(BaseClient)
        BaseClient.__init__(client, "key")
        with self.assertRaises(NotImplementedError):
            client.list_models()


# ---------------------------------------------------------------------------
# BaseClient.check_auth — concrete logic (delegates to list_models)
# ---------------------------------------------------------------------------

class TestBaseClientCheckAuth(unittest.TestCase):

    @patch("urllib3.PoolManager")
    def test_returns_true_when_list_models_succeeds(self, _):
        client = ConcreteClient("key")
        self.assertTrue(client.check_auth())

    @patch("urllib3.PoolManager")
    def test_returns_false_on_401(self, _):
        client = ConcreteClient("key")
        client.list_models = MagicMock(side_effect=APIError(401, "Unauthorized"))
        self.assertFalse(client.check_auth())

    @patch("urllib3.PoolManager")
    def test_returns_false_on_403(self, _):
        client = ConcreteClient("key")
        client.list_models = MagicMock(side_effect=APIError(403, "Forbidden"))
        self.assertFalse(client.check_auth())

    @patch("urllib3.PoolManager")
    def test_returns_false_on_400(self, _):
        # Covers providers that use 400 for invalid credentials (e.g. xAI).
        client = ConcreteClient("key")
        client.list_models = MagicMock(side_effect=APIError(400, "Bad Request"))
        self.assertFalse(client.check_auth())

    @patch("urllib3.PoolManager")
    def test_reraises_on_500(self, _):
        client = ConcreteClient("key")
        client.list_models = MagicMock(side_effect=APIError(500, "Server Error"))
        with self.assertRaises(APIError) as ctx:
            client.check_auth()
        self.assertEqual(ctx.exception.status, 500)

    @patch("urllib3.PoolManager")
    def test_reraises_on_network_error(self, _):
        client = ConcreteClient("key")
        client.list_models = MagicMock(side_effect=APIError(0, "connection refused"))
        with self.assertRaises(APIError) as ctx:
            client.check_auth()
        self.assertEqual(ctx.exception.status, 0)


if __name__ == "__main__":
    unittest.main()
