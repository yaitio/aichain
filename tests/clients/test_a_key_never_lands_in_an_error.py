"""
A key never lands in an error message, and a bug is not the network.

A key read with `open().read()` ends in a newline; the header library raised
with the whole `Bearer sk-…` in its message, and the transport's
`except Exception` turned that into a NetworkError — retried, switching the
fallback model, and carried into events, logs and run documents. A body that
would not serialise became a NetworkError the same way.
"""

import unittest
from unittest.mock import MagicMock

from yait_aichain import Model, NetworkError
from yait_aichain.clients._http import request

SECRET = "sk-live-0123456789"


class TestTheKey(unittest.TestCase):

    def test_a_trailing_newline_is_stripped(self):
        m = Model("gpt-4o", api_key=SECRET + "\n")
        self.assertEqual(m.client._auth_headers()["Authorization"], f"Bearer {SECRET}")

    def test_a_control_character_inside_is_refused_without_the_value(self):
        m = Model("gpt-4o", api_key="sk-a\x00b")
        with self.assertRaises(ValueError) as caught:
            m.client._auth_headers()
        self.assertNotIn("sk-a", str(caught.exception))

    def test_a_tools_header_is_held_the_same_way(self):
        http = MagicMock()
        http.request.return_value = MagicMock(status=200)
        request(http, "GET", "https://x", headers={"x-api-key": SECRET + "\n"})
        self.assertEqual(http.request.call_args.kwargs["headers"]["x-api-key"], SECRET)
        with self.assertRaises(ValueError) as caught:
            request(http, "GET", "https://x", headers={"x-api-key": "sk\r\nInjected: 1"})
        self.assertNotIn("Injected", str(caught.exception))


class TestABugIsNotTheNetwork(unittest.TestCase):

    def test_a_body_that_does_not_serialise_raises_as_itself(self):
        http = MagicMock()
        with self.assertRaises(TypeError):
            request(http, "POST", "https://x", json={"a": object()})
        http.request.assert_not_called()

    def test_an_error_in_our_code_during_the_call_is_not_retried_as_network(self):
        http = MagicMock()
        http.request.side_effect = KeyError("ours")
        with self.assertRaises(KeyError):
            request(http, "GET", "https://x")

    def test_a_socket_error_is_still_a_network_error(self):
        http = MagicMock()
        http.request.side_effect = ConnectionResetError("reset")
        with self.assertRaises(NetworkError):
            request(http, "GET", "https://x")


if __name__ == "__main__":
    unittest.main()
