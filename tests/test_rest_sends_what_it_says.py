"""
A REST endpoint sends what its settings say, and refuses what it would drop.

Every field was a string with no description; a typo in auth or an unset
variable sent the request anonymous; ``True`` went as "True" and a list as
its repr; a form content type got a JSON body; a field named ``input`` or
``options`` could not be sent; ``timeout=5`` removed the connect timeout;
retries spent were a raw ``MaxRetryError``; ``{id}`` without ``path_params``
went as written; a private host needed a process-wide flag; and a key in the
URL's query was in every error.
"""

import json
import os
import unittest
import urllib.parse
from unittest.mock import MagicMock, patch

from yait_aichain.clients._errors import NotFoundError
from yait_aichain.utilities.rest import REST, RestApiTool


def _tool(*a, **kw):
    t = RestApiTool(*a, **kw)
    t._http = MagicMock()
    t._http.request.return_value = MagicMock(status=200, read=MagicMock(return_value=b'{"ok": 1}'),
                                             headers={"content-type": "application/json"})
    return t


def _sent(t):
    call = t._http.request.call_args
    return call.args[0], call.args[1], call.kwargs


@patch("yait_aichain.tools._security.assert_safe_url", lambda u, **k: None)
class TestRest(unittest.TestCase):

    def test_fields_are_typed(self):
        t = _tool("search", "Search.", "GET", "https://x.example/s", query_params=["limit", "tags"],
                  fields={"limit": (int, "page size"), "tags": ("array", "labels")})
        props = t.parameters["properties"]
        self.assertEqual((props["limit"]["type"], props["limit"]["description"]), ("integer", "page size"))
        self.assertEqual(props["tags"]["type"], "array")

    def test_bools_and_lists_in_the_query(self):
        t = _tool("s", "S.", "GET", "https://x.example/s", query_params=["open", "tag"])
        t.run(open=True, tag=["a", "b"])
        _, url, _ = _sent(t)
        self.assertEqual(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query),
                         {"open": ["true"], "tag": ["a", "b"]})

    def test_a_form_body_is_a_form(self):
        t = _tool("s", "S.", "POST", "https://x.example/s", body_params=["a"],
                  content_type="application/x-www-form-urlencoded")
        t.run(a="1 2")
        self.assertEqual(_sent(t)[2]["body"], b"a=1+2")

    def test_fields_named_input_and_options_are_sent(self):
        t = _tool("s", "S.", "POST", "https://x.example/s", body_params=["input", "options"])
        t.run(input=["x"], options={"k": 1})
        self.assertEqual(json.loads(_sent(t)[2]["body"]), {"input": ["x"], "options": {"k": 1}})

    def test_a_timeout_keeps_the_connect_timeout(self):
        t = _tool("s", "S.", "GET", "https://x.example/s", timeout=5)
        t.run()
        self.assertEqual(_sent(t)[2]["timeout"].connect_timeout, 10.0)

    def test_an_error_is_typed_and_hides_the_query_key(self):
        os.environ["SOME_REST_KEY"] = "k-123"
        try:
            t = _tool("s", "S.", "GET", "https://x.example/s",
                      auth={"type": "query", "param": "key", "env": "SOME_REST_KEY"})
            t._http.request.return_value = MagicMock(status=404, read=MagicMock(return_value=b"nope"),
                                                     headers={})
            with self.assertRaises(NotFoundError) as ctx:
                t.run()
            self.assertIn("key=k-123", _sent(t)[1])
            self.assertNotIn("k-123", str(ctx.exception))
        finally:
            del os.environ["SOME_REST_KEY"]

    def test_an_unset_token_is_an_error_not_an_anonymous_request(self):
        t = _tool("s", "S.", "GET", "https://x.example/s",
                  auth={"type": "bearer", "env": "SURELY_NOT_SET_ANYWHERE_123"})
        with self.assertRaisesRegex(ValueError, "SURELY_NOT_SET"):
            t.run()
        t._http.request.assert_not_called()


class TestRefusedAtConstruction(unittest.TestCase):

    def test_an_unknown_auth_key_or_type(self):
        with self.assertRaisesRegex(ValueError, "ignored"):
            RestApiTool("s", "S.", "GET", "https://x.example/s", auth={"type": "bearer", "evn": "X"})
        with self.assertRaisesRegex(ValueError, "auth type"):
            RestApiTool("s", "S.", "GET", "https://x.example/s", auth={"type": "oauth", "env": "X"})

    def test_a_url_placeholder_without_a_path_param(self):
        with self.assertRaisesRegex(ValueError, "path_params does not"):
            RestApiTool("s", "S.", "GET", "https://x.example/items/{id}")

    def test_a_field_placed_twice_or_badly_named(self):
        with self.assertRaisesRegex(ValueError, "twice"):
            RestApiTool("s", "S.", "POST", "https://x.example/s", query_params=["a"], body_params=["a"])
        with self.assertRaisesRegex(ValueError, "field names"):
            RestApiTool("s", "S.", "GET", "https://x.example/s", query_params=["a-b"])

    def test_a_key_in_the_url_is_refused(self):
        with self.assertRaisesRegex(ValueError, "credential"):
            RestApiTool("s", "S.", "GET", "https://x.example/s?api_key=abc")

    def test_private_hosts_are_one_apis_setting(self):
        api = REST("http://10.0.0.5", description="Internal API", allow_private=True)
        self.assertTrue(api.endpoint("ping", "GET", "/ping", description="Health.")._allow_private)


if __name__ == "__main__":
    unittest.main()
