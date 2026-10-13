"""
A request reaches only public addresses, on every hop.

`assert_safe_url` checked the first URL's text and the HTTP library then
followed redirects on its own: a public page answering `302 Location:
http://169.254.169.254/` reached the metadata service, from fetchLocal (page
and robots.txt alike) and from REST. `http://127.0.0.1\\@example.com/` passed
because this parser and the connecting one disagree about its host, and
`100.100.100.200` — Alibaba's metadata — is not "private" to the stdlib.
"""

import ipaddress
import unittest
from unittest.mock import MagicMock, patch

from yait_aichain.tools import _security
from yait_aichain.tools._security import _public, assert_safe_url, safe_get, safe_request


class TestTheAddress(unittest.TestCase):

    def test_what_is_not_public(self):
        for ip in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "100.100.100.200", "100.64.0.1",
                   "::1", "::ffff:127.0.0.1", "64:ff9b::a9fe:a9fe", "224.0.0.1", "0.0.0.0"):
            with self.subTest(ip=ip):
                self.assertFalse(_public(ipaddress.ip_address(ip)))
        self.assertTrue(_public(ipaddress.ip_address("93.184.216.34")))

    def test_a_url_the_parsers_read_differently_is_refused(self):
        for url in ("http://127.0.0.1\\@example.com/", "http://user:pw@example.com/",
                    "http://user@example.com/"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    assert_safe_url(url, allow_private=True)


def _answers(*replies):
    """An HTTP stand-in that answers in order, recording what it was asked."""
    asked = []

    def request(method, url, **kw):
        asked.append((method, url, kw.get("redirect")))
        status, location = replies[len(asked) - 1]
        return MagicMock(status=status, headers={"location": location} if location else {})
    return MagicMock(request=MagicMock(side_effect=request)), asked


def _public_only(url, **_):
    host = url.split("/")[2]
    if host.startswith("169.254") or host == "localhost":
        raise ValueError(f"Refusing to reach non-public address ({host})")


@patch.object(_security, "assert_safe_url", _public_only)
class TestEveryHop(unittest.TestCase):

    def test_a_redirect_to_another_host_is_not_followed(self):
        http, asked = _answers((307, "https://elsewhere.example/collect"))
        with self.assertRaisesRegex(ValueError, "another host"):
            safe_request(http, "POST", "https://api.example.com/x", body=b"{}",
                         headers={"X-Api-Key": "secret"})
        self.assertEqual(len(asked), 1)

    def test_a_sessions_headers_are_left_off_on_another_host(self):
        session = MagicMock(headers={"Authorization": "Bearer s", "User-Agent": "ua"})
        session.get.side_effect = [MagicMock(status_code=302, headers={"location": "https://b.example/"}),
                                   MagicMock(status_code=200, headers={})]
        safe_get(session, "https://a.example/")
        self.assertNotIn("headers", session.get.call_args_list[0].kwargs)
        self.assertEqual(session.get.call_args_list[1].kwargs["headers"], {"Authorization": None})

    def test_a_redirect_to_the_metadata_service_is_refused(self):
        http, asked = _answers((302, "http://169.254.169.254/latest/meta-data/"))
        with self.assertRaisesRegex(ValueError, "non-public|another host"):
            safe_request(http, "GET", "https://example.com/")
        self.assertEqual(asked, [("GET", "https://example.com/", False)])   # never asked

    def test_a_public_redirect_is_followed_and_303_becomes_a_get(self):
        http, asked = _answers((303, "/done"), (200, None))
        self.assertEqual(safe_request(http, "POST", "https://example.com/x", body=b"{}").status, 200)
        self.assertEqual(asked[1], ("GET", "https://example.com/done", False))

    def test_a_loop_of_redirects_stops(self):
        http, _ = _answers(*[(302, "https://example.com/again")] * 10)
        with self.assertRaisesRegex(ValueError, "redirects"):
            safe_request(http, "GET", "https://example.com/")

    def test_a_requests_session_is_held_the_same_way(self):
        session = MagicMock()
        session.get.return_value = MagicMock(status_code=301, headers={"location": "http://localhost/admin"})
        with self.assertRaisesRegex(ValueError, "non-public"):
            safe_get(session, "https://example.com/")
        self.assertFalse(session.get.call_args.kwargs["allow_redirects"])


if __name__ == "__main__":
    unittest.main()
