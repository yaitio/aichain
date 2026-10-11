"""
An empty balance answered with 429 is not retried.

Z.ai answers 429 both for a rate limit and for an account with no money. The
transport retried both, five times with backoff, before the library read the
body — a minute a call, measured live on 2026-10-11, for an answer that was
never going to change. The retry policy reads a 429's body first now, and a
billing body ends the retries; a plain rate limit is retried as before, and
the body is still there for the error the caller gets.
"""

import io
import unittest
from unittest.mock import MagicMock

import urllib3
from urllib3.exceptions import MaxRetryError

from yait_aichain.clients._constants import DEFAULT_IDEMPOTENT_RETRIES, DEFAULT_RETRIES, Retry
from yait_aichain.clients._errors import InsufficientCreditsError, RateLimitError, error_from_status

ZAI = b'{"error":{"code":"1113","message":"Insufficient balance or no resource package. Please recharge."}}'
LIMIT = b'{"error":{"message":"Rate limit reached for requests"}}'


def _response(status, body, preload=False):
    return urllib3.HTTPResponse(body=io.BytesIO(body), status=status, preload_content=preload,
                                headers={"Content-Type": "application/json"})


def _pool(*responses):
    """A connection pool whose server answers with *responses* in turn."""
    pool = urllib3.HTTPConnectionPool("example.invalid")
    queue = list(responses)
    pool._make_request = MagicMock(side_effect=lambda *a, **k: queue.pop(0))
    pool._get_conn = MagicMock(return_value=MagicMock())
    pool._put_conn = MagicMock()
    return pool


class TestTheRetryPolicy(unittest.TestCase):

    def test_a_billing_429_ends_the_retries(self):
        r = _response(429, ZAI)
        with self.assertRaises(MaxRetryError):
            DEFAULT_RETRIES.increment("POST", "/chat", response=r)
        self.assertEqual(r.data, ZAI)                    # still there for the error

    def test_a_rate_limit_is_retried(self):
        nxt = DEFAULT_RETRIES.increment("POST", "/chat", response=_response(429, LIMIT))
        self.assertIsInstance(nxt, Retry)
        self.assertEqual(nxt.total, DEFAULT_RETRIES.total - 1)

    def test_other_statuses_are_left_to_urllib3(self):
        nxt = DEFAULT_IDEMPOTENT_RETRIES.increment("GET", "/models", response=_response(503, ZAI))
        self.assertEqual(nxt.total, DEFAULT_IDEMPOTENT_RETRIES.total - 1)

    def test_a_copy_keeps_the_policy(self):
        self.assertIsInstance(DEFAULT_RETRIES.new(status_forcelist={429, 529}), Retry)


class TestThroughAPool(unittest.TestCase):
    """The same, through urllib3's own retry loop rather than by calling it."""

    def _run(self, *responses, preload=True):
        pool = _pool(*responses)
        retries = DEFAULT_RETRIES.new(backoff_factor=0)
        out = pool.urlopen("POST", "/chat", body=b"{}", retries=retries, preload_content=preload)
        return out, pool._make_request.call_count

    def test_billing_is_asked_once_and_reads_as_billing(self):
        for preload in (True, False):                    # a call, and a stream
            with self.subTest(preload=preload):
                out, calls = self._run(_response(429, ZAI), _response(200, b"{}"), preload=preload)
                self.assertEqual(calls, 1)
                err = error_from_status(out.status, out.data.decode(), out.headers)
                self.assertIsInstance(err, InsufficientCreditsError)

    def test_a_rate_limit_is_asked_again(self):
        out, calls = self._run(_response(429, LIMIT), _response(200, b'{"ok": 1}'))
        self.assertEqual((calls, out.status, out.data), (2, 200, b'{"ok": 1}'))
        self.assertIsInstance(error_from_status(429, LIMIT.decode(), {}), RateLimitError)


if __name__ == "__main__":
    unittest.main()
