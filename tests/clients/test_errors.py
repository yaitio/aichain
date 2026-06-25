"""
Tests for error classification — especially out-of-credits vs auth/rate-limit.

Out-of-credits is a billing problem, not a credentials problem, and providers
signal it under several HTTP codes; it must surface as InsufficientCreditsError
so callers top up instead of re-checking the key, and it must never be retried.
"""

import unittest

from yait_aichain.clients._errors import (
    error_from_status,
    APIError, NetworkError, RateLimitError, AuthenticationError,
    InsufficientCreditsError, InvalidRequestError, NotFoundError, ServerError,
)


class TestBillingDetection(unittest.TestCase):

    def test_402_is_insufficient_credits(self):
        e = error_from_status(402, '{"error_code":"PARTNER_API_BUDGET_EXHAUSTED"}')
        self.assertIsInstance(e, InsufficientCreditsError)

    def test_xai_403_credits_not_auth(self):
        # Real xAI body: 403 permission-denied "used all available credits".
        e = error_from_status(
            403, '{"code":"permission-denied","error":"...used all available credits..."}')
        self.assertIsInstance(e, InsufficientCreditsError)
        self.assertNotIsInstance(e, AuthenticationError)

    def test_openai_429_insufficient_quota_not_rate_limit(self):
        e = error_from_status(429, '{"error":{"code":"insufficient_quota"}}')
        self.assertIsInstance(e, InsufficientCreditsError)
        self.assertNotIsInstance(e, RateLimitError)

    def test_anthropic_400_credit_balance_not_invalid_request(self):
        e = error_from_status(400, "Your credit balance is too low to access the API.")
        self.assertIsInstance(e, InsufficientCreditsError)
        self.assertNotIsInstance(e, InvalidRequestError)

    def test_openai_billing_hard_limit(self):
        e = error_from_status(400, "Billing hard limit has been reached")
        self.assertIsInstance(e, InsufficientCreditsError)

    def test_insufficient_credits_is_api_error(self):
        e = error_from_status(402, "out of credits")
        self.assertIsInstance(e, APIError)        # still catchable as APIError


class TestNoFalsePositives(unittest.TestCase):

    def test_plain_403_is_auth(self):
        e = error_from_status(403, '{"error":"invalid api key"}')
        self.assertIsInstance(e, AuthenticationError)
        self.assertNotIsInstance(e, InsufficientCreditsError)

    def test_plain_401_is_auth(self):
        self.assertIsInstance(error_from_status(401, "unauthorized"), AuthenticationError)

    def test_plain_429_is_rate_limit(self):
        e = error_from_status(429, '{"error":{"code":"rate_limit_exceeded"}}')
        self.assertIsInstance(e, RateLimitError)
        self.assertNotIsInstance(e, InsufficientCreditsError)

    def test_plain_400_is_invalid_request(self):
        e = error_from_status(400, "missing required parameter: prompt")
        self.assertIsInstance(e, InvalidRequestError)
        self.assertNotIsInstance(e, InsufficientCreditsError)

    def test_404_and_5xx_unchanged(self):
        self.assertIsInstance(error_from_status(404, "no such model"), NotFoundError)
        self.assertIsInstance(error_from_status(503, "service unavailable"), ServerError)
        self.assertIsInstance(error_from_status(0, "connection refused"), NetworkError)


class TestRetrySafety(unittest.TestCase):

    def test_billing_429_not_retried_by_skill(self):
        # A 429 billing error carries status 429 (transient set) but must NOT be
        # retried — the skill guard excludes InsufficientCreditsError.
        import json
        from unittest.mock import MagicMock
        from yait_aichain.models import Model
        from yait_aichain.skills import Skill

        m = Model("gpt-4o", api_key="k")
        calls = {"n": 0}

        def _raise(*a, **k):
            calls["n"] += 1
            raise error_from_status(429, '{"error":{"code":"insufficient_quota"}}')

        m.client._post = MagicMock(side_effect=_raise)
        sk = Skill(model=m, input={"messages": [{"role": "user", "parts": ["hi"]}]},
                   max_retries=3, retry_delay=0)
        with self.assertRaises(InsufficientCreditsError):
            sk.run()
        self.assertEqual(calls["n"], 1)        # exactly one attempt, no retries


if __name__ == "__main__":
    unittest.main()
