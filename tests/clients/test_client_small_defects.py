"""
Small client defects from the 4.0 audit.

Every Responses API conversation was stored at OpenAI by default though
nothing reads it back; `check_auth` answered False on a rate limit, an empty
balance and a missing endpoint; a 5xx body mentioning "quota" read as no
credit; any response with a `data` key read as an image.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.clients._errors import (InsufficientCreditsError, RateLimitError,
                                          ServerError, error_from_status)
from yait_aichain.models import Model

MSG = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]


class TestSmallDefects(unittest.TestCase):

    def test_the_responses_api_is_asked_not_to_store(self):
        _, body, _ = Model("gpt-5.5", api_key="k").build(MSG, {"format": {"type": "text"}})
        self.assertIs(body["store"], False)

    def test_extra_can_store(self):
        m = Model("gpt-5.5", api_key="k", options={"extra": {"store": True}})
        self.assertIs(m.build(MSG, {"format": {"type": "text"}})[1]["store"], True)

    def test_check_auth_raises_what_is_not_a_bad_key(self):
        client = Model("gpt-4o", api_key="k").client
        for exc in (RateLimitError(429, "slow down"), InsufficientCreditsError(402, "pay")):
            with self.subTest(type(exc).__name__), patch.object(type(client), "list_models",
                                                                side_effect=exc):
                with self.assertRaises(type(exc)):
                    client.check_auth()

    def test_a_server_error_is_not_billing(self):
        self.assertIsInstance(error_from_status(503, "quota service down"), ServerError)
        self.assertIsInstance(error_from_status(429, "insufficient quota"), InsufficientCreditsError)

    def test_a_data_key_is_not_an_image(self):
        r = {"data": {"note": "x"}, "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        self.assertEqual(Model("gpt-4o", api_key="k").from_response(r, {"format": {"type": "text"}}), "ok")


if __name__ == "__main__":
    unittest.main()
