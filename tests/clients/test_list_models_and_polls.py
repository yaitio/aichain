"""
`list_models` reads every page, and one poll cannot outlast the deadline.

Anthropic's first page was twenty models and Google's fifty, and the rest were
never read; the poll deadline bounded the loop, not the status request inside
it, which could wait the client's whole read timeout.
"""

import json
import os
import sys
import time
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.models import Model


class TestPages(unittest.TestCase):

    def test_anthropic_follows_has_more(self):
        client = Model("claude-sonnet-4-5", api_key="k").client
        pages = [{"data": [{"id": "a"}], "has_more": True, "last_id": "a"},
                 {"data": [{"id": "b"}], "has_more": False, "last_id": "b"}]
        client._get = MagicMock(side_effect=[json.dumps(p).encode() for p in pages])
        self.assertEqual(client.list_models(), ["a", "b"])
        self.assertIn("after_id=a", client._get.call_args_list[1].args[0])

    def test_google_follows_the_page_token(self):
        client = Model("gemini-2.5-flash", api_key="k").client
        pages = [{"models": [{"name": "models/x"}], "nextPageToken": "t2"},
                 {"models": [{"name": "models/y"}]}]
        client._get = MagicMock(side_effect=[json.dumps(p).encode() for p in pages])
        self.assertEqual(client.list_models(), ["x", "y"])
        self.assertIn("pageToken=t2", client._get.call_args_list[1].args[0])


class TestPollTimeout(unittest.TestCase):

    def test_a_poll_is_bounded_by_the_deadline(self):
        client = Model("gpt-4o", api_key="k").client
        t = client._poll_timeout(time.monotonic() + 5)
        self.assertLessEqual(t.read_timeout, 5.0)
        self.assertLessEqual(client._poll_timeout(time.monotonic() + 600).read_timeout, 30.0)


if __name__ == "__main__":
    unittest.main()
