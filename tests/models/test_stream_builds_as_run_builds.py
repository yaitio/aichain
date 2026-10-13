"""
A streamed request is built as a buffered one is.

The stream had its own copy of the request's tail: ``extra`` was dropped while
``effective_options`` said it was sent, and tools on a family without them
were not gated.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.models import Model

MSG = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
TOOL = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}]


class TestStreamBuildsAsRunBuilds(unittest.TestCase):

    def test_extra_reaches_a_streamed_request(self):
        m = Model("gpt-4o", api_key="k", options={"extra": {"service_tier": "flex"}})
        sent = {}

        def sse(path, body, headers, *a, **k):
            sent.update(body)
            yield {"choices": [{"delta": {"content": "ok"}}]}
        m.client._post_sse = MagicMock(side_effect=sse)
        m.client._auth_headers = MagicMock(return_value={})
        list(m.stream(MSG, {"format": {"type": "text"}}))
        self.assertEqual(sent.get("service_tier"), "flex")

    def test_tools_on_a_family_without_them_are_refused_on_both_paths(self):
        m = Model("flux-pro-1.1", api_key="k")
        out = {"modalities": ["image"], "format": {"type": "image"}}
        with self.assertRaisesRegex(ValueError, "tool calling"):
            m.build(MSG, out, TOOL)
        with self.assertRaisesRegex(ValueError, "tool calling"):
            list(m.stream(MSG, out, tools=TOOL))


if __name__ == "__main__":
    unittest.main()
