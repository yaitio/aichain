"""
searchExa — Exa's search API behind the standard search interface.

Added 2026-10-02. What is pinned, offline:

  * the request is the documented one — ``POST /search`` with ``x-api-key`` —
    and highlights are asked for by default, so each result carries the
    passages that answer the query;
  * every standard option is translated to Exa's field, and ``recency``, which
    Exa has no word for, becomes a published-after date; an explicit start
    date and a window both bound the same side, and the later one wins;
  * ``language``, which Exa cannot honour, is not declared — asking for it is
    reported, never dropped;
  * the answer is the same numbered text every search tool returns;
  * an API error is raised by ``run()`` with its status and returned, not
    raised, by a call.

The response fixture is a real one, captured on 2026-10-02 and trimmed.
"""

import json
import os
import sys
import unittest
import warnings
from datetime import datetime, timezone
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.tools import searchExa

#: Captured live, 2026-10-02 — highlights trimmed, images dropped.
LIVE = {
    "requestId": "req-fixture",
    "resolvedSearchType": "",
    "results": [
        {"id": "https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms",
         "title": "Understanding and Coding the KV Cache in LLMs from Scratch",
         "url": "https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms",
         "publishedDate": "2025-06-17T10:55:34.000Z",
         "author": "Sebastian Raschka, PhD",
         "highlights": ["KV caches are one of the most critical techniques for efficient "
                        "inference in LLMs in production."]},
        {"id": "https://huggingface.co/blog/not-lain/kv-caching",
         "title": "KV Caching Explained: Optimizing Transformer Inference ...",
         "url": "https://huggingface.co/blog/not-lain/kv-caching",
         "publishedDate": "2025-01-30T00:00:00.000Z",
         "highlights": ["Published January 30, 2025 \n\n Introduction \n\nWhen AI models "
                        "generate text, they often repeat many of the same calculations."]},
    ],
    "searchTime": 8.4,
    "costDollars": {"total": 0.007, "search": {"neural": 0.007}},
}
QUERY = "KV cache in LLM inference explained"
NOW = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)


def _tool(response=LIVE, status=200):
    tool = searchExa(api_key="k")
    reply = MagicMock()
    reply.status = status
    reply.data = (json.dumps(response) if isinstance(response, dict) else response).encode()
    tool._http = MagicMock()
    tool._http.request.return_value = reply
    return tool


class TestTheRequest(unittest.TestCase):

    def test_the_documented_request_with_highlights_by_default(self):
        tool = _tool()
        tool.run(input=QUERY)
        method, url = tool._http.request.call_args.args
        kw = tool._http.request.call_args.kwargs
        self.assertEqual((method, url), ("POST", "https://api.exa.ai/search"))
        self.assertEqual(kw["headers"]["x-api-key"], "k")
        self.assertEqual(json.loads(kw["body"]),
                         {"query": QUERY, "type": "auto", "numResults": 10,
                          "contents": {"highlights": True}})

    def test_every_standard_option_reaches_its_field(self):
        body = searchExa._body(QUERY, {
            "max_results": 5, "country": "GB", "domains": ["arxiv.org"],
            "exclude_domains": ["pinterest.com"], "category": "news",
            "search_type": "fast", "end_date": "2026-09-01", "text_chars": 2000,
        })
        self.assertEqual(body["numResults"], 5)
        self.assertEqual(body["userLocation"], "GB")
        self.assertEqual(body["includeDomains"], ["arxiv.org"])
        self.assertEqual(body["excludeDomains"], ["pinterest.com"])
        self.assertEqual(body["category"], "news")
        self.assertEqual(body["type"], "fast")
        self.assertEqual(body["endPublishedDate"], "2026-09-01")
        self.assertEqual(body["contents"], {"highlights": True,
                                            "text": {"maxCharacters": 2000}})

    def test_the_count_is_kept_inside_what_the_api_takes(self):
        self.assertEqual(searchExa._body(QUERY, {"max_results": 500})["numResults"], 100)
        self.assertEqual(searchExa._body(QUERY, {"max_results": 0})["numResults"], 1)

    def test_recency_becomes_a_published_after_date(self):
        body = searchExa._body(QUERY, {"recency": "week"}, now=NOW)
        self.assertEqual(body["startPublishedDate"], "2026-09-25T12:00:00.000Z")

    def test_a_window_and_an_explicit_start_keep_the_later_bound(self):
        # Both promises hold only at the later of the two dates.
        later = searchExa._body(QUERY, {"recency": "year",
                                        "start_date": "2026-06-01T00:00:00.000Z"}, now=NOW)
        self.assertEqual(later["startPublishedDate"], "2026-06-01T00:00:00.000Z")
        tighter = searchExa._body(QUERY, {"recency": "day",
                                          "start_date": "2026-06-01T00:00:00.000Z"}, now=NOW)
        self.assertEqual(tighter["startPublishedDate"], "2026-10-01T12:00:00.000Z")

    def test_an_unknown_recency_is_refused(self):
        with self.assertRaises(ValueError):
            searchExa._body(QUERY, {"recency": "fortnight"})

    def test_language_has_no_field_and_is_reported_not_dropped(self):
        tool = _tool()
        self.assertIn("options.language", tool.unknown_keys({"input": QUERY,
                                                             "options": {"language": "en"}}))
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            tool(input=QUERY, options={"language": "en"})
        self.assertTrue(any("options.language" in str(w.message) for w in seen))


class TestTheAnswer(unittest.TestCase):

    def test_the_numbered_list_every_search_tool_returns(self):
        text = _tool().run(input=QUERY)
        self.assertTrue(text.startswith(f'Search results for "{QUERY}" (2 results):'))
        self.assertIn("[1] Understanding and Coding the KV Cache in LLMs from Scratch", text)
        self.assertIn("    URL: https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms", text)
        self.assertIn("    Published: 2025-06-17 · Author: Sebastian Raschka, PhD", text)
        # Highlights are folded onto one line each — the page's own line breaks
        # would otherwise break the list the model reads.
        self.assertIn("    · Published January 30, 2025 Introduction When AI models", text)

    def test_no_results_is_said(self):
        self.assertIn("(0 results)", _tool({"results": []}).run(input=QUERY))

    def test_an_api_error_is_raised_by_run_with_its_status(self):
        tool = _tool({"error": "Invalid API key"}, status=401)
        with self.assertRaises(RuntimeError) as ctx:
            tool.run(input=QUERY)
        self.assertIn("[401]", str(ctx.exception))

    def test_and_returned_not_raised_by_a_call(self):
        result = _tool({"error": "out of credits"}, status=402)(input=QUERY)
        self.assertFalse(result.success)
        self.assertIn("[402]", result.error)

    def test_no_key_is_refused_at_construction(self):
        saved = os.environ.pop("EXA_API_KEY", None)
        try:
            with self.assertRaises(ValueError) as ctx:
                searchExa()
            self.assertIn("EXA_API_KEY", str(ctx.exception))
        finally:
            if saved is not None:
                os.environ["EXA_API_KEY"] = saved


class TestSearchExaLive(unittest.TestCase):

    def setUp(self):
        if os.environ.get("EXA_API_KEY", "") in ("", "test-key", "k"):
            self.skipTest("EXA_API_KEY not set to a real key")

    def test_a_real_search_returns_results_with_highlights(self):
        text = searchExa().run(input=QUERY, options={"max_results": 3})
        self.assertIn("(3 results)", text)
        self.assertIn("URL: https://", text)
        self.assertIn("    · ", text)


if __name__ == "__main__":
    unittest.main()
