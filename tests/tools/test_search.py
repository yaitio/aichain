"""
Search tools — one vocabulary of options, one record of results.

Rebuilt 2026-10-08 to the standard the decision tools set the same day. Before
it, the base declared five standard options and only one provider of five
honoured all of them — Brave and SerpAPI had neither ``domains`` nor
``recency``, OpenAI's had no ``max_results``, Exa no ``language`` — and the
results were text alone, each provider's in its own layout. ``searchOpenAI``
answered with a synthesised paragraph rather than results, a different
contract, and was removed.

What is pinned, offline, for every provider:

  * every standard option reaches the provider's own field, or ``site:`` in
    the query where it has none;
  * an option a provider cannot honour, or honours differently, is reported —
    a warning, a line the model reads, and ``notices`` — never dropped;
  * the results are the same record from every provider, and ``run()`` is the
    same numbered text built from it;
  * risk is ``read``; an API error is raised by ``run()`` and returned by a call.

Fixtures: Brave's and Perplexity's are real answers captured 2026-10-06 and
trimmed; Exa's real, 2026-10-02; SerpAPI's is built from its documented
``organic_results`` shape — no SerpAPI key was available to capture one.
"""

import json
import os
import sys
import unittest
import urllib.parse
import warnings
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.tools import Search, searchBrave, searchExa, searchPerplexity, searchSerp
from yait_aichain.tools.search._base import day

#: Captured live 2026-10-06, query "site:hbr.org article", trimmed.
BRAVE = {"web": {"results": [
    {"title": "AI Is Revolutionizing Strategic Decision-Making",
     "url": "https://hbr.org/2026/09/ai-is-revolutionizing-strategic-decision-making",
     "description": "A version of this <strong>article</strong> appeared in the September–October 2026 issue.",
     "page_age": "2026-09-30T17:39:42",
     "article": {"author": [], "date": "Sep 01, 2026", "isAccessibleForFree": False},
     "extra_snippets": ["For decades, the limits of time and brain capacity meant that teams could "
                        "consider only so many strategic options."]},
    {"title": "Harvard Business Review - The Latest", "url": "https://hbr.org/the-latest",
     "description": "Advice from the world&#x27;s best business and management experts.",
     "page_age": "2026-09-30T14:39:42"}]}}

#: Captured live 2026-10-06, query "article Harvard Business Review", trimmed.
PERPLEXITY = {"results": [
    {"title": "AI Is Making Verification the Bottleneck for Companies",
     "url": "https://hbr.org/2026/10/ai-is-making-verification-the-bottleneck-for-companies",
     "date": "2026-10-02", "last_updated": "2026-10-03",
     "snippet": "AI Is Making Verification the Bottleneck for Companies\nby Christian Catalini\nOctober 2, 2026"},
    {"title": "The Mindsets Leaders Need as AI Accelerates the Pace of Business",
     "url": "https://hbr.org/2026/10/the-mindsets-leaders-need-as-ai-accelerates-the-pace-of-business",
     "date": "2026-10-02", "snippet": "by Jacqueline Carter, Robert Hoppe and Paula Kelley"}]}

#: Captured live 2026-10-02, trimmed.
EXA = {"results": [
    {"title": "Understanding and Coding the KV Cache in LLMs from Scratch",
     "url": "https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms",
     "publishedDate": "2025-06-17T10:55:34.000Z", "author": "Sebastian Raschka, PhD",
     "highlights": ["KV caches are one of the most critical techniques for efficient inference."]},
    {"title": "KV Caching Explained", "url": "https://huggingface.co/blog/not-lain/kv-caching",
     "publishedDate": "2025-01-30T00:00:00.000Z",
     "highlights": ["Published January 30, 2025 \n\n Introduction \n\nWhen AI models generate text."]}],
    "costDollars": {"total": 0.007}}

#: SerpAPI's documented organic_results shape (no key to capture a real one).
SERP = {"organic_results": [
    {"position": 1, "title": "Python asyncio — official docs", "link": "https://docs.python.org/3/library/asyncio.html",
     "snippet": "asyncio is a library to write concurrent code.", "date": "Sep 30, 2026", "source": "Python"},
    {"position": 2, "title": "Asyncio tutorial", "link": "https://realpython.com/async-io-python/",
     "snippet": "A walkthrough of async IO.", "date": "3 days ago"}]}

FIXTURE = {searchBrave: BRAVE, searchPerplexity: PERPLEXITY, searchExa: EXA, searchSerp: SERP}
QUERY = "AI and strategy"


def _tool(cls, response=None, status=200):
    tool = cls(api_key="k")
    reply = MagicMock()
    reply.status = status
    response = FIXTURE[cls] if response is None else response
    reply.data = (json.dumps(response) if isinstance(response, dict) else response).encode()
    tool._http = MagicMock()
    tool._http.request.return_value = reply
    return tool


def _sent(tool):
    """What went on the wire: a JSON body, or SerpAPI's query string."""
    args, kw = tool._http.request.call_args
    if args[0] == "GET":
        return dict(urllib.parse.parse_qsl(urllib.parse.urlparse(args[1]).query))
    return json.loads(kw["body"])


def _search(cls, options=None, response=None):
    tool = _tool(cls, response)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        found = tool.search(QUERY, options)
    return tool, found, [str(w.message) for w in seen]


TODAY = datetime.now(timezone.utc).date()


class TestEveryStandardOptionReachesTheWire(unittest.TestCase):

    def test_max_results(self):
        for cls, field in ((searchBrave, "count"), (searchPerplexity, "max_results"),
                           (searchExa, "numResults"), (searchSerp, "num")):
            with self.subTest(provider=cls.__name__):
                tool, _, _ = _search(cls, {"max_results": 7})
                self.assertEqual(str(_sent(tool)[field]), "7")

    def test_recency_in_each_providers_words(self):
        week_ago = (datetime.now(timezone.utc) - timedelta(weeks=1)).date().isoformat()
        tool, _, _ = _search(searchBrave, {"recency": "week"}); self.assertEqual(_sent(tool)["freshness"], "pw")
        tool, _, _ = _search(searchPerplexity, {"recency": "week"}); self.assertEqual(_sent(tool)["search_recency_filter"], "week")
        tool, _, _ = _search(searchSerp, {"recency": "week"}); self.assertEqual(_sent(tool)["tbs"], "qdr:w")
        tool, _, _ = _search(searchExa, {"recency": "week"})
        self.assertEqual(_sent(tool)["startPublishedDate"], f"{week_ago}T00:00:00.000Z")

    def test_a_date_window_in_each_providers_words(self):
        w = {"start_date": "2026-09-28", "end_date": "2026-10-04"}
        tool, _, _ = _search(searchBrave, w); self.assertEqual(_sent(tool)["freshness"], "2026-09-28to2026-10-04")
        tool, _, _ = _search(searchPerplexity, w)
        self.assertEqual((_sent(tool)["search_after_date_filter"], _sent(tool)["search_before_date_filter"]),
                         ("09/28/2026", "10/04/2026"))
        tool, _, _ = _search(searchSerp, w); self.assertEqual(_sent(tool)["tbs"], "cdr:1,cd_min:9/28/2026,cd_max:10/4/2026")
        tool, _, _ = _search(searchExa, w)
        self.assertEqual((_sent(tool)["startPublishedDate"], _sent(tool)["endPublishedDate"]),
                         ("2026-09-28T00:00:00.000Z", "2026-10-04T23:59:59.999Z"))

    def test_an_explicit_start_tighter_than_recency_wins(self):
        start = (TODAY - timedelta(days=2)).isoformat()
        tool, _, _ = _search(searchPerplexity, {"recency": "month", "start_date": start})
        sent = _sent(tool)
        self.assertNotIn("search_recency_filter", sent)
        self.assertEqual(sent["search_after_date_filter"], "/".join([start[5:7], start[8:10], start[:4]]))

    def test_domains_by_field_or_by_site_in_the_query(self):
        d = {"domains": ["hbr.org", "sloanreview.mit.edu"]}
        tool, _, _ = _search(searchPerplexity, d); self.assertEqual(_sent(tool)["search_domain_filter"], d["domains"])
        tool, _, _ = _search(searchExa, d); self.assertEqual(_sent(tool)["includeDomains"], d["domains"])
        for cls, field in ((searchBrave, "q"), (searchSerp, "q")):
            with self.subTest(provider=cls.__name__):
                tool, _, _ = _search(cls, d)
                self.assertEqual(_sent(tool)[field], f"{QUERY} (site:hbr.org OR site:sloanreview.mit.edu)")

    def test_country_and_language(self):
        o = {"country": "GB", "language": "en"}
        tool, _, _ = _search(searchBrave, o); self.assertEqual((_sent(tool)["country"], _sent(tool)["search_lang"]), ("GB", "en"))
        tool, _, _ = _search(searchPerplexity, o)
        self.assertEqual((_sent(tool)["country"], _sent(tool)["search_language_filter"]), ("GB", ["en"]))
        tool, _, _ = _search(searchSerp, o); self.assertEqual((_sent(tool)["gl"], _sent(tool)["hl"]), ("gb", "en"))
        tool, _, notes = _search(searchExa, o)
        self.assertEqual(_sent(tool)["userLocation"], "GB")
        self.assertNotIn("language", json.dumps(_sent(tool)))

    def test_a_malformed_date_or_window_is_refused_before_the_network(self):
        for opts in ({"start_date": "28.09.2026"}, {"recency": "fortnight"},
                     {"start_date": "2026-10-04", "end_date": "2026-09-28"}):
            with self.subTest(opts=opts):
                tool = _tool(searchExa)
                with self.assertRaises(ValueError):
                    tool.search(QUERY, opts)
                tool._http.request.assert_not_called()


class TestNothingIsDroppedInSilence(unittest.TestCase):

    def test_exa_declines_language_and_says_so_three_ways(self):
        tool, found, warned = _search(searchExa, {"language": "en"})
        self.assertTrue(any("language" in n and "Exa has no language filter" in n for n in found["notices"]))
        self.assertTrue(any("language" in w for w in warned))
        self.assertIn("Note: language is not supported by searchExa", tool.format(found))
        # declared, so a model can see it is not applied rather than be refused
        self.assertIn("Not applied here", searchExa.parameters["properties"]["options"]["properties"]["language"]["description"])

    def test_brave_widens_an_hour_and_says_so(self):
        tool, found, _ = _search(searchBrave, {"recency": "hour"})
        self.assertEqual(_sent(tool)["freshness"], "pd")
        self.assertTrue(any("widened to a day" in n for n in found["notices"]))

    def test_serp_on_another_engine_says_which_filters_it_did_not_apply(self):
        tool, found, _ = _search(searchSerp, {"engine": "duckduckgo", "recency": "week", "country": "GB"})
        sent = _sent(tool)
        self.assertNotIn("tbs", sent); self.assertNotIn("gl", sent)
        self.assertEqual(sum("was not applied" in n for n in found["notices"]), 2)

    def test_a_count_above_the_ceiling_is_capped_and_said(self):
        tool, found, _ = _search(searchBrave, {"max_results": 50})
        self.assertEqual(_sent(tool)["count"], 20)
        self.assertTrue(any("ceiling" in n for n in found["notices"]))

    def test_perplexity_refuses_more_domains_than_it_filters(self):
        with self.assertRaises(ValueError):
            _tool(searchPerplexity).search(QUERY, {"domains": [f"d{i}.com" for i in range(21)]})

    def test_a_typo_is_still_caught_by_the_schema(self):
        self.assertIn("options.recencyy", searchBrave(api_key="k").unknown_keys(
            {"input": QUERY, "options": {"recencyy": "day"}}))


class TestOneRecord(unittest.TestCase):

    KEYS = {"title", "url", "date", "author", "snippets", "text"}

    def test_every_provider_returns_the_same_keys(self):
        for cls in FIXTURE:
            with self.subTest(provider=cls.__name__):
                _, found, _ = _search(cls)
                self.assertEqual(set(found), {"query", "provider", "results", "notices"})
                self.assertEqual(len(found["results"]), 2)
                for r in found["results"]:
                    self.assertEqual(set(r), self.KEYS)
                    self.assertTrue(r["url"].startswith("https://"))
                    self.assertTrue(r["date"] == "" or len(r["date"]) == 10)

    def test_brave_takes_the_articles_own_date_over_when_it_was_seen(self):
        _, found, _ = _search(searchBrave)
        first, second = found["results"]
        self.assertEqual(first["date"], "2026-09-01")      # article.date, not page_age 09-30
        self.assertEqual(second["date"], "2026-09-30")     # no article: page_age
        self.assertEqual(first["snippets"][0], "A version of this article appeared in the September–October 2026 issue.")
        self.assertIn("world's best", second["snippets"][0])

    def test_dates_are_normalised_from_every_spelling(self):
        self.assertEqual(day("2026-09-30T17:39:42"), "2026-09-30")
        self.assertEqual(day("Sep 30, 2026"), "2026-09-30")
        self.assertEqual(day("30 September 2026"), "2026-09-30")
        self.assertEqual(day("3 days ago"), (TODAY - timedelta(days=3)).isoformat())
        self.assertEqual(day("last week"), "")
        _, found, _ = _search(searchSerp)
        self.assertEqual([r["date"] for r in found["results"]], ["2026-09-30", (TODAY - timedelta(days=3)).isoformat()])

    def test_exa_carries_author_highlights_and_text_on_request(self):
        resp = json.loads(json.dumps(EXA)); resp["results"][0]["text"] = "Full page text."
        tool, found, _ = _search(searchExa, {"text_chars": 2000}, resp)
        r = found["results"][0]
        self.assertEqual(r["author"], "Sebastian Raschka, PhD")
        self.assertEqual(r["text"], "Full page text.")
        self.assertEqual(_sent(tool)["contents"], {"highlights": {"maxCharacters": 1000}, "text": {"maxCharacters": 2000}})


class TestTheTextAModelReads(unittest.TestCase):

    def test_one_layout_built_from_the_record(self):
        tool, found, _ = _search(searchExa)
        text = tool.format(found)
        self.assertTrue(text.startswith(f'Search results for "{QUERY}" (2 results):'))
        self.assertIn("[1] Understanding and Coding the KV Cache in LLMs from Scratch", text)
        self.assertIn("    URL: https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms", text)
        self.assertIn("    Published: 2025-06-17 · Author: Sebastian Raschka, PhD", text)
        # folded onto one line each — the page's own breaks would break the list
        self.assertIn("    · Published January 30, 2025 Introduction When AI models generate text.", text)

    def test_run_returns_exactly_that_text(self):
        for cls in FIXTURE:
            with self.subTest(provider=cls.__name__):
                self.assertEqual(_tool(cls).run(input=QUERY), cls.format(_tool(cls).search(QUERY)))

    def test_no_results_is_said(self):
        self.assertIn("(0 results)", _tool(searchExa, {"results": []}).run(input=QUERY))


class TestContract(unittest.TestCase):

    def test_read_risk_a_shared_base_and_a_named_key(self):
        for cls, var in ((searchBrave, "BRAVE_SEARCH_API_KEY"), (searchPerplexity, "PERPLEXITY_API_KEY"),
                         (searchExa, "EXA_API_KEY"), (searchSerp, "SERPAPI_API_KEY")):
            with self.subTest(provider=cls.__name__):
                self.assertTrue(issubclass(cls, Search))
                self.assertEqual(cls.risk, "read")
                saved = os.environ.pop(var, None)
                try:
                    with self.assertRaises(ValueError) as ctx:
                        cls()
                    self.assertIn(var, str(ctx.exception))
                finally:
                    if saved is not None:
                        os.environ[var] = saved

    def test_an_api_error_is_raised_by_run_and_returned_by_a_call(self):
        for cls in FIXTURE:
            with self.subTest(provider=cls.__name__):
                with self.assertRaises(RuntimeError) as ctx:
                    _tool(cls, {"error": "Invalid API key"}, status=401).run(input=QUERY)
                self.assertIn("[401]", str(ctx.exception))
                result = _tool(cls, {"error": "out of credits"}, status=402)(input=QUERY)
                self.assertFalse(result.success)
                self.assertIn("[402]", result.error)

    def test_an_empty_query_is_refused(self):
        with self.assertRaises(ValueError):
            _tool(searchBrave).search("  ")

    def test_serp_with_no_results_is_an_empty_list_not_an_error(self):
        _, found, _ = _search(searchSerp, response={"error": "Google hasn't returned any results for this query."})
        self.assertEqual(found["results"], [])


def _real(var):
    return os.environ.get(var, "") not in ("", "test-key", "k")


class TestSearchLive(unittest.TestCase):
    """The same query and window to every provider with a key: the same record back."""

    OPTS = {"max_results": 3, "start_date": (TODAY - timedelta(days=30)).isoformat(), "domains": ["hbr.org"]}

    def _check(self, cls, var):
        if not _real(var):
            self.skipTest(f"{var} not set to a real key")
        found = cls().search("leadership and AI", self.OPTS)
        self.assertTrue(found["results"])
        for r in found["results"]:
            self.assertEqual(set(r), TestOneRecord.KEYS)
            self.assertIn("hbr.org", r["url"])

    def test_brave(self): self._check(searchBrave, "BRAVE_SEARCH_API_KEY")
    def test_perplexity(self): self._check(searchPerplexity, "PERPLEXITY_API_KEY")
    def test_exa(self): self._check(searchExa, "EXA_API_KEY")
    def test_serp(self): self._check(searchSerp, "SERPAPI_API_KEY")


if __name__ == "__main__":
    unittest.main()
