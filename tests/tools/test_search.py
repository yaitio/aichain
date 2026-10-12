"""
Search tools — one vocabulary of options, one record of results.

Rebuilt 2026-10-08 to the standard the decision tools set the same day. Before
it, the base declared five standard options and only one provider of five
honoured all of them — Brave had neither ``domains`` nor
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
trimmed; Exa's real, 2026-10-02; Firecrawl's in its documented shape,
and Tavily's in their documented shapes, checked against live answers on
2026-10-08, as is Parallel's.
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

from yait_aichain.tools import (Record, Search, search, searchBrave, searchExa, searchFirecrawl,
                                searchPerplexity, searchParallel, searchTavily)
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

#: Firecrawl's v2 response shape, as documented and as answered live on 2026-10-08:
#: web results carry no date; a news result does; a scraped page has markdown.
FIRECRAWL = {"success": True, "creditsUsed": 2, "data": {
    "web": [{"title": "Firecrawl — turn websites into LLM-ready data", "url": "https://www.firecrawl.dev/",
             "description": "Scrape, crawl and search the web for AI.",
             "markdown": "# Firecrawl\n\nTurn websites into LLM-ready data.",
             "metadata": {"title": "Firecrawl", "sourceURL": "https://www.firecrawl.dev/", "statusCode": 200}}],
    "news": [{"title": "Firecrawl raises a round", "url": "https://news.example.com/firecrawl",
              "snippet": "The scraping company announced funding.", "date": "Sep 30, 2026"}]}}

#: Tavily's documented example answer, two results; the live shape matched it on 2026-10-08.
TAVILY = {"query": "Who is Leo Messi?", "images": [], "response_time": "1.67", "usage": {"credits": 1},
          "results": [
              {"title": "Lionel Messi Facts | Britannica", "url": "https://www.britannica.com/facts/Lionel-Messi",
               "content": "Lionel Messi, an Argentine footballer [...] Born in 1987, Messi spent his career at Barcelona",
               "score": 0.81, "raw_content": "# Lionel Messi\n\nArgentine footballer.",
               "published_date": "Tue, 11 Mar 2025 17:00:00 GMT"},
              {"title": "Lionel Messi — Wikipedia", "url": "https://en.wikipedia.org/wiki/Lionel_Messi",
               "content": "Lionel Andrés Messi is an Argentine professional footballer.", "score": 0.7,
               "raw_content": None, "published_date": None}]}

#: Parallel's documented answer shape, two results; the live shape matched it on 2026-10-08.
PARALLEL = {"search_id": "search_fcb2b4f3c75e418687bccaa1a8381331",
            "session_id": "session_fcb2b4f3c75e418687bccaa1a8381331",
            "results": [{"url": "https://www.example.com", "title": "Sample webpage title",
                         "publish_date": "2024-01-15", "excerpts": ["Sample excerpt 1", "Sample excerpt 2"]},
                        {"url": "https://www.example.org/two", "title": "Second page", "publish_date": None,
                         "excerpts": ["Only excerpt"]}]}

FIXTURE = {searchBrave: BRAVE, searchPerplexity: PERPLEXITY, searchExa: EXA,
           searchFirecrawl: FIRECRAWL, searchTavily: TAVILY, searchParallel: PARALLEL}
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
    """What went on the wire: the JSON body."""
    args, kw = tool._http.request.call_args
    if args[0] == "GET":
        return dict(urllib.parse.parse_qsl(urllib.parse.urlparse(args[1]).query))
    return json.loads(kw["body"])


def _search(cls, options=None, response=None):
    tool = _tool(cls, response)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        found = tool.run(QUERY, options)
    return tool, found, [str(w.message) for w in seen]


TODAY = datetime.now(timezone.utc).date()


class TestEveryStandardOptionReachesTheWire(unittest.TestCase):

    def test_max_results(self):
        for cls, field in ((searchBrave, "count"), (searchPerplexity, "max_results"),
                           (searchExa, "numResults"), (searchFirecrawl, "limit"),
                           (searchTavily, "max_results")):
            with self.subTest(provider=cls.__name__):
                tool, _, _ = _search(cls, {"max_results": 7})
                self.assertEqual(str(_sent(tool)[field]), "7")

    def test_recency_in_each_providers_words(self):
        week_ago = (datetime.now(timezone.utc) - timedelta(weeks=1)).date().isoformat()
        tool, _, _ = _search(searchBrave, {"recency": "week"}); self.assertEqual(_sent(tool)["freshness"], "pw")
        tool, _, _ = _search(searchPerplexity, {"recency": "week"}); self.assertEqual(_sent(tool)["search_recency_filter"], "week")
        tool, _, _ = _search(searchFirecrawl, {"recency": "hour"}); self.assertEqual(_sent(tool)["tbs"], "qdr:h")
        tool, _, _ = _search(searchTavily, {"recency": "week"}); self.assertEqual(_sent(tool)["time_range"], "week")
        tool, _, _ = _search(searchExa, {"recency": "week"})
        self.assertEqual(_sent(tool)["startPublishedDate"], f"{week_ago}T00:00:00.000Z")

    def test_a_date_window_in_each_providers_words(self):
        w = {"start_date": "2026-09-28", "end_date": "2026-10-04"}
        tool, _, _ = _search(searchBrave, w); self.assertEqual(_sent(tool)["freshness"], "2026-09-28to2026-10-04")
        tool, _, _ = _search(searchPerplexity, w)
        self.assertEqual((_sent(tool)["search_after_date_filter"], _sent(tool)["search_before_date_filter"]),
                         ("09/28/2026", "10/04/2026"))
        tool, _, _ = _search(searchFirecrawl, w)
        self.assertEqual(_sent(tool)["tbs"], "cdr:1,cd_min:09/28/2026,cd_max:10/04/2026")
        tool, _, _ = _search(searchTavily, w)
        self.assertEqual((_sent(tool)["start_date"], _sent(tool)["end_date"]), ("2026-09-28", "2026-10-04"))
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
        tool, _, _ = _search(searchFirecrawl, d); self.assertEqual(_sent(tool)["includeDomains"], d["domains"])
        tool, _, _ = _search(searchTavily, d); self.assertEqual(_sent(tool)["include_domains"], d["domains"])
        tool, _, _ = _search(searchBrave, d)
        self.assertEqual(_sent(tool)["q"], f"{QUERY} (site:hbr.org OR site:sloanreview.mit.edu)")

    def test_country_and_language(self):
        o = {"country": "GB", "language": "en"}
        tool, _, _ = _search(searchBrave, o); self.assertEqual((_sent(tool)["country"], _sent(tool)["search_lang"]), ("GB", "en"))
        tool, _, _ = _search(searchPerplexity, o)
        self.assertEqual((_sent(tool)["country"], _sent(tool)["search_language_filter"]), ("GB", ["en"]))
        tool, _, notes = _search(searchExa, o)
        self.assertEqual(_sent(tool)["userLocation"], "GB")
        self.assertNotIn("language", json.dumps(_sent(tool)))
        tool, found, _ = _search(searchFirecrawl, o)
        self.assertEqual(_sent(tool)["country"], "GB")
        self.assertTrue(any("Firecrawl has no language filter" in n for n in found["notices"]))
        tool, _, _ = _search(searchTavily, o)
        self.assertEqual((_sent(tool)["country"], _sent(tool)["language"]), ("united kingdom", "en"))

    def test_a_malformed_date_or_window_is_refused_before_the_network(self):
        for opts in ({"start_date": "28.09.2026"}, {"recency": "fortnight"},
                     {"start_date": "2026-10-04", "end_date": "2026-09-28"}):
            with self.subTest(opts=opts):
                tool = _tool(searchExa)
                with self.assertRaises(ValueError):
                    tool.run(QUERY, opts)
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

    def test_a_count_above_the_ceiling_is_capped_and_said(self):
        tool, found, _ = _search(searchBrave, {"max_results": 50})
        self.assertEqual(_sent(tool)["count"], 20)
        self.assertTrue(any("ceiling" in n for n in found["notices"]))

    def test_perplexity_refuses_more_domains_than_it_filters(self):
        with self.assertRaises(ValueError):
            _tool(searchPerplexity).run(QUERY, {"domains": [f"d{i}.com" for i in range(21)]})

    def test_a_typo_is_still_caught_by_the_schema(self):
        self.assertIn("options.recencyy", searchBrave(api_key="k").unknown_keys(
            {"input": QUERY, "options": {"recencyy": "day"}}))


class TestFirecrawl(unittest.TestCase):

    def test_a_domain_with_a_path_goes_as_site_and_is_said(self):
        tool, found, _ = _search(searchFirecrawl, {"domains": ["hbr.org/2026", "cmr.berkeley.edu"]})
        sent = _sent(tool)
        self.assertNotIn("includeDomains", sent)
        self.assertEqual(sent["query"], f"{QUERY} (site:hbr.org/2026 OR site:cmr.berkeley.edu)")
        self.assertTrue(any("host names only" in n for n in found["notices"]))

    def test_domains_and_exclude_domains_together_are_refused(self):
        tool = _tool(searchFirecrawl)
        with self.assertRaises(ValueError):
            tool.run(QUERY, {"domains": ["a.com"], "exclude_domains": ["b.com"]})
        tool._http.request.assert_not_called()

    def test_text_chars_scrapes_the_pages_and_cuts_them(self):
        tool, found, _ = _search(searchFirecrawl, {"text_chars": 11})
        self.assertEqual(_sent(tool)["scrapeOptions"], {"formats": [{"type": "markdown"}]})
        self.assertEqual(found["results"][0]["text"], "# Firecrawl")
        tool, found, _ = _search(searchFirecrawl)
        self.assertNotIn("scrapeOptions", _sent(tool))
        self.assertEqual(found["results"][0]["text"], "")     # not asked, not carried

    def test_news_is_dated_and_web_is_undated_not_misdated(self):
        tool, found, _ = _search(searchFirecrawl, {"sources": ["web", "news"]})
        self.assertEqual(_sent(tool)["sources"], [{"type": "web"}, {"type": "news"}])
        self.assertEqual([r["date"] for r in found["results"]], ["", "2026-09-30"])

    def test_its_own_categories_and_a_warning_become_a_notice(self):
        resp = dict(FIRECRAWL, warning="research category changes on 2026-11-16")
        tool, found, _ = _search(searchFirecrawl, {"categories": ["research", "pdf"]}, resp)
        self.assertEqual(_sent(tool)["categories"], [{"type": "research"}, {"type": "pdf"}])
        self.assertTrue(any("research category changes" in n for n in found["notices"]))

    def test_a_time_filter_is_sent_and_said_not_to_be_enforced(self):
        # Measured 2026-10-08: a 28.09–04.10 range returned the unfiltered results.
        for opts in ({"recency": "week"}, {"start_date": "2026-09-28"}, {"tbs": "qdr:m"}):
            with self.subTest(opts=opts):
                tool, found, _ = _search(searchFirecrawl, opts)
                self.assertIn("tbs", _sent(tool))
                self.assertTrue(any("did not enforce" in n for n in found["notices"]))
        _, found, _ = _search(searchFirecrawl)
        self.assertEqual(found["notices"], [])

    def test_success_false_is_an_error(self):
        with self.assertRaises(RuntimeError):
            _tool(searchFirecrawl, {"success": False, "error": "Insufficient credits"}).run(QUERY)


class TestTavily(unittest.TestCase):

    def test_dates_are_always_asked_for_and_undated_pages_pass_unless_told(self):
        tool, _, _ = _search(searchTavily, {"start_date": "2026-09-28"})
        self.assertTrue(_sent(tool)["include_published_date"])
        self.assertNotIn("filter_by_published_date", _sent(tool))
        tool, _, _ = _search(searchTavily, {"start_date": "2026-09-28", "filter_by_published_date": True})
        self.assertTrue(_sent(tool)["filter_by_published_date"])

    def test_an_hour_is_widened_and_said(self):
        tool, found, _ = _search(searchTavily, {"recency": "hour"})
        self.assertEqual(_sent(tool)["time_range"], "day")
        self.assertTrue(any("widened to a day" in n for n in found["notices"]))

    def test_country_is_a_name_and_only_on_the_general_topic(self):
        for code, name in (("DE", "germany"), ("us", "united states"), ("Japan", "japan")):
            tool, _, _ = _search(searchTavily, {"country": code})
            self.assertEqual(_sent(tool)["country"], name)
        tool, found, _ = _search(searchTavily, {"country": "DE", "topic": "news"})
        self.assertNotIn("country", _sent(tool))
        self.assertTrue(any("general topic" in n for n in found["notices"]))

    def test_passages_split_rfc_dates_read_and_page_text_on_request(self):
        tool, found, _ = _search(searchTavily, {"text_chars": 15})
        self.assertEqual(_sent(tool)["include_raw_content"], "markdown")
        first, second = found["results"]
        self.assertEqual(first["snippets"], ["Lionel Messi, an Argentine footballer",
                                             "Born in 1987, Messi spent his career at Barcelona"])
        self.assertEqual(first["date"], "2025-03-11")
        self.assertEqual(first["text"], "# Lionel Messi\n")
        self.assertEqual(second["date"], "")


class TestParallel(unittest.TestCase):

    def test_the_query_is_the_objective_and_a_keyword_query(self):
        tool, _, _ = _search(searchParallel, {"max_results": 7, "search_queries": ["AI strategy 2026", QUERY]})
        sent = _sent(tool)
        self.assertEqual(sent["objective"], QUERY)
        self.assertEqual(sent["search_queries"], [QUERY, "AI strategy 2026"])     # no duplicate
        self.assertEqual(sent["advanced_settings"]["max_results"], 7)
        tool, _, _ = _search(searchParallel, {"objective": "Find what HBR published on AI this week"})
        self.assertEqual(_sent(tool)["objective"], "Find what HBR published on AI this week")

    def test_domains_with_paths_a_start_date_and_a_country(self):
        tool, _, _ = _search(searchParallel, {"domains": ["bcg.com/publications", ".edu"],
                                              "start_date": "2026-09-28", "country": "GB"})
        s = _sent(tool)["advanced_settings"]
        self.assertEqual(s["source_policy"], {"include_domains": ["bcg.com/publications", ".edu"],
                                              "after_date": "2026-09-28"})
        self.assertEqual(s["location"], "gb")

    def test_recency_is_a_start_date_and_an_hour_is_said_to_be_a_day(self):
        tool, found, _ = _search(searchParallel, {"recency": "week"})
        week_ago = (datetime.now(timezone.utc) - timedelta(weeks=1)).date().isoformat()
        self.assertEqual(_sent(tool)["advanced_settings"]["source_policy"]["after_date"], week_ago)
        self.assertEqual(found["notices"], [])
        _, found, _ = _search(searchParallel, {"recency": "hour"})
        self.assertTrue(any("widened to a day" in n for n in found["notices"]))

    def test_an_end_date_is_declined_and_said(self):
        tool, found, _ = _search(searchParallel, {"start_date": "2026-09-28", "end_date": "2026-10-04"})
        self.assertNotIn("before_date", json.dumps(_sent(tool)))
        self.assertTrue(any("end_date" in n and "start date only" in n for n in found["notices"]))

    def test_domains_and_exclusions_together_are_refused(self):
        tool = _tool(searchParallel)
        with self.assertRaises(ValueError):
            tool.run(QUERY, {"domains": ["a.com"], "exclude_domains": ["b.com"]})
        tool._http.request.assert_not_called()

    def test_excerpts_are_the_snippets_and_a_warning_is_a_notice(self):
        resp = dict(PARALLEL, warnings=[{"type": "warning", "message": "objective was truncated"}])
        tool, found, _ = _search(searchParallel, None, resp)
        self.assertEqual(found["results"][0]["snippets"], ["Sample excerpt 1", "Sample excerpt 2"])
        self.assertEqual([r["date"] for r in found["results"]], ["2024-01-15", ""])
        self.assertTrue(any("objective was truncated" in n for n in found["notices"]))
        self.assertEqual(tool._http.request.call_args.kwargs["headers"]["x-api-key"], "k")


class TestOneRecord(unittest.TestCase):

    KEYS = {"title", "url", "date", "author", "snippets", "text"}

    def test_every_provider_returns_the_same_keys(self):
        for cls in FIXTURE:
            with self.subTest(provider=cls.__name__):
                _, found, _ = _search(cls)
                self.assertEqual(set(found), {"query", "provider", "results", "notices", "cost", "cost_basis", "usage"})
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
        self.assertEqual(day("Tue, 11 Mar 2025 17:00:00 GMT"), "2025-03-11")
        self.assertEqual(day("last week"), "")

    def test_exa_carries_author_highlights_and_text_on_request(self):
        resp = json.loads(json.dumps(EXA)); resp["results"][0]["text"] = "Full page text."
        tool, found, _ = _search(searchExa, {"text_chars": 2000}, resp)
        r = found["results"][0]
        self.assertEqual(r["author"], "Sebastian Raschka, PhD")
        self.assertEqual(r["text"], "Full page text.")
        self.assertEqual(_sent(tool)["contents"], {"highlights": {"maxCharacters": 1000}, "text": {"maxCharacters": 2000}})


class TestWhatACallCost(unittest.TestCase):
    """search() prices every call: the provider's own figure where it reports one, its units at the published
    rate where it reports units, the published rate alone where it reports nothing (rates read 2026-10-08)."""

    def test_exa_reports_dollars(self):
        _, found, _ = _search(searchExa)
        self.assertEqual((found["cost"], found["cost_basis"]), (0.007, "reported by Exa"))

    def test_firecrawl_reports_credits(self):
        _, found, _ = _search(searchFirecrawl)
        self.assertEqual(found["usage"], {"credits": 2})
        self.assertAlmostEqual(found["cost"], 2 * 19 / 5000)
        self.assertIn("credits reported by Firecrawl", found["cost_basis"])

    def test_tavily_is_asked_for_its_usage_and_reports_credits(self):
        resp = dict(TAVILY, usage={"credits": 2})
        tool, found, _ = _search(searchTavily, None, resp)
        self.assertTrue(_sent(tool)["include_usage"])
        self.assertEqual(found["usage"], {"credits": 2})
        self.assertAlmostEqual(found["cost"], 0.016)

    def test_parallel_reports_operations_and_extra_results_are_priced(self):
        resp = dict(PARALLEL, usage=[{"name": "sku_search", "count": 1}, {"name": "sku_extract_excerpts", "count": 10}])
        _, found, _ = _search(searchParallel, None, resp)
        self.assertAlmostEqual(found["cost"], 0.005 + 10 * 0.001)
        self.assertEqual(found["usage"], {"sku_search": 1, "sku_extract_excerpts": 10})
        _, fast, _ = _search(searchParallel, {"mode": "fast"}, dict(PARALLEL, usage=[{"name": "sku_search", "count": 1}]))
        self.assertAlmostEqual(fast["cost"], 0.001)

    def test_a_provider_that_reports_nothing_is_priced_at_its_published_rate(self):
        for cls in (searchPerplexity, searchBrave):
            with self.subTest(provider=cls.__name__):
                _, found, _ = _search(cls)
                self.assertEqual(found["cost"], 0.005)
                self.assertTrue(found["cost_basis"].startswith("published rate ("))
                self.assertEqual(found["usage"], {})
        _, found, _ = _search(searchParallel)           # its answer here carries no usage
        self.assertEqual(found["cost"], 0.005)

    def test_the_model_reading_run_sees_no_price(self):
        self.assertNotIn("$", _tool(searchExa).run(input=QUERY))


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

    def test_run_returns_a_record_whose_text_is_that_layout(self):
        # 3.3.0: run() returns the fields; the model and a prompt read str().
        for cls in FIXTURE:
            with self.subTest(provider=cls.__name__):
                found = _tool(cls).run(input=QUERY)
                self.assertIsInstance(found, Record)
                self.assertEqual(set(found), {"query", "provider", "results", "notices",
                                              "cost", "cost_basis", "usage"})
                self.assertEqual(str(found), cls.format(dict(found)))

    def test_search_is_run_under_its_old_name(self):
        tool = _tool(searchExa)
        self.assertEqual(dict(tool.run(QUERY)), dict(tool.run(QUERY)))

    def test_no_results_is_said(self):
        self.assertIn("(0 results)", str(_tool(searchExa, {"results": []}).run(input=QUERY)))


class TestFactory(unittest.TestCase):
    """search(service) — the one way the documentation shows."""

    def test_every_service_by_name(self):
        from yait_aichain.tools.search import SERVICES
        self.assertEqual(set(SERVICES), {"perplexity", "brave", "exa", "firecrawl", "tavily", "parallel"})
        for service, cls in SERVICES.items():
            with self.subTest(service):
                self.assertIs(type(search(service, api_key="k")), cls)
                self.assertIs(type(search(service.upper(), api_key="k")), cls)

    def test_an_unknown_service_names_the_ones_that_exist(self):
        with self.assertRaisesRegex(ValueError, "one of: perplexity, brave, exa"):
            search("exaa", api_key="k")

    def test_name_and_description_follow_the_service_and_can_be_overridden(self):
        self.assertEqual(search("exa", api_key="k").name, "searchExa")
        self.assertEqual(search("perplexity", api_key="k").name, "searchPerplexity")
        t = search("exa", api_key="k", name="web", description="Search the web.")
        self.assertEqual((t.name, t.description), ("web", "Search the web."))
        self.assertEqual(t.schema()["function"]["name"], "web")
        self.assertEqual(searchExa.name, "searchExa")       # the class is untouched

    def test_a_call_setting_is_not_a_factory_setting(self):
        # One place for each setting: options belong to the call.
        with self.assertRaises(TypeError):
            search("exa", api_key="k", max_results=5)


class TestContract(unittest.TestCase):

    def test_read_risk_a_shared_base_and_a_named_key(self):
        for cls, var in ((searchBrave, "BRAVE_SEARCH_API_KEY"), (searchPerplexity, "PERPLEXITY_API_KEY"),
                         (searchExa, "EXA_API_KEY"),
                         (searchFirecrawl, "FIRECRAWL_API_KEY"), (searchTavily, "TAVILY_API_KEY"),
                         (searchParallel, "PARALLEL_API_KEY")):
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
                self.assertIn("HTTP 401", str(ctx.exception))
                self.assertEqual(ctx.exception.status, 401)
                result = _tool(cls, {"error": "out of credits"}, status=402)(input=QUERY)
                self.assertFalse(result.success)
                self.assertIn("HTTP 402", result.error)

    def test_an_empty_query_is_refused(self):
        with self.assertRaises(ValueError):
            _tool(searchBrave).run("  ")


def _real(var):
    return os.environ.get(var, "") not in ("", "test-key", "k")


class TestSearchLive(unittest.TestCase):
    """The same query and window to every provider with a key: the same record back."""

    OPTS = {"max_results": 3, "start_date": (TODAY - timedelta(days=30)).isoformat(), "domains": ["hbr.org"]}

    def _check(self, cls, var):
        if not _real(var):
            self.skipTest(f"{var} not set to a real key")
        found = cls().run("leadership and AI", self.OPTS)
        self.assertTrue(found["results"])
        for r in found["results"]:
            self.assertEqual(set(r), TestOneRecord.KEYS)
            self.assertIn("hbr.org", r["url"])

    def test_brave(self): self._check(searchBrave, "BRAVE_SEARCH_API_KEY")
    def test_perplexity(self): self._check(searchPerplexity, "PERPLEXITY_API_KEY")
    def test_exa(self): self._check(searchExa, "EXA_API_KEY")
    def test_firecrawl(self): self._check(searchFirecrawl, "FIRECRAWL_API_KEY")
    def test_tavily(self): self._check(searchTavily, "TAVILY_API_KEY")
    def test_parallel(self): self._check(searchParallel, "PARALLEL_API_KEY")


class TestADayThatDoesNotExist(unittest.TestCase):

    def test_an_impossible_date_is_no_date(self):
        for value in ("2026-13-45", "2026-02-30T10:00:00Z", "Feb 30, 2026", "31 Apr 2026"):
            with self.subTest(value=value):
                self.assertEqual(day(value), "")
        self.assertEqual((day("2026-09-30T12:00:00Z"), day("Sep 3, 2026"), day("3 Sep 2026")),
                         ("2026-09-30", "2026-09-03", "2026-09-03"))


if __name__ == "__main__":
    unittest.main()
