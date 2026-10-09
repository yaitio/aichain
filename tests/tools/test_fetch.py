"""
Fetch tools — one record of a page from every provider.

What is pinned, offline:

  * every provider's answer becomes the same page record, failures included —
    one bad URL is a page with ``error`` set, not a failed call;
  * URLs are batched to what each provider takes, and the answer is returned
    in the order asked;
  * the cost is read from the answer where the provider reports it, and from
    the published rate where it does not;
  * ``max_chars`` reaches the provider where it takes a limit and cuts here
    where it does not;
  * ``fetchLocal`` reads robots.txt before the page and skips what it
    disallows, without making the request; it needs MarkItDown and says how
    to install it when it is missing.

Fixtures are real answers for https://example.com captured 2026-10-09 and
trimmed; the failure entries are the shapes the same APIs returned for an
unreachable host.
"""

import builtins
import json
import os
import sys
import unittest
import warnings
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.tools import (Fetch, Record, fetch, fetchExa, fetchFirecrawl, fetchLocal,
                                fetchParallel, fetchTavily)

OK, BAD = "https://example.com", "https://example.invalid/x"
TEXT = "This domain is for use in documentation examples without needing permission."

FIRECRAWL = {"success": True, "data": {"markdown": TEXT, "metadata": {
    "title": "Example Domain", "language": "en", "sourceURL": OK, "url": OK + "/",
    "statusCode": 200, "contentType": "text/html; charset=utf-8", "creditsUsed": 1,
    "publishedTime": "2026-09-30T10:00:00Z", "author": "IANA"}}}

TAVILY = {"results": [{"url": OK, "title": "Example Domain", "raw_content": TEXT, "images": []}],
          "failed_results": [{"url": BAD, "error": "Failed to fetch url"}],
          "response_time": 0.26, "usage": {"credits": 0}}

EXA = {"results": [{"id": OK, "title": "Example Domain", "url": OK, "author": None,
                    "publishedDate": "2026-09-30T00:00:00.000Z", "text": TEXT}],
       "statuses": [{"id": OK, "status": "success", "source": "cached"},
                    {"id": BAD, "status": "error", "error": {"tag": "CRAWL_LIVECRAWL_TIMEOUT", "httpStatusCode": 504}}],
       "costDollars": {"total": 0.001, "contents": {"text": 0.001}}}

PARALLEL = {"extract_id": "extract_1",
            "results": [{"url": OK, "title": "Example Domain", "publish_date": None,
                         "excerpts": [TEXT], "full_content": TEXT}],
            "errors": [{"url": BAD, "error_type": "connect_error"}],
            "warnings": [{"type": "warning", "message": "Neither objective nor search_queries were provided, "
                                                        "provide at least one to increase the relevance of excerpts."}],
            "usage": [{"name": "sku_extract_excerpts", "count": 1}]}

FIXTURE = {fetchFirecrawl: FIRECRAWL, fetchTavily: TAVILY, fetchExa: EXA, fetchParallel: PARALLEL}
KEYS = {"url", "final_url", "title", "published", "author", "text", "error"}


def _tool(cls, response=None, status=200):
    tool = cls(api_key="k")
    reply = MagicMock()
    reply.status = status
    response = FIXTURE[cls] if response is None else response
    reply.data = (json.dumps(response) if isinstance(response, dict) else response).encode()
    tool._http = MagicMock()
    tool._http.request.return_value = reply
    return tool


def _sent(tool, call=-1):
    return json.loads(tool._http.request.call_args_list[call].kwargs["body"])


class TestOneRecord(unittest.TestCase):

    def test_every_provider_returns_the_same_record(self):
        for cls in FIXTURE:
            with self.subTest(cls.__name__):
                found = _tool(cls).fetch(OK)
                self.assertEqual(set(found), {"pages", "notices", "cost", "cost_basis", "usage"})
                (p,) = found["pages"]
                self.assertEqual(set(p), KEYS)
                self.assertEqual(p["url"], OK)
                self.assertEqual(p["title"], "Example Domain")
                self.assertIn("documentation examples", p["text"])
                self.assertEqual(p["error"], "")

    def test_a_failed_url_is_a_page_with_an_error(self):
        for cls in (fetchTavily, fetchExa, fetchParallel):
            with self.subTest(cls.__name__):
                pages = _tool(cls).fetch([OK, BAD])["pages"]
                self.assertEqual([p["url"] for p in pages], [OK, BAD])
                self.assertEqual(pages[0]["error"], "")
                self.assertTrue(pages[1]["error"])
                self.assertEqual(pages[1]["text"], "")

    def test_exa_says_why(self):
        self.assertEqual(_tool(fetchExa).fetch([OK, BAD])["pages"][1]["error"], "CRAWL_LIVECRAWL_TIMEOUT (HTTP 504)")

    def test_a_url_the_provider_never_mentions_is_still_a_page(self):
        tool = _tool(fetchTavily, {"results": [], "failed_results": []})
        (p,) = tool.fetch(OK)["pages"]
        self.assertIn("returned nothing", p["error"])

    def test_dates_and_authors_where_the_provider_has_them(self):
        p = _tool(fetchFirecrawl).fetch(OK)["pages"][0]
        self.assertEqual((p["published"], p["author"], p["final_url"]), ("2026-09-30", "IANA", OK + "/"))
        self.assertEqual(_tool(fetchExa).fetch(OK)["pages"][0]["published"], "2026-09-30")
        # Tavily has no date or author at all: empty, never guessed.
        p = _tool(fetchTavily).fetch(OK)["pages"][0]
        self.assertEqual((p["published"], p["author"]), ("", ""))

    def test_firecrawl_error_page_is_not_the_page(self):
        bad = json.loads(json.dumps(FIRECRAWL)); bad["data"]["metadata"]["statusCode"] = 404
        p = _tool(fetchFirecrawl, bad).fetch(OK)["pages"][0]
        self.assertEqual(p["text"], "")
        self.assertIn("404", p["error"])

    def test_firecrawl_failure_is_a_page_and_out_of_credits_raises(self):
        p = _tool(fetchFirecrawl, {"success": False, "error": "timeout"}, status=408).fetch(OK)["pages"][0]
        self.assertIn("timeout", p["error"])
        with self.assertRaises(RuntimeError):
            _tool(fetchFirecrawl, {"success": False, "error": "Payment required"}, status=402).fetch(OK)

    def test_an_api_error_raises(self):
        for cls in (fetchTavily, fetchExa, fetchParallel):
            with self.subTest(cls.__name__), self.assertRaises(RuntimeError):
                _tool(cls, {"error": "bad key"}, status=401).fetch(OK)

    def test_a_url_that_is_not_http_is_refused_before_the_wire(self):
        tool = _tool(fetchExa)
        for bad in ("file:///etc/passwd", "", "example.com", []):
            with self.subTest(bad), self.assertRaises(ValueError):
                tool.fetch(bad)
        tool._http.request.assert_not_called()


class TestBatching(unittest.TestCase):

    def test_each_provider_takes_its_own_batch(self):
        urls = [f"https://example.com/{i}" for i in range(45)]
        for cls, calls in ((fetchFirecrawl, 45), (fetchTavily, 3), (fetchParallel, 3), (fetchExa, 1)):
            with self.subTest(cls.__name__):
                tool = _tool(cls, {"results": [], "data": {"metadata": {}}, "success": True})
                tool.fetch(urls)
                self.assertEqual(tool._http.request.call_count, calls)

    def test_duplicates_are_fetched_once(self):
        tool = _tool(fetchExa)
        self.assertEqual(len(tool.fetch([OK, OK, OK])["pages"]), 1)
        self.assertEqual(_sent(tool)["urls"], [OK])


class TestMaxChars(unittest.TestCase):

    def test_reaches_the_provider_where_it_takes_a_limit(self):
        tool = _tool(fetchExa); tool.fetch(OK, {"max_chars": 500})
        self.assertEqual(_sent(tool)["text"], {"maxCharacters": 500})
        tool = _tool(fetchParallel); tool.fetch(OK, {"max_chars": 500})
        self.assertEqual(_sent(tool)["advanced_settings"]["full_content"], {"max_chars_per_result": 500})

    def test_cut_here_for_every_provider(self):
        for cls in FIXTURE:
            with self.subTest(cls.__name__):
                self.assertEqual(len(_tool(cls).fetch(OK, {"max_chars": 10})["pages"][0]["text"]), 10)


class TestProviderOptions(unittest.TestCase):

    def test_each_reaches_the_wire(self):
        tool = _tool(fetchFirecrawl); tool.fetch(OK, {"only_main_content": False})
        self.assertIs(_sent(tool)["onlyMainContent"], False)
        tool = _tool(fetchFirecrawl); tool.fetch(OK)
        self.assertIs(_sent(tool)["onlyMainContent"], True)
        tool = _tool(fetchTavily); tool.fetch(OK, {"extract_depth": "advanced"})
        self.assertEqual(_sent(tool)["extract_depth"], "advanced")
        tool = _tool(fetchExa); tool.fetch(OK, {"max_age_hours": 0})
        self.assertEqual(_sent(tool)["maxAgeHours"], 0)
        tool = _tool(fetchParallel); tool.fetch(OK, {"max_age_seconds": 600})
        self.assertEqual(_sent(tool)["advanced_settings"]["fetch_policy"], {"max_age_seconds": 600})

    def test_options_are_declared_in_the_schema(self):
        for cls, opt in ((fetchFirecrawl, "only_main_content"), (fetchTavily, "extract_depth"),
                         (fetchExa, "max_age_hours"), (fetchParallel, "max_age_seconds")):
            props = cls.parameters["properties"]["options"]["properties"]
            self.assertIn(opt, props); self.assertIn("max_chars", props)

    def test_parallel_drops_only_its_no_objective_warning(self):
        tool = _tool(fetchParallel)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.assertEqual(tool.fetch(OK)["notices"], [])
        other = dict(PARALLEL, warnings=[{"type": "warning", "message": "URL was truncated"}])
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            self.assertEqual(_tool(fetchParallel, other).fetch(OK)["notices"], ["Parallel: URL was truncated"])
        self.assertTrue(seen)


class TestWhatACallCost(unittest.TestCase):

    def test_read_from_the_answer(self):
        self.assertAlmostEqual(_tool(fetchExa).fetch(OK)["cost"], 0.001)
        self.assertEqual(_tool(fetchExa).fetch(OK)["cost_basis"], "reported by Exa")
        found = _tool(fetchFirecrawl).fetch(OK)
        self.assertAlmostEqual(found["cost"], 19 / 5000)
        self.assertEqual(found["usage"], {"credits": 1})
        found = _tool(fetchParallel).fetch(OK)
        self.assertAlmostEqual(found["cost"], 0.001)
        self.assertEqual(found["usage"], {"sku_extract_excerpts": 1})
        # Tavily bills per five URLs, and reports 0 below that: what it says is what is returned.
        self.assertEqual(_tool(fetchTavily).fetch(OK)["cost"], 0.0)

    def test_summed_over_batches(self):
        tool = _tool(fetchFirecrawl)
        found = tool.fetch([OK, OK + "/a", OK + "/b"])
        self.assertAlmostEqual(found["cost"], 3 * 19 / 5000)
        self.assertEqual(found["usage"], {"credits": 3})

    def test_published_rate_when_the_answer_is_silent(self):
        tool = _tool(fetchExa, {"results": EXA["results"]})
        found = tool.fetch(OK)
        self.assertAlmostEqual(found["cost"], 0.001)
        self.assertIn("published rate", found["cost_basis"])


class TestTheTextAModelReads(unittest.TestCase):

    def test_run_returns_a_record_whose_text_is_the_page(self):
        found = _tool(fetchFirecrawl).run(OK)
        self.assertIsInstance(found, Record)
        self.assertEqual(found["pages"][0]["title"], "Example Domain")
        self.assertTrue(str(found).startswith("Title: Example Domain\nURL: https://example.com/\n"
                                              "Published: 2026-09-30 · Author: IANA\n\n"))
        self.assertIn("documentation examples", str(found))

    def test_a_failed_page_is_its_reason_not_an_exception(self):
        # 3.3.0: run() raised on a single failed URL; a failed page is now a
        # record like any other, and the model reads why.
        found = _tool(fetchTavily).run([OK, BAD])
        self.assertIn("Could not fetch https://example.invalid/x: Failed to fetch url", str(found))
        self.assertIn("Title: Example Domain", str(found))

    def test_call_succeeds_with_the_failed_page_in_it(self):
        res = _tool(fetchTavily)(BAD)
        self.assertTrue(res.success)
        self.assertEqual(res.output["pages"][0]["error"], "Failed to fetch url")

    def test_fetch_is_run_under_its_old_name(self):
        tool = _tool(fetchExa)
        self.assertEqual(dict(tool.fetch(OK)), dict(tool.run(OK)))


class TestFactory(unittest.TestCase):
    """fetch(service) — the one way the documentation shows."""

    def test_every_service_by_name(self):
        from yait_aichain.tools.fetch import SERVICES
        self.assertEqual(set(SERVICES), {"firecrawl", "tavily", "exa", "parallel", "local"})
        for service in ("firecrawl", "tavily", "exa", "parallel"):
            with self.subTest(service):
                self.assertIs(type(fetch(service, api_key="k")), SERVICES[service])
        self.assertIs(type(fetch("local")), fetchLocal)

    def test_an_unknown_service_names_the_ones_that_exist(self):
        with self.assertRaisesRegex(ValueError, "one of: firecrawl, tavily, exa, parallel, local"):
            fetch("firecrawlx", api_key="k")

    def test_local_settings_are_locals_and_a_key_is_not(self):
        t = fetch("local", respect_robots=False, robots_agents=("*",), timeout=(5, 10))
        self.assertEqual((t.respect_robots, t.robots_agents), (False, ("*",)))
        with self.assertRaisesRegex(TypeError, "are fetch\\('local'\\)'s"):
            fetch("exa", api_key="k", timeout=3)
        with self.assertRaisesRegex(TypeError, "takes no api_key"):
            fetch("local", api_key="k")
        with self.assertRaisesRegex(TypeError, "max_chars"):
            fetch("local", max_chars=10)              # an option of the call

    def test_name_and_description_override(self):
        t = fetch("exa", api_key="k", name="read_page", description="Read a page.")
        self.assertEqual((t.name, t.description), ("read_page", "Read a page."))
        t = fetch("local", name="read_local")
        self.assertEqual(t.name, "read_local")


class TestContract(unittest.TestCase):

    def test_risk_is_read(self):
        for cls in (*FIXTURE, fetchLocal):
            self.assertEqual(cls.risk, "read")

    def test_a_missing_key_says_which(self):
        for cls in FIXTURE:
            with self.subTest(cls.__name__), patch.dict(os.environ, {cls._ENV_KEY: ""}):
                with self.assertRaisesRegex(ValueError, cls._ENV_KEY):
                    cls()

    def test_every_fetch_is_a_fetch(self):
        for cls in (*FIXTURE, fetchLocal):
            self.assertTrue(issubclass(cls, Fetch))


def _local(robots=None, robots_status=200, page_status=200, robots_raises=False, **kw):
    """A fetchLocal whose network is a mock: robots.txt and one HTML page."""
    tool = fetchLocal(**kw)

    def get(url, **_):
        r = MagicMock()
        if url.endswith("/robots.txt"):
            if robots_raises:
                raise ConnectionError("no route to host")
            r.status_code, r.text = robots_status, robots or ""
        else:
            r.status_code, r.url = page_status, url
            r.headers = {"content-type": "text/html; charset=utf-8"}
            r.encoding = "utf-8"
            r.content = (b'<html><head><title>Example Domain</title>'
                         b'<meta property="article:published_time" content="2026-09-30T10:00:00Z">'
                         b'<meta name="author" content="IANA"></head><body><p>Hi</p></body></html>')
        return r

    tool._session = MagicMock()
    tool._session.get.side_effect = get
    tool._md = MagicMock()
    tool._md.convert_response.return_value.text_content = TEXT
    return tool


@patch("yait_aichain.tools.fetch.local.assert_safe_url", lambda u: None)
class TestLocal(unittest.TestCase):

    def test_a_page_with_its_head(self):
        found = _local().fetch(OK)
        (p,) = found["pages"]
        self.assertEqual((p["title"], p["published"], p["author"], p["text"]),
                         ("Example Domain", "2026-09-30", "IANA", TEXT))
        self.assertEqual((found["cost"], found["cost_basis"]), (0.0, "runs locally"))

    def test_robots_refusal_skips_the_page_without_requesting_it(self):
        tool = _local("User-agent: ClaudeBot\nDisallow: /\n")
        (p,) = tool.fetch(OK + "/article")["pages"]
        self.assertEqual(p["error"], "robots.txt disallows ClaudeBot")
        asked = [c.args[0] for c in tool._session.get.call_args_list]
        self.assertEqual(asked, [OK + "/robots.txt"])

    def test_robots_is_read_once_per_host(self):
        tool = _local("User-agent: *\nAllow: /\n")
        tool.fetch([OK + "/a", OK + "/b"])
        asked = [c.args[0] for c in tool._session.get.call_args_list]
        self.assertEqual(asked.count(OK + "/robots.txt"), 1)

    def test_rfc_9309_status_codes(self):
        self.assertEqual(_local(robots_status=404).fetch(OK)["pages"][0]["error"], "")
        self.assertIn("server error", _local(robots_status=503).fetch(OK)["pages"][0]["error"])
        self.assertIn("could not be reached", _local(robots_raises=True).fetch(OK)["pages"][0]["error"])

    def test_robots_can_be_turned_off_or_narrowed(self):
        self.assertEqual(_local("User-agent: *\nDisallow: /\n", respect_robots=False).fetch(OK)["pages"][0]["error"], "")
        tool = _local("User-agent: ClaudeBot\nDisallow: /\n", robots_agents=("*",))
        self.assertEqual(tool.fetch(OK)["pages"][0]["error"], "")

    def test_an_http_error_is_a_page(self):
        self.assertEqual(_local(page_status=403).fetch(OK)["pages"][0]["error"], "HTTP 403")


class TestLocalNeedsMarkItDown(unittest.TestCase):

    def test_says_how_to_install(self):
        real = builtins.__import__

        def no_markitdown(name, *a, **kw):
            if name == "markitdown":
                raise ImportError("no markitdown")
            return real(name, *a, **kw)

        with patch("builtins.__import__", no_markitdown):
            with self.assertRaisesRegex(ImportError, r'yait-aichain\[fetch\]'):
                fetchLocal()


def _real(var):
    return os.environ.get(var, "") not in ("", "test-key", "k")


class TestFetchLive(unittest.TestCase):
    """The same open page from every provider with a key: the same record back."""

    URL = "https://www.iana.org/help/example-domains"

    def _check(self, cls, var=None):
        if var and not _real(var):
            self.skipTest(f"{var} not set to a real key")
        (p,) = cls().fetch(self.URL)["pages"]
        self.assertEqual(set(p), KEYS)
        self.assertEqual(p["error"], "")
        self.assertIn("example", p["text"].lower())

    def test_firecrawl(self): self._check(fetchFirecrawl, "FIRECRAWL_API_KEY")
    def test_tavily(self): self._check(fetchTavily, "TAVILY_API_KEY")
    def test_exa(self): self._check(fetchExa, "EXA_API_KEY")
    def test_parallel(self): self._check(fetchParallel, "PARALLEL_API_KEY")
    def test_local(self): self._check(fetchLocal)


if __name__ == "__main__":
    unittest.main()
