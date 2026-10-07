"""
convertToMD carries headers out and the page's metadata back.

Two gaps found collecting a week of business-school articles, 2026-10-07:

  * blogs.lse.ac.uk answers 403 to the ``python-requests`` agent on every
    path, so ``headers=`` sends a caller's ``User-Agent`` (or anything else);
  * MarkItDown drops ``<head>``, and library.hbs.edu states an article's
    publication date only there, so the date was lost. ``last_metadata`` now
    holds what the head says, and ``metadata: true`` puts it on top of the
    Markdown as front matter.

Offline: the pages below are trimmed shapes of the real ones, served by a
mocked session; the response goes through MarkItDown's own converter.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import markitdown  # noqa: F401
    import requests
except ImportError:  # pragma: no cover - the convert extra is optional
    markitdown = None

from yait_aichain.tools.convert.to_md import convertToMD, page_metadata

#: JSON-LD only, the date nowhere in the body — the library.hbs.edu shape.
LD_PAGE = """<html><head><title>Where Senators Sit | Working Knowledge</title>
<script type="application/ld+json">{"@context": "https://schema.org", "@graph": [
  {"@type": "WebSite", "name": "Working Knowledge"},
  {"@type": "Article", "headline": "Where Senators Sit Can Shape How They Vote",
   "datePublished": "2026-09-30", "dateModified": "2026-10-01T09:00:00Z",
   "author": [{"@type": "Person", "name": "Jane Roe"}, {"@type": "Person", "name": "John Doe"}]}]}
</script><link rel="canonical" href="https://library.hbs.edu/working-knowledge/where-senators-sit"></head>
<body><h1>Where Senators Sit Can Shape How They Vote</h1><p>Seating changes votes.</p></body></html>"""

#: Meta tags only — the common Open Graph shape.
META_PAGE = """<html><head><title>Plain title</title>
<meta property="og:title" content="Organisational digital twins are coming">
<meta property="article:published_time" content="2026-09-28T07:00:00+00:00">
<meta name="author" content="A. Writer">
<meta property="og:site_name" content="LSE Business Review">
<meta name="description" content="Digital twins of whole organisations.">
</head><body><p>Body text.</p></body></html>"""


class TestPageMetadata(unittest.TestCase):

    def test_json_ld_is_read_through_a_graph(self):
        m = page_metadata(LD_PAGE)
        self.assertEqual(m["published"], "2026-09-30")
        self.assertEqual(m["modified"], "2026-10-01T09:00:00Z")
        self.assertEqual(m["title"], "Where Senators Sit Can Shape How They Vote")
        self.assertEqual(m["author"], "Jane Roe, John Doe")
        self.assertEqual(m["canonical"], "https://library.hbs.edu/working-knowledge/where-senators-sit")

    def test_meta_tags_when_there_is_no_json_ld(self):
        m = page_metadata(META_PAGE)
        self.assertEqual(m["published"], "2026-09-28T07:00:00+00:00")
        self.assertEqual(m["title"], "Organisational digital twins are coming")
        self.assertEqual(m["author"], "A. Writer")
        self.assertEqual(m["site"], "LSE Business Review")
        self.assertEqual(m["description"], "Digital twins of whole organisations.")

    def test_a_field_the_page_lacks_is_left_out_not_guessed(self):
        m = page_metadata("<html><head><title>Only a title</title></head><body>x</body></html>")
        self.assertEqual(m, {"title": "Only a title"})

    def test_broken_json_ld_does_not_lose_the_meta_tags(self):
        m = page_metadata('<head><script type="application/ld+json">{not json</script>'
                          '<meta name="date" content="2026-09-29"></head>')
        self.assertEqual(m["published"], "2026-09-29")


def _response(html, url="https://example.com/a", ctype="text/html; charset=utf-8"):
    r = requests.Response()
    r.status_code, r.url, r._content, r._content_consumed = 200, url, html.encode(), True
    r.headers["content-type"] = ctype
    return r


@unittest.skipIf(markitdown is None, "markitdown is not installed")
class TestTheTool(unittest.TestCase):

    def _run(self, tool, html, **kw):
        tool._get_converter()
        with patch("yait_aichain.tools.convert.to_md.assert_safe_url"), \
             patch.object(type(tool._session), "get", return_value=_response(html)) as get:
            out = tool.run(input="https://example.com/a", **kw)
        return out, get

    def test_headers_ride_every_request(self):
        tool = convertToMD(headers={"User-Agent": "Mozilla/5.0 test"})
        session = tool._get_converter()._requests_session
        self.assertIs(session, tool._session)
        self.assertEqual(session.headers["User-Agent"], "Mozilla/5.0 test")
        self.assertIn("text/markdown", session.headers["Accept"])

    def test_a_url_is_fetched_once_and_its_head_kept(self):
        tool = convertToMD()
        text, get = self._run(tool, LD_PAGE)
        self.assertEqual(get.call_count, 1)
        self.assertIn("Seating changes votes.", text)
        self.assertFalse(text.startswith("---"))            # unchanged unless asked
        self.assertEqual(tool.last_metadata["published"], "2026-09-30")
        self.assertEqual(tool.last_metadata["url"], "https://example.com/a")

    def test_front_matter_on_request(self):
        tool = convertToMD()
        text, _ = self._run(tool, META_PAGE, options={"metadata": True})
        head, body = text.split("\n---\n", 1)
        self.assertTrue(head.startswith("---\n"))
        self.assertIn('published: "2026-09-28T07:00:00+00:00"', head)
        self.assertIn('title: "Organisational digital twins are coming"', head)
        self.assertIn("Body text.", body)

    def test_a_page_without_a_charset_is_read_as_utf8(self):
        tool = convertToMD()
        self._run(tool, '<head><meta property="og:title" content="Café — naïve"></head>')
        self.assertEqual(tool.last_metadata["title"], "Café — naïve")

    def test_a_non_html_answer_has_no_metadata(self):
        tool = convertToMD()
        tool._get_converter()
        with patch("yait_aichain.tools.convert.to_md.assert_safe_url"), \
             patch.object(type(tool._session), "get",
                          return_value=_response("a,b\n1,2\n", ctype="text/csv")):
            tool.run(input="https://example.com/a.csv")
        self.assertEqual(tool.last_metadata, {})

    def test_a_local_html_file_is_read_too(self):
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
            fh.write(LD_PAGE)
        try:
            tool = convertToMD()
            text = tool.run(input=fh.name, options={"metadata": True})
            self.assertEqual(tool.last_metadata["published"], "2026-09-30")
            self.assertTrue(text.startswith("---\n"))
        finally:
            os.unlink(fh.name)

    def test_an_http_error_is_a_failed_call(self):
        tool = convertToMD()
        tool._get_converter()
        forbidden = _response("<html>Forbidden</html>")
        forbidden.status_code = 403
        with patch("yait_aichain.tools.convert.to_md.assert_safe_url"), \
             patch.object(type(tool._session), "get", return_value=forbidden):
            result = tool(input="https://blogs.lse.ac.uk/businessreview/")
        self.assertFalse(result.success)
        self.assertIn("403", result.error)


if __name__ == "__main__":
    unittest.main()
