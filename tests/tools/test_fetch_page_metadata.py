"""
A web page's metadata, read from its <head> — what fetchLocal reports as a
page's title, publication date and author.

MarkItDown keeps only the body, and some sites state the date nowhere else
(library.hbs.edu, checked 2026-10-07). Written for convertToMD in 3.3.0; the
reader moved to tools.fetch._page in 3.8.0, when the convert tools left the
library, and these tests moved with it.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.tools.fetch._page import page_metadata

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




if __name__ == "__main__":
    unittest.main()
