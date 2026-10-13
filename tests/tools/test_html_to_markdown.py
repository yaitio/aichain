"""
A page as Markdown, from the standard library.

Until 4.0 fetchLocal converted pages with MarkItDown, which brought magika
and onnxruntime to turn tags into text. Pinned here: what a reader needs
survives, and what a reader does not see is left out.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.tools.fetch._html import html_to_markdown


class TestHtmlToMarkdown(unittest.TestCase):

    def test_blocks_and_inline(self):
        md = html_to_markdown("<h2>Plan</h2><p>Use <b>this</b>, <em>not</em> "
                              "<a href='https://x.io'>that</a>.</p>")
        self.assertEqual(md, "## Plan\n\nUse **this**, *not* [that](https://x.io).")

    def test_lists_code_and_tables(self):
        md = html_to_markdown("<ol><li>a</li><li>b</li></ol><pre>x = 1\n  y</pre>"
                              "<table><tr><th>k</th><th>v</th></tr><tr><td>1</td><td>2</td></tr></table>")
        self.assertIn("1. a\n2. b", md)
        self.assertIn("```\nx = 1\n  y\n```", md)
        self.assertIn("| k | v |\n|---|---|\n| 1 | 2 |", md)

    def test_what_a_reader_does_not_see_is_left_out(self):
        md = html_to_markdown("<head><title>T</title></head><script>steal()</script>"
                              "<style>p{}</style><p hidden>secret</p><p>seen</p>")
        self.assertEqual(md, "seen")

    def test_entities_are_text(self):
        self.assertEqual(html_to_markdown("<p>R&amp;D &lt;3</p>"), "R&D <3")


if __name__ == "__main__":
    unittest.main()
