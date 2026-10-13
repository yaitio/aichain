"""
The chunker keeps every line and joins only what belongs together.

Overlap glued a table's header to a data row and started the 'Pricing'
section with the end of 'Intro'; one ``_`` switched a CSV to Markdown and cut
its rows; CJK text without spaces was cut at fixed offsets; CRLF paragraphs
were not paragraphs; a "```python" line inside a block closed it and the rest
was lost; a negative overlap broke the size contract; typos in options were
ignored.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.tools import chunk


class TestOverlapJoinsTextOnly(unittest.TestCase):

    def test_a_section_does_not_start_with_the_one_before(self):
        doc = "# Intro\n\n" + "intro words " * 30 + "\n\n# Pricing\n\n" + "price words " * 30
        pieces = chunk(max_chars=200, overlap_chars=40).run(doc)
        pricing = [p for p in pieces if p["metadata"]["headings"] == ["Pricing"]]
        self.assertNotIn("intro", pricing[0]["text"])

    def test_a_table_gets_no_overlap(self):
        table = "| a | b |\n|---|---|\n" + "".join(f"| {i} | x |\n" for i in range(40))
        pieces = chunk(max_chars=120, overlap_chars=30).run("# T\n\n" + table)
        for p in pieces:
            self.assertTrue(p["text"].startswith("| a | b |"), p["text"][:30])


class TestTextIsKept(unittest.TestCase):

    def test_a_csv_with_an_underscore_is_plain_text(self):
        csv = "user_id,name\n" + "".join(f"{i},n{i}\n" for i in range(200))
        pieces = chunk(max_chars=300, overlap_chars=0).run(csv)
        for p in pieces:
            for line in p["text"].split("\n"):
                self.assertRegex(line, r"^(user_id,name|\d+,n\d+)$")

    def test_cjk_is_cut_at_sentence_ends(self):
        text = "这是一个句子。" * 100
        pieces = chunk(max_chars=100, overlap_chars=0).run(text)
        self.assertTrue(all(p["text"].endswith("。") for p in pieces))

    def test_crlf_paragraphs_are_paragraphs(self):
        text = ("alpha " * 20).strip() + "\r\n\r\n" + ("beta " * 20).strip()
        pieces = chunk(max_chars=150, overlap_chars=0).run(text)
        self.assertEqual(len(pieces), 2)
        self.assertNotIn("\r", pieces[0]["text"])

    def test_an_info_string_line_does_not_close_a_block(self):
        doc = "```md\nbefore\n```python\nprint(1)\n```\n\nafter"
        text = " ".join(p["text"] for p in chunk(max_chars=500, overlap_chars=0).run(doc))
        for word in ("before", "print(1)", "after"):
            self.assertIn(word, text)


class TestRefused(unittest.TestCase):

    def test_bad_sizes(self):
        for kw in ({"max_chars": 0}, {"max_chars": True}, {"overlap_chars": -5}, {"max_chars": "800"}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                chunk(**kw)

    def test_unknown_options(self):
        with self.assertRaises(ValueError):
            chunk().run("text", {"max_char": 10})


if __name__ == "__main__":
    unittest.main()
