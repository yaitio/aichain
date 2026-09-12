"""
The chunker's contract, as its own docstring states it.

`PLAN.md` has listed these as missing since the June audit. The reason they
are worth writing is that the contract is *arithmetic* — `chars <= max_chars`,
overlap of exactly N characters, an edge case per input shape — and arithmetic
is the one thing a reader cannot verify by looking at a chunk and thinking it
seems about right. Everything asserted here is a sentence already written in
`_chunk.py`; what was missing is anything that fails when the sentence stops
being true.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain.tools.vectordb import vectorChunk               # noqa: E402

PROSE = ("Alpha beta gamma delta epsilon zeta eta theta iota kappa. " * 40)

MARKDOWN = """# Title

Some opening prose about the subject at hand, long enough to matter.

## First section

More prose here, also reasonably long so that it survives a merge.

```python
def hello():
    return "world"
```

| a | b |
|---|---|
| 1 | 2 |
"""


class TestTheCeilingIsNeverExceeded(unittest.TestCase):
    """`chars: int — len(text), always <= max_chars`. The word is *always*.

    Note the constraint that caught the first draft of these tests: the
    default `overlap_chars` is 150, and an overlap at least as wide as the
    ceiling raises. So a small `max_chars` needs a small overlap named with
    it — which is the contract working, not a limitation to route around.
    """

    def test_prose(self):
        for size in (60, 120, 400):
            with self.subTest(max_chars=size):
                chunker = vectorChunk(max_chars=size, overlap_chars=size // 4)
                for chunk in chunker.run(PROSE):
                    self.assertLessEqual(chunk["chars"], size)

    def test_markdown_with_code_and_a_table(self):
        for chunk in vectorChunk(max_chars=80, overlap_chars=20).run(MARKDOWN):
            self.assertLessEqual(chunk["chars"], 80)

    def test_a_single_word_longer_than_the_ceiling(self):
        """Documented: hard-split at the character level. The alternative is
        one chunk that breaks the ceiling and the embedder downstream."""
        chunker = vectorChunk(max_chars=20, overlap_chars=5)
        for chunk in chunker.run("x" * 205):
            self.assertLessEqual(chunk["chars"], 20)

    def test_chars_is_the_length_it_claims(self):
        """A count carried beside the text is a second source of truth, and
        the cheap failure is that it stops matching."""
        for chunk in vectorChunk(max_chars=400).run(PROSE):
            self.assertEqual(chunk["chars"], len(chunk["text"]))


class TestOverlap(unittest.TestCase):

    def test_consecutive_chunks_share_their_boundary(self):
        chunks = vectorChunk(max_chars=200, overlap_chars=40,
                             merge_peers=False).run(PROSE)
        self.assertGreater(len(chunks), 2)
        overlapped = 0
        for before, after in zip(chunks, chunks[1:]):
            tail = before["text"][-40:].strip()
            if tail and tail.split()[-1] in after["text"][:80]:
                overlapped += 1
        self.assertGreater(overlapped, 0,
                           "no chunk carried its predecessor's tail")

    def test_overlap_does_not_break_the_ceiling(self):
        """The documented reason the window is trimmed: overlap is prepended
        *inside* the budget, not on top of it."""
        for chunk in vectorChunk(max_chars=150, overlap_chars=60).run(PROSE):
            self.assertLessEqual(chunk["chars"], 150)

    def test_an_overlap_at_least_as_wide_as_the_ceiling_is_refused(self):
        """Documented as a ValueError on construction *and* on run — because
        one of the two is the path a Chain takes."""
        with self.assertRaises(ValueError):
            vectorChunk(max_chars=100, overlap_chars=100)
        with self.assertRaises(ValueError):
            vectorChunk(max_chars=100, overlap_chars=120)


class TestTheEdgeCasesItNames(unittest.TestCase):
    """Each of these is a line in the module's "Edge cases" table. A
    documented edge case with no test is a claim."""

    def test_empty_string(self):
        self.assertEqual(vectorChunk().run(""), [])

    def test_whitespace_only(self):
        self.assertEqual(vectorChunk().run("   \n\n  \t "), [])

    def test_headings_only(self):
        self.assertEqual(vectorChunk().run("# One\n\n## Two\n\n### Three\n"), [])

    def test_an_unclosed_code_block_is_still_text(self):
        text = "# T\n\nBefore.\n\n```python\ndef f():\n    return 1\n"
        chunks = vectorChunk(max_chars=400).run(text)
        self.assertTrue(chunks)
        self.assertIn("def f()", " ".join(c["text"] for c in chunks))


class TestSeparators(unittest.TestCase):

    def test_a_code_block_within_budget_is_kept_whole(self):
        chunks = vectorChunk(max_chars=400).run(MARKDOWN)
        code = [c for c in chunks if "def hello()" in c["text"]]
        self.assertTrue(code)
        self.assertIn('return "world"', code[0]["text"])

    def test_nothing_is_lost_between_chunks(self):
        """The property that matters more than any boundary rule: a chunker
        that drops a sentence produces a store that answers confidently and
        wrongly."""
        words = PROSE.split()
        joined = " ".join(c["text"] for c in
                          vectorChunk(max_chars=120, overlap_chars=0,
                                      merge_peers=False).run(PROSE)).split()
        self.assertEqual([w for w in words if w not in joined], [])


if __name__ == "__main__":
    unittest.main()
