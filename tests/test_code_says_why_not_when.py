"""
The code says why, the CHANGELOG says when.

A comment carrying the history of a line — "until 2026-09-12 this was…", "it
used to…" — is read by every agent that loads the module, at its price in
tokens, and goes stale the day after it is true. The reason a line is the way
it is belongs beside it, in the present tense; the story of how it got there
belongs in the CHANGELOG, which is dated and searchable.
"""

import re
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "yait_aichain"
HISTORY = re.compile(
    r"\buntil 20\d\d-\d\d-\d\d|\bsince 20\d\d-\d\d-\d\d|\bbefore 20\d\d-\d\d-\d\d"
    r"|\bit used to\b|\bused to be (accepted|read|sent|dropped|ignored)\b|\bwhich this did until\b")


class TestNoHistoryInTheCode(unittest.TestCase):

    def test_no_line_tells_its_own_story(self):
        found = [f"{p.relative_to(PACKAGE.parent)}:{i}: {line.strip()[:90]}"
                 for p in sorted(PACKAGE.rglob("*.py"))
                 for i, line in enumerate(p.read_text().split("\n"), 1)
                 if HISTORY.search(line)]
        self.assertEqual(found, [], "move the history to the CHANGELOG and keep the reason:\n"
                         + "\n".join(found))


if __name__ == "__main__":
    unittest.main()
