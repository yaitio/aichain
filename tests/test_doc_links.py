"""
Every relative link in the documentation leads to a file that exists.

Written for the move of 2026-10-10, when the pages left `primitives/`,
`getting-started/` and `tools-reference/` for folders named after what a reader
builds; the move rewrote every link it could compute, and this is what says it
missed none, and that no later edit breaks one.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = (sorted((ROOT / "docs").rglob("*.md")) + sorted((ROOT / "design").glob("*.md"))
         + [ROOT / "README.md", ROOT / "examples" / "README.md", ROOT / "SKILL.md"])
LINK = re.compile(r"\]\(([^)\s]+)\)")


def _broken(page: Path):
    text = re.sub(r"```.*?```", "", page.read_text(), flags=re.S)
    for target in LINK.findall(text):
        path = target.split("#", 1)[0]
        if not path or re.match(r"^[a-z][a-z0-9+.-]*:", path):
            continue
        if not (page.parent / path).exists():
            yield target


class TestDocumentationLinks(unittest.TestCase):

    def test_the_pages_are_found(self):
        self.assertGreater(len(PAGES), 40)

    def test_every_relative_link_resolves(self):
        for page in PAGES:
            broken = list(_broken(page))
            with self.subTest(page=page.relative_to(ROOT).as_posix()):
                self.assertEqual(broken, [])


if __name__ == "__main__":
    unittest.main()
