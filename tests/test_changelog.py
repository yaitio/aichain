"""
The changelog is a surface, and it had stopped being one.

Until 2026-09-09 the maintained changelog was excluded from git — `1.0.0` to
`1.6.0` had lived in the README, `2.0` onward in a file nobody outside the
working copy could see, and the README's copy had stopped at `1.6.0`. Three
minor releases shipped with no public record of what changed in them.

Two things are held here. A release must carry an entry — that is the whole
defect, and it is cheap to check. And the file must stay reachable: tracked
by git, so that publishing it is not something a `.gitignore` line can undo
by accident again.
"""

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"

_ENTRY = re.compile(r"^## \[(\d+\.\d+\.\d+)\](?: — (\d{4}-\d{2}(?:-\d{2})?))?",
                    re.MULTILINE)


def _version() -> str:
    text = (ROOT / "yait_aichain" / "__init__.py").read_text()
    return re.search(r'__version__ = "([^"]+)"', text).group(1)


class TestTheChangelogCoversWhatShipped(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.text = CHANGELOG.read_text()
        cls.entries = dict(_ENTRY.findall(cls.text))

    def test_the_current_version_has_an_entry(self):
        """The check that stops the gap growing. Seventeen releases between
        1.2.5 and 1.6.0 have none, which is recorded in the file itself
        rather than reconstructed from commit messages."""
        v = _version()
        self.assertIn(v, self.entries,
                      f"version {v} is about to ship with no changelog entry")

    def test_the_current_version_is_dated(self):
        """An undated entry is an entry still being written."""
        self.assertTrue(self.entries.get(_version()),
                        f"the {_version()} entry carries no date")

    def test_it_is_tracked_by_git(self):
        """It was excluded for months, so this is the thing that broke."""
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "CHANGELOG.md"],
            cwd=ROOT, capture_output=True)
        self.assertEqual(tracked.returncode, 0,
                         "CHANGELOG.md is not tracked — it cannot be read by "
                         "anyone who did not clone this working copy")

    def test_the_gap_before_2_0_is_named_rather_than_hidden(self):
        self.assertIn("## Before 2.0, incompletely", self.text)

    def test_entries_run_newest_first(self):
        order = [tuple(int(n) for n in v.split("."))
                 for v, _date in _ENTRY.findall(self.text)]
        self.assertEqual(order, sorted(order, reverse=True),
                         "entries are out of order; the top of the file is "
                         "where a reader looks for the newest release")


class TestTheReadmeDoesNotKeepASecondCopy(unittest.TestCase):
    """Two changelogs is how one of them goes stale — which is exactly what
    happened: the README's stopped at 1.6.0 and stayed there through 2.0,
    2.1, 2.2 and 2.3."""

    def test_the_readme_points_at_the_file(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn("[CHANGELOG.md](CHANGELOG.md)", readme)

    def test_the_readme_does_not_narrate_a_release(self):
        readme = (ROOT / "README.md").read_text()
        section = readme.split("## Changelog", 1)[1]
        self.assertNotRegex(section, r"^### \d+\.\d+\.\d+", )


if __name__ == "__main__":
    unittest.main()
