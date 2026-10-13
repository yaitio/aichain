"""
A run document carries its format, so a later release can read it.

`to_dict`/`from_dict` had no version: a format change could only break every
saved run. A document written before the field is format 0 and migrates; one
from a newer release is refused by name rather than misread.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.state import RunDocument
from yait_aichain.state._run_document import FORMAT


class TestFormat(unittest.TestCase):

    def test_a_document_is_written_with_its_format(self):
        self.assertEqual(RunDocument.new("chain", ["a"]).to_dict()["format"], FORMAT)

    def test_an_older_document_is_read(self):
        old = RunDocument.new("chain", ["a"]).to_dict()
        old.pop("format")
        self.assertEqual(RunDocument.from_dict(old).to_dict()["format"], FORMAT)

    def test_a_newer_document_is_refused_by_name(self):
        newer = {**RunDocument.new("chain", ["a"]).to_dict(), "format": FORMAT + 1}
        with self.assertRaisesRegex(ValueError, "newer"):
            RunDocument.from_dict(newer)


if __name__ == "__main__":
    unittest.main()
