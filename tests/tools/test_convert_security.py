"""
Output-path confinement for the convert tools.

convertToHTML must write through confine_output_path (like its siblings), so an
out-of-root output_path is rejected and the file lands at the confined path.
The markdown conversion itself is mocked — this only exercises the write path.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tempfile
from tools.convert.to_html import convertToHTML


class TestConvertToHtmlConfinement(unittest.TestCase):

    def setUp(self):
        self._root = tempfile.mkdtemp()
        os.environ["AICHAIN_OUTPUT_ROOT"] = self._root

    def tearDown(self):
        os.environ.pop("AICHAIN_OUTPUT_ROOT", None)

    def _tool(self):
        t = convertToHTML()
        # bypass the optional mistletoe dependency
        t._convert = staticmethod(lambda text, fmt: "<p>x</p>")
        return t

    def test_out_of_root_path_is_rejected(self):
        with self.assertRaises(PermissionError):
            self._tool().run("# hi", options={"output_path": "/etc/evil.html"})

    def test_in_root_path_writes_to_confined_location(self):
        target = os.path.join(self._root, "sub", "out.html")
        self._tool().run("# hi", options={"output_path": target})
        self.assertTrue(os.path.exists(target))
        with open(target) as fh:
            self.assertIn("<p>x</p>", fh.read())


if __name__ == "__main__":
    unittest.main()
