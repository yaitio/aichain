"""
The shared guards for URLs and output paths a model may choose.

Until 4.0 these were exercised through `tools.convert`, and its tests went with
it; nothing else held them, and coverage of the module fell below the register's
line without anyone deciding so. Pinned here directly, offline: a literal IP
needs no DNS.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from yait_aichain.tools._security import assert_safe_url, assert_url_scheme, confine_output_path, within


class TestURLs(unittest.TestCase):

    def test_only_http_and_https(self):
        assert_url_scheme("https://example.com")
        assert_url_scheme("http://example.com")
        for url in ("file:///etc/passwd", "gopher://x", "example.com"):
            with self.subTest(url), self.assertRaisesRegex(ValueError, "Refusing non-http"):
                assert_url_scheme(url)

    def test_a_private_address_is_refused(self):
        for url in ("http://127.0.0.1/", "http://10.0.0.5/x", "http://169.254.169.254/latest/meta-data"):
            with self.subTest(url), patch.dict(os.environ, {"AICHAIN_ALLOW_PRIVATE_URLS": ""}):
                with self.assertRaisesRegex(ValueError, "non-public address"):
                    assert_safe_url(url)

    def test_private_is_allowed_when_said(self):
        assert_safe_url("http://127.0.0.1/", allow_private=True)
        with patch.dict(os.environ, {"AICHAIN_ALLOW_PRIVATE_URLS": "1"}):
            assert_safe_url("http://127.0.0.1/")

    def test_a_url_needs_a_host(self):
        with self.assertRaisesRegex(ValueError, "no host"):
            assert_safe_url("http:///path")


class TestWithin(unittest.TestCase):

    def test_by_components_not_by_prefix(self):
        self.assertTrue(within("/tmp/a", "/tmp/a"))
        self.assertTrue(within("/tmp/a", "/tmp/a/b/c"))
        self.assertFalse(within("/tmp/a", "/tmp/a-evil"))
        self.assertFalse(within("/tmp/a", "/tmp"))

    def test_the_filesystem_root_contains_everything(self):
        # A prefix check made every path need to start with '//'.
        self.assertTrue(within("/", "/etc/hosts"))

    def test_a_shell_at_the_root_reads(self):
        from yait_aichain.utilities import Shell
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "f.txt"), "w") as fh:
            fh.write("hello\n")
        self.assertIn("hello", Shell("/").read.run(os.path.join(d, "f.txt")))


class TestOutputPaths(unittest.TestCase):

    def test_without_a_root_the_path_is_returned(self):
        with patch.dict(os.environ, {"AICHAIN_OUTPUT_ROOT": ""}):
            self.assertEqual(confine_output_path("a/b.mp3"), os.path.realpath("a/b.mp3"))

    def test_a_root_confines(self):
        root = tempfile.mkdtemp()
        with patch.dict(os.environ, {"AICHAIN_OUTPUT_ROOT": root}):
            inside = os.path.join(root, "out", "a.mp3")
            self.assertEqual(confine_output_path(inside), os.path.realpath(inside))
            self.assertEqual(confine_output_path(root), os.path.realpath(root))
            for outside in (os.path.join(root, "..", "escape.mp3"), "/etc/passwd", root + "-sibling/x"):
                with self.subTest(outside), self.assertRaisesRegex(PermissionError, "outside"):
                    confine_output_path(outside)


if __name__ == "__main__":
    unittest.main()
