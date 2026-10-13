"""
The local tools stay inside what they were given.

On a timeout only ``/bin/sh`` was killed — a ``&`` child or a pipeline lived
on — and the output so far was lost; a model's timeout was unbounded; the
child inherited every provider key, which ``env`` printed; one invalid byte
raised after the command had run; ``local_write`` emptied the file before
encoding, so a failure lost it; browse ended on a dangling link and showed
the host's path; ``end_line=-1`` read nothing.
"""

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.utilities import Shell


@unittest.skipIf(sys.platform == "win32", "POSIX process groups")
class TestRun(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.sh = Shell(self.dir)

    def test_a_timeout_kills_the_group_and_keeps_the_output(self):
        marker = os.path.join(self.dir, "survived")
        out = self.sh.run.run(f"echo started; (sleep 3; touch {marker}) & sleep 5",
                              {"timeout": 1})
        self.assertEqual(out["returncode"], -1)
        self.assertIn("started", out["stdout"])
        time.sleep(3.5)
        self.assertFalse(os.path.exists(marker))

    def test_the_child_does_not_inherit_keys(self):
        os.environ["SOME_PROVIDER_API_KEY"] = "sk-secret"
        try:
            out = self.sh.run.run("env")
        finally:
            del os.environ["SOME_PROVIDER_API_KEY"]
        self.assertNotIn("sk-secret", out["stdout"])
        self.assertIn("PATH=", out["stdout"])

    def test_invalid_bytes_are_replaced_not_raised(self):
        out = self.sh.run.run("printf 'ok \\377\\n'")
        self.assertIn("ok", out["stdout"])

    def test_a_huge_timeout_is_capped(self):
        self.assertEqual(self.sh.run.run("true", {"timeout": 10**9})["returncode"], 0)


class TestFiles(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.sh = Shell(self.dir)

    def test_a_write_that_cannot_encode_keeps_the_old_file(self):
        self.sh.write.run("old", {"path": "f.txt"})
        with self.assertRaises(UnicodeEncodeError):
            self.sh.write.run("é", {"path": "f.txt", "encoding": "ascii"})
        with open(os.path.join(self.dir, "f.txt")) as fh:
            self.assertEqual(fh.read(), "old")

    def test_a_bad_mode_is_refused(self):
        with self.assertRaises(ValueError):
            self.sh.write.run("x", {"path": "f.txt", "mode": "x"})

    def test_browse_survives_a_dangling_link_and_hides_the_host_path(self):
        os.symlink(os.path.join(self.dir, "nowhere"), os.path.join(self.dir, "dangling"))
        listing = self.sh.browse.run(".")
        self.assertIn("dangling", listing)
        self.assertNotIn(self.dir, listing)

    def test_end_line_minus_one_reads_to_the_end(self):
        self.sh.write.run("a\nb\nc\n", {"path": "f.txt"})
        self.assertIn("c", self.sh.read.run("f.txt", {"end_line": -1}))


if __name__ == "__main__":
    unittest.main()
