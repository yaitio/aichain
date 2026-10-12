"""
What an install brings with it.

Provider data is TOML, and the standard library reads TOML only from Python
3.11. On 3.10 the package needs `tomli`, and from 1.2.3 to 4.0.0 nothing
declared it: a bare `pip install yait-aichain` on 3.10 failed at import. The
suite never noticed, because pytest depends on `tomli` itself on that version —
so a test that only imports the package proves nothing here. This one reads
what the package declares.
"""

import os
import re
import sys
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")

if sys.version_info >= (3, 11):
    import tomllib
else:                                   # pragma: no cover
    import tomli as tomllib


class TestDeclaredDependencies(unittest.TestCase):

    def setUp(self):
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
            self.project = tomllib.load(fh)["project"]

    def test_the_floor_python_gets_a_toml_reader(self):
        floor = re.match(r">=\s*(\d+)\.(\d+)", self.project["requires-python"])
        self.assertIsNotNone(floor)
        if (int(floor.group(1)), int(floor.group(2))) >= (3, 11):
            self.skipTest("every supported Python has tomllib")
        tomli = [d for d in self.project["dependencies"] if d.replace(" ", "").startswith("tomli")]
        self.assertEqual(len(tomli), 1, "tomli must be declared while 3.10 is supported")
        self.assertIn("python_version<'3.11'", tomli[0].replace(" ", "").replace('"', "'"))

    def test_the_library_reads_toml_only_through_the_declared_names(self):
        # A third TOML reader would need a third declaration.
        src = open(os.path.join(ROOT, "yait_aichain", "models", "_data.py")).read()
        self.assertEqual(set(re.findall(r"import (tomllib|tomli)\b", src)), {"tomllib", "tomli"})


class TestWhatShips(unittest.TestCase):
    """A type checker reads the annotations, a star import brings the public
    names, and the sdist carries no tests it cannot run."""

    def test_the_package_says_it_is_typed(self):
        self.assertTrue(os.path.exists(os.path.join(ROOT, "yait_aichain", "py.typed")))
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
            data = tomllib.load(fh)["tool"]["setuptools"]["package-data"]
        self.assertIn("py.typed", data.get("yait_aichain", []))

    def test_a_star_import_is_the_public_names(self):
        import yait_aichain
        ns: dict = {}
        exec("from yait_aichain import *", ns)
        names = {k for k in ns if not k.startswith("__")}
        self.assertEqual(names, set(yait_aichain.__all__))
        self.assertIn("AgentResult", names)
        self.assertFalse(names & {"agent", "chain", "clients", "models", "pool", "skills", "state", "tools"})

    def test_the_sdist_leaves_the_tests_out(self):
        with open(os.path.join(ROOT, "MANIFEST.in")) as fh:
            self.assertIn("prune tests", fh.read())


if __name__ == "__main__":
    unittest.main()
