"""
Live tests are selected by a marker, and the selection cannot swallow others.

CI ran `pytest -k "not Live"`. `-k` is a case-insensitive substring match, so
two offline tests of the parameter matrix — `..._goes_undelivered` and
`..._not_a_delivery` — were deselected along with the live suite, and had not
run in CI since they were written. Nothing failed; they simply were not there.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conftest = _load("_conftest_under_test", ROOT / "tests" / "conftest.py")
summary = _load("live_summary", ROOT / "scripts" / "live_summary.py")


class TestTheMarkerFollowsTheClassName(unittest.TestCase):

    def test_a_live_class_is_marked(self):
        class TestOpenAILive: ...
        self.assertTrue(conftest.is_live(SimpleNamespace(cls=TestOpenAILive)))

    def test_a_name_that_merely_contains_live_is_not(self):
        class TestMatrix: ...
        item = SimpleNamespace(cls=TestMatrix, name="test_nothing_claimed_goes_undelivered")
        self.assertFalse(conftest.is_live(item))

    def test_a_module_level_function_is_not(self):
        self.assertFalse(conftest.is_live(SimpleNamespace(cls=None, name="test_live_thing")))


class TestWorkflowsSelectByMarker(unittest.TestCase):

    def test_no_workflow_selects_by_substring(self):
        for wf in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
            text = wf.read_text()
            with self.subTest(workflow=wf.name):
                self.assertNotIn('-k "not Live"', text)
                if "pytest" in text:
                    self.assertTrue('-m "not live"' in text or "-m live" in text,
                                    f"{wf.name} runs pytest without selecting by marker")


class TestTheLiveSummary(unittest.TestCase):

    XML = """<?xml version="1.0"?><testsuites><testsuite>
      <testcase classname="tests.clients.test_openai.TestOpenAILive" name="a"/>
      <testcase classname="tests.clients.test_openai.TestOpenAILive" name="b"><failure/></testcase>
      <testcase classname="tests.clients.test_xai.TestXAILive" name="c"><skipped/></testcase>
    </testsuite></testsuites>"""

    def _write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
        f.write(text); f.close()
        return f.name

    def test_counts_per_class(self):
        counts = summary.tally(self._write(self.XML))
        self.assertEqual(counts["TestOpenAILive"], {"passed": 1, "failed": 1, "skipped": 0})
        self.assertEqual(counts["TestXAILive"], {"passed": 0, "failed": 0, "skipped": 1})

    def test_a_run_where_nothing_passed_fails(self):
        skipped_only = self.XML.replace('name="a"/>', 'name="a"><skipped/></testcase>') \
                               .replace("<failure/>", "<skipped/>")
        self.assertEqual(summary.main([self._write(skipped_only)]), 1)

    def test_a_run_with_passes_succeeds(self):
        self.assertEqual(summary.main([self._write(self.XML)]), 0)


if __name__ == "__main__":
    unittest.main()
