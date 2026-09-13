"""The coverage register is checked both ways: gaps need a reason, stale entries go."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("coverage_register", ROOT / "scripts" / "coverage_register.py")
reg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reg)


def _report(total, **files):
    return {"totals": {"percent_covered": total},
            "files": {f: {"summary": {"percent_covered": p}} for f, p in files.items()}}


class TestRegister(unittest.TestCase):

    ROWS = reg.register("""
| Module | Kind | Reason |
|---|---|---|
| `yait_aichain/tools/search/brave.py` | external | the request path needs Brave's API |
""")

    def test_rows_parse(self):
        self.assertEqual(self.ROWS["yait_aichain/tools/search/brave.py"][0], "external")

    def test_a_gap_without_a_reason_fails(self):
        found = reg.problems(_report(80, **{"yait_aichain/tools/search/brave.py": 30,
                                            "yait_aichain/tools/local/_run.py": 45}),
                             self.ROWS, 77)
        self.assertEqual(len(found), 1)
        self.assertIn("_run.py", found[0])

    def test_an_entry_that_reached_the_threshold_is_stale(self):
        found = reg.problems(_report(80, **{"yait_aichain/tools/search/brave.py": 90}), self.ROWS, 77)
        self.assertTrue(any("remove it" in p for p in found))

    def test_a_registered_module_that_is_gone_fails(self):
        found = reg.problems(_report(80), self.ROWS, 77)
        self.assertTrue(any("not measured" in p for p in found))

    def test_the_floor_holds(self):
        found = reg.problems(_report(70, **{"yait_aichain/tools/search/brave.py": 30}), self.ROWS, 77)
        self.assertTrue(any("below the floor" in p for p in found))

    def test_the_floor_is_read_from_pyproject(self):
        self.assertGreaterEqual(reg.fail_under(), 77)

    def test_the_committed_register_is_well_formed(self):
        rows = reg.register((ROOT / "COVERAGE.md").read_text())
        self.assertTrue(rows, "COVERAGE.md has no register rows")
        for module, (kind, reason) in rows.items():
            self.assertIn(kind, reg.KINDS, module)
            self.assertTrue(reason, module)
            self.assertTrue((ROOT / module).exists(), module)


if __name__ == "__main__":
    unittest.main()
