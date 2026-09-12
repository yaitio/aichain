"""
The examples index is generated, and the examples that can run offline do.

`examples/README.md` described three examples as working for the whole 2.x
line; none of them could construct their agent. Two things let that stand: the
table was typed by hand, and nothing ever ran an example. This file holds both:

* the committed index — and the table in the root README — equal what
  `scripts/examples_index.py` produces from the scripts' own docstrings;
* every example that needs no key and no network is executed, and its output
  checked for the thing it exists to show.

Binding (right constructor arguments) is `test_lightness_invariant`'s job.
"""

import importlib.util
import io
import runpy
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "examples_index", ROOT / "scripts" / "examples_index.py")
examples_index = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(examples_index)


class TestTheIndexIsGenerated(unittest.TestCase):

    def test_committed_pages_equal_the_generated_ones(self):
        for path, text in examples_index.targets().items():
            with self.subTest(page=path.relative_to(ROOT).as_posix()):
                self.assertEqual(
                    path.read_text(), text,
                    "stale — run: python scripts/examples_index.py")

    def test_every_example_has_a_row(self):
        index = examples_index.render_index()
        for script in sorted((ROOT / "examples").glob("*.py")):
            self.assertIn(f"[`{script.name}`]", index)

    def test_a_missing_keys_block_is_an_error_not_a_blank_cell(self):
        # The failure this generator replaced was a cell nobody could check.
        # A script that does not say what it needs must stop the generator.
        fake = ROOT / "examples" / "__probe_missing_block.py"
        try:
            fake.write_text('"""99_probe.py — nothing here."""\n')
            with self.assertRaises(examples_index.IndexError_):
                examples_index.keys(fake)
        finally:
            fake.unlink()


# Examples that need no key, no server and no network. Adding one here is how
# it earns a place in the suite; an example that stops running fails the build.
OFFLINE = {
    "18_chain_external_trigger.py": [
        "⏸  Paused — A manager must approve the refund.",
        "💳  Refunded $42.0 for order A-123.",
        "{'approved': False, 'skipped': True}",
    ],
    "20_observability.py": [
        "approval.requested",
        "refunds over $50 need a ticket number",
        "IssueRefund.run() executed — refunded $42",
    ],
}


class TestOfflineExamplesRun(unittest.TestCase):

    def _run(self, name, tmp):
        import os
        os.environ["AICHAIN_RUNS"] = tmp
        out = io.StringIO()
        started = time.monotonic()
        try:
            with redirect_stdout(out):
                runpy.run_path(str(ROOT / "examples" / name), run_name="__main__")
        finally:
            os.environ.pop("AICHAIN_RUNS", None)
        return out.getvalue(), time.monotonic() - started

    def test_each_runs_and_shows_what_it_is_for(self):
        import logging
        import tempfile
        for name, expected in OFFLINE.items():
            with self.subTest(example=name), tempfile.TemporaryDirectory() as tmp:
                root = logging.getLogger()
                handlers = list(root.handlers)
                try:
                    output, seconds = self._run(name, tmp)
                finally:
                    # 20 calls logging.basicConfig; do not leak it into the suite.
                    for h in root.handlers[:]:
                        if h not in handlers:
                            root.removeHandler(h)
                for line in expected:
                    self.assertIn(line, output)
                self.assertLess(seconds, 5)

    def test_a_refused_refund_never_executes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            output, _ = self._run("20_observability.py", tmp)
        refused = output.split("=== $500")[1]
        self.assertNotIn("IssueRefund.run() executed", refused)


if __name__ == "__main__":
    unittest.main()
