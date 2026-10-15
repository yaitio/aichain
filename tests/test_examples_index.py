"""
The examples index is generated, and the examples that can run offline do.

`examples/README.md` described three examples as working for the whole 2.x
line; none of them could construct their agent. Two things let that stand: the
table was typed by hand, and nothing ever ran an example. This file holds both:

* the committed index — and the table in the root README — equal what
  `.github/scripts/examples_index.py` produces from the scripts' own docstrings;
* running the examples is `test_examples_run`'s job.

Binding (right constructor arguments) is `test_lightness_invariant`'s job.
"""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "examples_index", ROOT / ".github" / "scripts" / "examples_index.py")
examples_index = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(examples_index)


class TestTheIndexIsGenerated(unittest.TestCase):

    def test_committed_pages_equal_the_generated_ones(self):
        for path, text in examples_index.targets().items():
            with self.subTest(page=path.relative_to(ROOT).as_posix()):
                self.assertEqual(
                    path.read_text(), text,
                    "stale — run: python .github/scripts/examples_index.py")

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


if __name__ == "__main__":
    unittest.main()
