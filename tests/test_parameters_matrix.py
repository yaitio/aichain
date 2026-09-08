"""
The parameter matrix is a promise about behaviour, and this is where it is
held to.

`scripts/parameters.py` builds a request for every (option, model) pair and
records what happened to the option. Two things are enforced here:

1. **Behaviour matches the committed snapshot.** Any cell that moves fails
   the run with a readable diff. A change in how an option is handled then
   has to be regenerated and reviewed as a diff in the snapshot — it cannot
   happen by accident in a family client nobody was looking at.

2. **The number of silent cells only goes down.** A silent cell is an option
   the caller set and did not get, with nothing said. The library may adapt a
   request to a provider; it may not do so quietly. The ceiling below is
   lowered as cells are fixed and is never raised.

Regenerate with `python scripts/parameters.py`, then lower SILENT_CEILING to
the number it prints.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import parameters as matrix                                       # noqa: E402

#: Measured 2026-09-10, the day the matrix was first built: 69 of 116 cells.
#: Lower it as they are fixed. Never raise it.
SILENT_CEILING = 69


class TestMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.live = matrix.build_matrix()
        cls.snapshot = json.loads(matrix.SNAPSHOT.read_text())

    def test_behaviour_matches_the_snapshot(self):
        moved = [(m, p, self.snapshot.get(m, {}).get(p), c)
                 for m in self.live for p, c in self.live[m].items()
                 if self.snapshot.get(m, {}).get(p) != c]
        self.assertEqual(moved, [], "\n".join(
            f"{m} · {p}: {was} → {now}" for m, p, was, now in moved)
            + "\n\nBehaviour moved. If intended, run scripts/parameters.py "
              "and review the snapshot diff.")

    def test_no_model_or_option_was_dropped_from_the_matrix(self):
        """Shrinking the matrix is how a defect leaves the table unseen."""
        self.assertEqual(set(self.live), set(self.snapshot))
        for m in self.live:
            self.assertEqual(set(self.live[m]), set(self.snapshot[m]), m)

    def test_silent_cells_do_not_increase(self):
        silent = [(m, p) for m in self.live for p, c in self.live[m].items()
                  if matrix.verdict(c) != "ok"]
        self.assertLessEqual(
            len(silent), SILENT_CEILING,
            f"{len(silent)} silent cells, ceiling is {SILENT_CEILING}. A new "
            "option is being adapted without saying so:\n"
            + "\n".join(f"  {m} · {p}" for m, p in silent))

    def test_the_ceiling_is_not_slack(self):
        """When cells get fixed the ceiling must come down with them, or the
        ratchet stops holding anything."""
        silent = sum(1 for m in self.live for c in self.live[m].values()
                     if matrix.verdict(c) != "ok")
        self.assertEqual(
            silent, SILENT_CEILING,
            f"{silent} silent cells but the ceiling is {SILENT_CEILING} — "
            "lower it to match.")


if __name__ == "__main__":
    unittest.main()
