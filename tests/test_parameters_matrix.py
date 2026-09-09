"""
The parameter matrix is a promise about behaviour, and this is where it is
held to.

`scripts/parameters.py` builds a request for every (option, model) pair and
records what happened to the option. Two things are enforced here:

1. **Behaviour matches the committed snapshot.** Any cell that moves fails
   the run with a readable diff. A change in how an option is handled then
   has to be regenerated and reviewed as a diff in the snapshot — it cannot
   happen by accident in a family client nobody was looking at.

2. **The number of unreported cells only goes down.** A cell counts against
   the ceiling when the library changed the request and said nothing — or
   said the wrong thing, which is worse. Describing a conversion as "this
   provider has no such control" was briefly happening to the three providers
   that honour `reasoning` best, so the check requires a notice whose kind
   matches what actually happened, and one that names the replacement when
   there is one.

3. **Nothing the provider data claims goes undelivered.** A provider
   declaring a control that none of its models honours means either the
   declaration is wrong or the option was never wired — the matrix cannot say
   which, only that they disagree, and that is enough to look.

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

#: 69 of 116 the day the matrix was first built; 0 once the notice channel
#: landed, and still 0 across 174 cells after the probe was widened to the
#: edits path. Lower it as cells are fixed. Never raise it.
SILENT_CEILING = 0


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

    def test_nothing_claimed_goes_undelivered(self):
        gaps = matrix.unfulfilled(self.live)
        self.assertEqual(gaps, [], "\n".join(
            f"{p} claims {o} and no probed model delivers it" for p, o in gaps))

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


class TestTheDocumentAnswersAllThree(unittest.TestCase):
    """The page has to carry three things, not one.

    It began as "what happens when you ask", which is only the last
    question. A reader also needs to know what there is to ask for and what
    each provider claims to take — and all three have to come out of the same
    run, or the page drifts from the code exactly as the hand-written one did
    ("`top_k` — provider-dependent").
    """

    @classmethod
    def setUpClass(cls):
        cls.doc = matrix.DOC.read_text()

    def test_it_says_what_is_sent_when_nothing_is_asked_for(self):
        """The divergence is the point of the section: a reader comparing two
        providers with no options set is not holding sampling constant, and
        the page has to say so where they are looking."""
        self.assertIn("## 0. What you get when you ask for nothing", self.doc)
        for value in ("`0.0`", "`0.2`", "`0.7`", "`2048`", "`32768`"):
            self.assertIn(value, self.doc)

    def test_it_lists_what_can_be_asked_for(self):
        self.assertIn("## 1. What you can ask for", self.doc)
        for option in ("temperature", "reasoning", "aspect_ratio"):
            self.assertIn(f"`{option}`", self.doc)

    def test_it_lists_what_each_provider_declares(self):
        self.assertIn("## 2. What each provider declares it takes", self.doc)

    def test_it_lists_what_happens(self):
        self.assertIn("## 3. What happens when you ask", self.doc)

    def test_it_shows_the_value_scales(self):
        """A universal name is half the promise; the reader needs the other
        half — what `medium` becomes on each provider."""
        self.assertIn("own scale", self.doc)
        self.assertIn("`medium`→`10000`", self.doc)

    def test_a_range_is_not_shown_as_two_choices(self):
        """`output_compression` is 0-100, not the set {0, 100}."""
        self.assertIn("`0`–`100`", self.doc)

    def test_it_is_marked_as_generated(self):
        self.assertIn("GENERATED by scripts/parameters.py", self.doc)


class TestBothDirections(unittest.TestCase):
    """The declaration and the code have to agree, each way.

    Only one direction was checked at first — a provider claiming a control
    no model honours. The other went unnoticed until a rename dropped
    `aspect_ratio` from OpenAI's declaration while the client went on
    converting it: the library did more than the documentation said, and the
    page told a reader a working option did not work. An under-claim is
    quieter than an over-claim and no less wrong.
    """

    def test_a_refusal_is_not_a_delivery(self):
        """Perplexity raises on `reasoning`; counting that as delivered made
        it look like an undeclared capability."""
        live = matrix.build_matrix()
        self.assertNotIn(
            ("perplexity", "reasoning", "delivered, declared by nobody"),
            matrix.unfulfilled(live))

    def test_the_page_names_both_kinds_when_there_are_any(self):
        """The section is rendered only when something disagrees, so the
        page is asserted against a matrix that has a disagreement in it
        rather than against today's clean one."""
        # render_doc walks the full option set per model, so the disagreement
        # is injected into a real row rather than a stub one.
        live = matrix.build_matrix()
        live["gpt-4o"] = {**live["gpt-4o"],
                          "top_k": {"outcome": "passed", "said": [],
                                    "detail": ""}}
        rendered = matrix.render_doc(live)
        self.assertIn("Where the declaration and the code disagree", rendered)
        self.assertIn("delivered, declared by nobody", rendered)
