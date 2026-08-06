"""
Eval — the measuring instrument, and the ways instruments lie.

Every test here pins a behaviour that exists because the alternative already
cost something:

  * numbers are recounted from raw records, never carried as a tally;
  * ``Pass^k`` and the mean disagree, and both are reported;
  * ``flips`` shows the shape neither of them does;
  * a cell that cannot be trusted is **absent**, not zero;
  * ratios are pooled across cases, so silence cannot score as precision;
  * a scorer that cannot decide does not decide;
  * a finished attempt is never re-run, and the ledger outlives the process.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

# Full path, not the legacy top-level alias: ``eval`` is a builtin name and
# aliasing it would shadow it for every test in the process.
from yait_aichain.eval import (Case, Eval, Record, Report, adapt, contains,
                               exact, normalise, numeric, read, regex, se)


def _rec(arm, case, trial, ok, **kw):
    return Record(arm=arm, case=case, trial=trial, ok=ok, **kw)


class TestScorers(unittest.TestCase):

    def test_exact_collapses_whitespace(self):
        c = Case("1", expect="Oslo is  the capital")
        self.assertTrue(normalise(exact()(c, "oslo is the capital"))[0])

    def test_contains_gives_partial_credit(self):
        # All-or-nothing throws away the difference between three of four and
        # none of four, which is most of what a benchmark is for.
        c = Case("1", expect=["a", "b", "c", "d"])
        ok, score, _ = normalise(contains()(c, "a b c"))
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.75)

    def test_contains_any_passes_on_one_hit(self):
        c = Case("1", expect=["a", "z"])
        self.assertTrue(normalise(contains(all_of=False)(c, "a"))[0])

    def test_numeric_within_tolerance(self):
        c = Case("1", expect=42.0)
        self.assertTrue(normalise(numeric(tolerance=0.5)(c, "about 42.3 units"))[0])
        self.assertFalse(normalise(numeric(tolerance=0.1)(c, "about 42.3"))[0])

    def test_regex(self):
        c = Case("1", expect=r"INV-\d{4}")
        self.assertTrue(normalise(regex()(c, "see INV-1234 attached"))[0])

    def test_a_scorer_returning_none_is_a_loud_error(self):
        # Silently marking every case failed reads as a bad model, and the
        # model is not what is broken.
        with self.assertRaises(TypeError):
            normalise(None)


class TestPassK(unittest.TestCase):

    def test_pass_k_needs_every_trial(self):
        recs = [_rec("a", "c1", 1, True), _rec("a", "c1", 2, True),
                _rec("a", "c2", 1, True), _rec("a", "c2", 2, False)]
        r = Report(recs)
        self.assertAlmostEqual(r.mean("a"), 0.75)      # three of four attempts
        self.assertAlmostEqual(r.pass_k("a", 2), 0.5)  # one of two cases

    def test_undersampled_case_is_skipped_not_failed(self):
        # An interrupted run is under-sampled, not wrong. Counting the missing
        # trial as a failure makes a crash look like a bad participant.
        recs = [_rec("a", "c1", 1, True), _rec("a", "c1", 2, True),
                _rec("a", "c2", 1, True)]
        self.assertAlmostEqual(Report(recs).pass_k("a", 2), 1.0)

    def test_mean_and_pass_k_can_rank_arms_differently(self):
        # The finding from the A+ run: the arm with the better mean had the
        # worse Pass^3. Reporting one number would have ranked them backwards.
        steady = [_rec("steady", f"c{i}", t, i < 2)
                  for i in range(4) for t in (1, 2, 3)]
        lucky  = [_rec("lucky", f"c{i}", t, (i + t) % 3 != 0
                       or i == 0)
                  for i in range(4) for t in (1, 2, 3)]
        r = Report(steady + lucky)
        self.assertGreater(r.mean("lucky"), r.mean("steady"))
        self.assertGreater(r.pass_k("steady", 3), r.pass_k("lucky", 3))


class TestFlips(unittest.TestCase):

    def test_reproducible_arm_never_flips(self):
        recs = [_rec("a", f"c{i}", t, i == 0) for i in range(3) for t in (1, 2, 3)]
        self.assertEqual(Report(recs).flips("a"), 0.0)

    def test_flips_catches_what_the_mean_hides(self):
        # Both arms answer half the attempts correctly. One solves a fixed
        # half every time; the other solves a different half each run. Only
        # the second is unreliable, and only `flips` says so.
        fixed = [_rec("fixed", f"c{i}", t, i % 2 == 0)
                 for i in range(4) for t in (1, 2)]
        roving = [_rec("roving", f"c{i}", t, (i + t) % 2 == 0)
                  for i in range(4) for t in (1, 2)]
        r = Report(fixed + roving)
        self.assertAlmostEqual(r.mean("fixed"), r.mean("roving"))
        self.assertEqual(r.flips("fixed"), 0.0)
        self.assertEqual(r.flips("roving"), 1.0)

    def test_zero_flips_includes_reproducibly_wrong(self):
        recs = [_rec("a", f"c{i}", t, False) for i in range(3) for t in (1, 2)]
        r = Report(recs)
        self.assertEqual(r.flips("a"), 0.0)
        self.assertEqual(r.mean("a"), 0.0)     # which is why it is never read alone


class TestValidity(unittest.TestCase):

    def test_a_rejected_cell_is_absent_not_zero(self):
        good = [_rec("good", f"c{i}", 1, True) for i in range(6)]
        bad  = [_rec("bad", f"c{i}", 1, False, output="") for i in range(6)]
        r = Report(good + bad)
        why = r.reject(max_empty=2)
        self.assertIn("bad", why)
        self.assertNotIn("bad", r.arms())          # absent from every figure
        self.assertEqual(r.mean("bad"), 0.0)       # nothing to count, not a score
        self.assertIn("rejected", r.table())

    def test_errors_over_threshold_reject(self):
        recs = [_rec("a", f"c{i}", 1, False, error="RateLimitError")
                for i in range(6)]
        self.assertIn("a", Report(recs).reject(max_errors=5))

    def test_clean_cell_survives(self):
        recs = [_rec("a", f"c{i}", 1, True, output="x") for i in range(6)]
        r = Report(recs)
        self.assertEqual(r.reject(), {})
        self.assertEqual(r.arms(), ["a"])


class TestPooling(unittest.TestCase):

    def test_ratios_pool_across_cases(self):
        # Averaging per-case ratios rewards silence: a case answered with
        # nothing has no wrong answers in it. Pooled, the arm that answered
        # twice and got one right is at 0.5, not at 0.75.
        recs = [_rec("a", "c1", 1, True), _rec("a", "c2", 1, False),
                _rec("a", "c3", 1, True), _rec("a", "c4", 1, False)]
        self.assertAlmostEqual(Report(recs).mean("a"), 0.5)


class TestGroups(unittest.TestCase):

    def test_groups_are_reported_apart(self):
        recs = ([_rec("a", f"p{i}", 1, True, group="PASS") for i in range(3)]
                + [_rec("a", f"f{i}", 1, False, group="FAIL") for i in range(3)])
        r = Report(recs)
        self.assertEqual(r.groups(), ["FAIL", "PASS"])
        self.assertEqual(r.mean("a", "PASS"), 1.0)
        self.assertEqual(r.mean("a", "FAIL"), 0.0)
        self.assertEqual(r.mean("a"), 0.5)          # pooled hides both
        self.assertIn("PASS", r.by_group())


class TestPaired(unittest.TestCase):

    def test_only_discordant_cases_carry_signal(self):
        recs = []
        for i in range(10):                       # both arms agree everywhere
            recs += [_rec("a", f"c{i}", 1, True), _rec("b", f"c{i}", 1, True)]
        for i in range(10, 18):                   # a wins eight
            recs += [_rec("a", f"c{i}", 1, True), _rec("b", f"c{i}", 1, False)]
        got = Report(recs).paired("a", "b")
        self.assertEqual((got["only_a"], got["only_b"], got["tied"]), (8, 0, 10))
        self.assertTrue(got["enough"])
        self.assertLess(got["p"], 0.01)

    def test_too_few_discordant_pairs_says_so(self):
        recs = []
        for i in range(3):
            recs += [_rec("a", f"c{i}", 1, True), _rec("b", f"c{i}", 1, False)]
        got = Report(recs).paired("a", "b")
        self.assertFalse(got["enough"])

    def test_cases_one_arm_never_attempted_are_dropped(self):
        recs = [_rec("a", "c1", 1, True), _rec("a", "c2", 1, True),
                _rec("b", "c1", 1, False)]
        self.assertEqual(Report(recs).paired("a", "b")["discordant"], 1)


class TestControls(unittest.TestCase):

    def test_a_sound_instrument_passes(self):
        oracle = Report([_rec("o", f"c{i}", 1, True) for i in range(10)])
        noise  = Report([_rec("n", f"c{i}", 1, False) for i in range(10)])
        self.assertTrue(Report.controls(oracle, noise)["ok"])

    def test_an_oracle_below_the_floor_condemns_the_metric(self):
        # If feeding the harness the right answers does not score ~1.0, the
        # metric is broken and every comparison built on it is biased.
        oracle = Report([_rec("o", f"c{i}", 1, i < 7) for i in range(10)])
        noise  = Report([_rec("n", f"c{i}", 1, False) for i in range(10)])
        got = Report.controls(oracle, noise)
        self.assertFalse(got["ok"])
        self.assertIn("oracle", got["why"])

    def test_noise_scoring_high_condemns_it_too(self):
        oracle = Report([_rec("o", f"c{i}", 1, True) for i in range(10)])
        noise  = Report([_rec("n", f"c{i}", 1, True) for i in range(10)])
        self.assertFalse(Report.controls(oracle, noise)["ok"])


class TestForeignData(unittest.TestCase):

    def test_someone_elses_rows_are_adaptable(self):
        # τ² and the retrieval bench own their own loops; reporting is where
        # the mistakes were, so reporting has to accept their shapes.
        rows = [{"agent": "langgraph", "task_id": "retail_5", "success": True,
                 "reward": 1.0, "domain": "retail"}]
        got = list(adapt(rows, arm="agent", case="task_id", ok="success"))
        self.assertEqual((got[0].arm, got[0].case, got[0].ok),
                         ("langgraph", "retail_5", True))
        self.assertEqual(got[0].meta["domain"], "retail")   # nothing dropped

    def test_report_of_dispatches_on_what_it_is_given(self):
        rows = [{"a": "x", "c": "1", "o": True}]
        r = Report.of(rows, arm="a", case="c", ok="o")
        self.assertEqual(r.arms(), ["x"])


class TestLedger(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "run.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _ev(self, arms, trials=1, cases=None):
        return Eval(cases or [Case("c1", expect="x"), Case("c2", expect="y")],
                    arms, exact(), trials=trials, out=self.out, verbose=False)

    def test_every_attempt_reaches_disk(self):
        self._ev({"a": lambda c: c.expect}).run()
        self.assertEqual(len(read(self.out)), 2)

    def test_a_finished_attempt_is_not_re_run(self):
        calls = []
        arm = lambda c: (calls.append(c.id), c.expect)[1]
        self._ev({"a": arm}).run()
        self._ev({"a": arm}).run()              # second pass finds the ledger
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(read(self.out)), 2)

    def test_resume_completes_only_what_is_missing(self):
        self._ev({"a": lambda c: c.expect}, trials=1).run()
        r = self._ev({"a": lambda c: c.expect}, trials=3).run()
        self.assertEqual(len(read(self.out)), 6)
        self.assertAlmostEqual(r.pass_k("a", 3), 1.0)

    def test_a_raising_arm_is_data_not_a_stop(self):
        def half(case):
            if case.id == "c1":
                raise RuntimeError("boom")
            return case.expect
        r = self._ev({"a": half}).run()
        recs = read(self.out)
        self.assertEqual(len(recs), 2)                    # both attempted
        self.assertIn("RuntimeError", [x.error[:12] for x in recs if x.error][0])
        self.assertEqual(r.spend("a")["errors"], 1)

    def test_a_scorer_that_cannot_decide_does_not_pass_the_case(self):
        def broken(case, output):
            raise ValueError("judge returned no verdict")
        ev = Eval([Case("c1", expect="x")], {"a": lambda c: "x"}, broken,
                  out=self.out, verbose=False)
        ev.run()
        rec = read(self.out)[0]
        self.assertFalse(rec.ok)
        self.assertIn("scorer:", rec.error)

    def test_arm_may_report_what_it_spent(self):
        arm = lambda c: {"output": c.expect, "cost": 0.25, "tokens": 100,
                         "note": "kept"}
        r = self._ev({"a": arm}).run()
        self.assertAlmostEqual(r.spend("a")["cost"], 0.5)
        self.assertEqual(read(self.out)[0].meta["note"], "kept")

    def test_duplicate_case_ids_are_refused_at_construction(self):
        # Two cases sharing an id merge silently, and Pass^k then counts a
        # case that was never attempted k times as if it had been.
        with self.assertRaises(ValueError):
            Eval([Case("same"), Case("same")], {"a": lambda c: ""}, exact(),
                 out=self.out, verbose=False)

    def test_the_ledger_survives_a_broken_last_line(self):
        self._ev({"a": lambda c: c.expect}).run()
        with open(self.out, "a", encoding="utf-8") as fh:
            fh.write('{"arm": "a", "case": trunc')      # a crash mid-write
        self.assertEqual(len(read(self.out)), 2)

    def test_output_defaults_somewhere_durable(self):
        ev = Eval([Case("c1")], {"a": lambda c: ""}, exact(), name="probe",
                  verbose=False)
        self.assertNotIn(tempfile.gettempdir(), str(ev.out))
        self.assertTrue(str(ev.out).endswith("eval-runs/probe.jsonl"))

    def test_smoke_does_not_contaminate_the_real_ledger(self):
        ev = self._ev({"a": lambda c: c.expect}, trials=3)
        ev.smoke(1)
        self.assertFalse(self.out.exists())
        self.assertTrue(self.out.with_suffix(".smoke.jsonl").exists())


class TestStatistics(unittest.TestCase):

    def test_standard_error_matches_the_published_figures(self):
        self.assertAlmostEqual(se(0.5, 184), 0.0369, places=3)
        self.assertAlmostEqual(se(0.5, 30), 0.0913, places=3)


if __name__ == "__main__":
    unittest.main()
