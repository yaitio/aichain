"""
The eval instrument counts what happened, not what is convenient to count.

Each test names a way the report used to mislead: a different k per arm, a
Pass^k read from the first k trials, a control passing with no data, a failed
run scored as an answer, a resumed ledger trusted after its cases changed, a
scorer reading the first number, an abstention lost inside ``all_of``, a score
of 0.01 counted a pass, and the string "false" read as true.
"""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.agent import AgentResult
from yait_aichain.eval import (Case, Eval, Record, Report, abstain, adapt,
                               all_of, ci, exact, normalise, numeric, read)


def _rec(arm, case, trial, ok, **kw):
    return Record(arm=arm, case=case, trial=trial, ok=ok, **kw)


class TestReportCounts(unittest.TestCase):

    def test_one_k_for_every_arm(self):
        # An interrupted arm with two trials was ranked at Pass^2 beside a
        # finished arm at Pass^3, and read as more reliable.
        recs = [_rec("full", "c", t, True) for t in range(3)]
        recs += [_rec("cut", "c", t, True) for t in range(2)]
        r = Report(recs)
        self.assertEqual(r.trials(), 3)
        self.assertTrue(math.isnan(r.pass_k("cut")))
        self.assertEqual(r.pass_k("full"), 1.0)

    def test_pass_k_uses_every_trial(self):
        # Two passes in four trials: C(2,2)/C(4,2) = 1/6, where the first two
        # trials alone gave 1 or 0 depending on the order.
        recs = [_rec("a", "c", t, ok) for t, ok in enumerate([True, True, False, False])]
        self.assertAlmostEqual(Report(recs).pass_k("a", 2), 1 / 6)

    def test_no_data_is_nan_not_zero(self):
        self.assertTrue(math.isnan(Report([]).pass_k("a", 1)))

    def test_controls_refuse_an_arm_with_no_data(self):
        oracle = Report([_rec("o", "c", 0, True)])
        with self.assertRaises(ValueError):
            Report.controls(oracle, Report([]), oracle_arm="o", noise_arm="n")

    def test_controls_read_differently_named_arms(self):
        oracle = Report([_rec("o", "c", 0, True)])
        noise = Report([_rec("n", "c", 0, False)])
        out = Report.controls(oracle, noise, oracle_arm="o", noise_arm="n")
        self.assertEqual((out["oracle"], out["noise"], out["ok"]), (1.0, 0.0, True))

    def test_wilson_interval_is_wide_at_the_edge(self):
        low, high = ci(10, 10)
        self.assertEqual(high, 1.0)
        self.assertLess(low, 0.8)

    def test_thresholds_are_fractions(self):
        recs = [_rec("a", f"c{i}", 0, False, output="") for i in range(2)]
        recs += [_rec("a", f"d{i}", 0, True, output="x") for i in range(8)]
        self.assertIn("a", Report(recs).reject(max_empty=0.1))
        self.assertNotIn("a", Report(recs).reject(max_empty=0.3))


class TestEvalRuns(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "run.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_failed_agent_result_is_an_error_not_an_answer(self):
        failed = AgentResult(output="partial", success=False, error="ceiling",
                             mode="auto", steps_taken=1, tokens_used=0)
        Eval([Case("c", expect="partial")], {"a": lambda c: failed}, exact(),
             out=self.out, verbose=False).run()
        rec = read(self.out)[0]
        self.assertFalse(rec.ok)
        self.assertIn("run failed", rec.error)

    def test_a_ledger_for_other_cases_is_not_resumed(self):
        calls = []
        arm = lambda c: (calls.append(c.id), c.expect)[1]
        Eval([Case("c", expect="x")], {"a": arm}, exact(), out=self.out, verbose=False).run()
        Eval([Case("c", expect="y")], {"a": arm}, exact(), out=self.out, verbose=False).run()
        self.assertEqual(len(calls), 2)

    def test_an_errored_attempt_is_retried(self):
        state = {"n": 0}
        def flaky(case):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("transient")
            return case.expect
        for _ in range(2):
            r = Eval([Case("c", expect="x")], {"a": flaky}, exact(),
                     out=self.out, verbose=False).run()
        self.assertEqual(state["n"], 2)
        self.assertEqual(r.mean("a"), 1.0)

    def test_concurrency_keeps_every_attempt(self):
        cases = [Case(f"c{i}", expect=str(i)) for i in range(8)]
        r = Eval(cases, {"a": lambda c: c.expect}, exact(), trials=2,
                 out=self.out, verbose=False).run(concurrency=4)
        self.assertEqual(len(read(self.out)), 16)
        self.assertEqual(r.pass_k("a"), 1.0)


class TestScorers(unittest.TestCase):

    def test_numeric_reads_the_last_number(self):
        c = Case("c", expect=42)
        self.assertTrue(normalise(numeric()(c, "Of the 7 items, the total is 42"))[0])

    def test_numeric_reads_thousands(self):
        self.assertTrue(normalise(numeric()(Case("c", expect=1234), "1,234"))[0])

    def test_numeric_refuses_a_case_with_no_expect(self):
        with self.assertRaises(ValueError):
            numeric()(Case("c"), "3")

    def test_all_of_keeps_an_abstention(self):
        s = all_of(lambda c, o: True, lambda c, o: abstain("unread"))
        ok, _, extra = normalise(s(Case("c", expect="x"), "x"))
        self.assertFalse(ok)
        self.assertTrue(extra["abstained"])

    def test_all_of_with_no_scorers_is_refused(self):
        with self.assertRaises(ValueError):
            all_of()

    def test_a_small_score_is_not_a_pass(self):
        self.assertFalse(normalise(0.01)[0])
        self.assertTrue(normalise(1.0)[0])

    def test_nan_abstains(self):
        ok, _, extra = normalise(float("nan"))
        self.assertFalse(ok)
        self.assertTrue(extra["abstained"])

    def test_a_dict_without_ok_is_refused(self):
        with self.assertRaises(TypeError):
            normalise({"score": 1.0})


class TestRecords(unittest.TestCase):

    def test_adapt_reads_false_as_false(self):
        recs = list(adapt([{"arm": "a", "case": "c", "trial": "0", "ok": "false"}]))
        self.assertIs(recs[0].ok, False)
        self.assertEqual(recs[0].trial, 0)

    def test_read_survives_a_torn_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            p.write_text('{"arm": "a", "case": "c", "trial": 0, "ok": true}\n{"arm": "a", "ca')
            with self.assertWarns(RuntimeWarning):
                recs = read(p)
            self.assertEqual(len(recs), 1)


if __name__ == "__main__":
    unittest.main()
