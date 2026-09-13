"""
The instrument that measures whether an agent can build on this library is itself held.

`evals/agent_builds/` runs model-written programs in a sandbox and checks what
they did. An instrument that passes everything reports a library anyone can
build on, and nothing about the number would reveal it. So three controls run
in the suite, offline, on every change:

* every reference solution passes its own check (the oracle);
* a program that builds nothing fails every check (the noise);
* a reference with one plausible mistake — the wrong ceiling, a silent error
  policy, a missing approver, `run()` where the loop was to be driven by hand —
  fails the check it belongs to.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "evals" / "agent_builds"
sys.path.insert(0, str(HERE))

_spec = importlib.util.spec_from_file_location("agent_builds_run", HERE / "run.py")
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)
from tasks import TASKS  # noqa: E402

BY_ID = {t.id: t for t in TASKS}

MISTAKES = [
    ("agent-tool-ceiling", [("step_count(6)", "step_count(7)")]),
    ("chain-skip-failure", [('on_step_error="skip"', 'on_step_error="collect"')]),
    ("agent-approval", [('permissions=PermissionPolicy({"financial": "approve"}), ', ""),
                        ("approve=approve)", ")")]),
    ("skill-two-providers", [('("claude-sonnet-4-6", "gpt-5.4-mini")', '("gpt-5.4-mini", "gpt-4o")')]),
    ("chain-pause-resume", [(', store=FileStore("runs"))', ")")]),
    ("pool-fanout", [("max_flows=3", "max_flows=6")]),
    ("agent-stall-nudge", [(", stalled(3)]", "]")]),
    ("chain-two-skills", [('prompt="Turn this into a tweet: {description}"',
                           'prompt="Write a tweet about {product}."')]),
    ("agent-verified-stop", [('check(lambda state: "answer" in store, name="stored"), ', "")]),
]


class TestTheInstrument(unittest.TestCase):

    def test_there_are_twenty_tasks(self):
        self.assertEqual(len(TASKS), 20)

    def test_every_reference_passes_its_own_check(self):
        for t in TASKS:
            with self.subTest(task=t.id):
                verdict = t.check(run.sandbox(t.reference, t.prefer))
                self.assertTrue(verdict["ok"], verdict["why"])

    def test_a_program_that_builds_nothing_fails_every_check(self):
        noise = run.sandbox("print('hello')\n")
        for t in TASKS:
            with self.subTest(task=t.id):
                self.assertFalse(t.check(noise)["ok"])

    def test_no_code_is_a_failure_not_a_crash(self):
        empty = run.sandbox("")
        for t in TASKS:
            self.assertFalse(t.check(empty)["ok"])

    def test_a_plausible_mistake_is_caught(self):
        for case_id, changes in MISTAKES:
            t = BY_ID[case_id]
            code = t.reference
            for old, new in changes:
                self.assertIn(old, code, f"{case_id}: the reference changed; update the probe")
                code = code.replace(old, new, 1)
            with self.subTest(task=case_id):
                self.assertFalse(t.check(run.sandbox(code, t.prefer))["ok"])

    def test_the_sandbox_never_reaches_a_provider(self):
        out = run.sandbox(BY_ID["skill-summarise"].reference)
        self.assertEqual(out["exit"], 0, out["stderr"])
        self.assertTrue(out["trace"]["calls"])
        self.assertTrue(all(c.get("reply_kind") for c in out["trace"]["calls"]),
                        "a call was answered by something other than the stub")

    def test_code_is_extracted_from_a_reply(self):
        reply = "Here you go:\n```python\nprint('a')\n```\nand a note."
        self.assertEqual(run.extract(reply), "print('a')\n")
        self.assertEqual(run.extract("no code here"), "")


if __name__ == "__main__":
    unittest.main()
