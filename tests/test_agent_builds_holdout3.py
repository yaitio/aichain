"""
Held-out set 3 — the 3.x conventions — is an instrument too, and it is frozen.

The controls of the earlier sets: every reference passes its own check, a
program that builds nothing fails every one, each check bites on a plausible
mistake, and every literal a check reads is in its task's text.
"""

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "evals" / "agent_builds"
sys.path.insert(0, str(HERE))

_spec = importlib.util.spec_from_file_location("agent_builds_run_holdout3", HERE / "run.py")
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)
from holdout3 import HOLDOUT3  # noqa: E402

BY_ID = {t.id: t for t in HOLDOUT3}

MISTAKES = [
    ("h3-tool-function", [('print(count_vowels.run(word="education"))', "print(5)")]),
    ("h3-tool-destructive-approval", [('approve=lambda request: ApprovalDecision(False, "never delete")',
                                       "approve=lambda request: True")]),
    ("h3-tool-chunk", [("print(len(pieces))", "print(1)")]),
    ("h3-tool-image", [("tools=[painter]", "tools=[]")]),
    ("h3-shell-read-only", [("tools=[sh.browse, sh.read]", "tools=sh.tools")]),
    ("h3-rest-endpoint", [("tools=[get_order]", "tools=[]")]),
    ("h3-search-and-fetch", [('fetch("local")', 'fetch("tavily")')]),
    ("h3-chain-function-step", [("return slogan.upper()", "return slogan")]),
    ("h3-chain-template", [('{"text": "Title: {title}"}', '{"text": "title"}')]),
    ("h3-chain-dict-fields", [("{temp} degrees", "warm")]),
    ("h3-budget-chain", [(", max_cost=0.05", "")]),
]

#: Literals a check reads, by task (case aside), each of which must appear in the task text.
LITERALS = {
    "h3-tool-destructive-approval": ["destructive", "delete"],
    "h3-tool-chunk": ["chunking"],
    "h3-tool-image": ["gpt-image-1.5"],
    "h3-search-and-fetch": ["exa", "on this machine"],
    "h3-shell-read-only": ["read", "write", "run commands"],
    "h3-chain-function-step": ["lemonade", "upper case"],
    "h3-chain-template": ["Title: ", "the quick brown fox"],
    "h3-chain-dict-fields": ["Paris", "21"],
}


class TestHeldOutSetThree(unittest.TestCase):

    def test_eleven_tasks(self):
        self.assertEqual(len(HOLDOUT3), 11)

    def test_the_file_is_frozen(self):
        recorded = (HERE / "HOLDOUT3.sha256").read_text().split()[0]
        actual = hashlib.sha256((HERE / "holdout3.py").read_bytes()).hexdigest()
        self.assertEqual(actual, recorded,
                         "holdout3.py changed after it was frozen; write a new set instead of updating the hash.")

    def test_no_task_repeats_another_set(self):
        from holdout import HOLDOUT
        from holdout2 import HOLDOUT2
        from tasks import TASKS
        others = {t.text for t in TASKS + HOLDOUT + HOLDOUT2}
        for t in HOLDOUT3:
            self.assertNotIn(t.text, others)

    def test_every_literal_a_check_reads_is_in_the_task_text(self):
        for case_id, literals in LITERALS.items():
            for literal in literals:
                with self.subTest(task=case_id, literal=literal):
                    self.assertIn(literal.lower(), BY_ID[case_id].text.lower())

    def test_the_documents_an_agent_reads_do_not_mention_it(self):
        for name in ("SKILL.md", "llms.txt"):
            text = (ROOT / name).read_text()
            with self.subTest(document=name):
                self.assertNotIn("holdout", text.lower())
                for t in HOLDOUT3:
                    self.assertNotIn(t.text, text)
                    self.assertNotIn(t.reference.strip(), text)

    def test_every_reference_passes_its_own_check(self):
        for t in HOLDOUT3:
            with self.subTest(task=t.id):
                verdict = t.check(run.sandbox(t.reference, t.prefer))
                self.assertTrue(verdict["ok"], verdict["why"])

    def test_a_program_that_builds_nothing_fails_every_check(self):
        noise, empty = run.sandbox("print('hello')\n"), run.sandbox("")
        for t in HOLDOUT3:
            with self.subTest(task=t.id):
                self.assertFalse(t.check(noise)["ok"])
                self.assertFalse(t.check(empty)["ok"])

    def test_every_task_has_a_mistake_probe(self):
        self.assertEqual({case for case, _ in MISTAKES}, set(BY_ID))

    def test_a_plausible_mistake_is_caught(self):
        for case_id, changes in MISTAKES:
            t = BY_ID[case_id]
            code = t.reference
            for old, new in changes:
                self.assertIn(old, code, f"{case_id}: the reference changed; update the probe")
                code = code.replace(old, new, 1)
            with self.subTest(task=case_id):
                self.assertFalse(t.check(run.sandbox(code, t.prefer))["ok"])


if __name__ == "__main__":
    unittest.main()
