"""
The held-out set is an instrument too, and it is frozen.

`evals/agent_builds/holdout.py` is the exam for the skill: twenty tasks that
nobody tunes `SKILL.md`, `llms.txt` or the recipes against. The same controls
as the tuning set run here — every reference passes, a program that builds
nothing fails, a plausible mistake in each reference is caught — plus two that
only a held-out set needs: its bytes match the hash recorded when it was
frozen, and the documents an agent is given do not mention it.
"""

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "evals" / "agent_builds"
sys.path.insert(0, str(HERE))

_spec = importlib.util.spec_from_file_location("agent_builds_run_holdout", HERE / "run.py")
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)
from holdout import HOLDOUT  # noqa: E402

BY_ID = {t.id: t for t in HOLDOUT}

MISTAKES = [
    ("holdout-skill-category", [('"enum": ["billing", "shipping", "technical", "account", "other"]',
                                 '"description": "billing, shipping, technical, account or other"')]),
    ("holdout-skill-image", [('{"type": "image", "source": {"kind": "url", "url": "https://example.com/cat.jpg"}},',
                              '"https://example.com/cat.jpg",')]),
    ("holdout-skill-cost-cap", [(", max_cost=0.01", "")]),
    ("holdout-skill-stream", [("for piece in s.stream():", "for piece in [s.run()]:")]),
    ("holdout-tool-structured", [('["status"])', ")")]),
    ("holdout-tool-failed-call", [("Divide()(input=0)", "Divide().run(input=0)")]),
    ("holdout-chain-invoice", [('(AddVat(), "total"), ', ""), ("{total}", "{amount}"),
                               ('result["total"], ', "")]),
    ("holdout-chain-rename", [(', {"text": "draft"}', "")]),
    ("holdout-chain-gate-reject", [('signal={"approved": False}', 'signal={"approved": True}')]),
    ("holdout-chain-fresh-resume", [('build().resume(paused.run_id, signal={"title": "Edited title"})',
                                     'build().run(variables={"topic": "city gardens"})')]),
    ("holdout-chain-merge", [("Write a greeting to {name} from {city}.", "Write a greeting to {name}.")]),
    ("holdout-pool-partial-failure", [("print(result.error)", "")]),
    ("holdout-pool-agents", [("stop_when=[step_count(5)]", "stop_when=[step_count(6)]")]),
    ("holdout-agent-two-tools", [("tools=[Weather(), LocalTime()]", "tools=[LocalTime()]")]),
    ("holdout-agent-deny", [('{"destructive": "deny"}', '{"destructive": "allow"}')]),
    ("holdout-agent-cost-cap", [(", max_cost=0.05", "")]),
    ("holdout-agent-delegate", [("team=[worker], ", "")]),
    ("holdout-agent-follow-up", [('f"You said: {first.output}. What is its population?"',
                                  '"What is the population of the capital of Chile?"')]),
    ("holdout-agent-stream", [("print(event.type)", "pass")]),
    ("holdout-agent-private-model", [('Model("private/Qwen/Qwen3-8B", client_options={"url": "http://gpu.internal:8000"})',
                                      'Model("gpt-5.4-mini")')]),
]


class TestTheHeldOutSet(unittest.TestCase):

    def test_twenty_tasks_in_the_same_groups_as_the_tuning_set(self):
        self.assertEqual(len(HOLDOUT), 20)
        groups = [t.group for t in HOLDOUT]
        self.assertEqual({g: groups.count(g) for g in set(groups)},
                         {"skill": 4, "tool": 2, "chain": 5, "pool": 2, "agent": 7})

    def test_the_file_is_frozen(self):
        recorded = (HERE / "HOLDOUT.sha256").read_text().split()[0]
        actual = hashlib.sha256((HERE / "holdout.py").read_bytes()).hexdigest()
        self.assertEqual(actual, recorded,
                         "holdout.py changed after it was frozen. A held-out set edited after a "
                         "measurement is a rehearsal; write a new set instead of updating the hash.")

    def test_the_documents_an_agent_reads_do_not_mention_it(self):
        for name in ("SKILL.md", "llms.txt"):
            with self.subTest(document=name):
                self.assertNotIn("holdout", (ROOT / name).read_text().lower())

    def test_every_reference_passes_its_own_check(self):
        for t in HOLDOUT:
            with self.subTest(task=t.id):
                verdict = t.check(run.sandbox(t.reference, t.prefer))
                self.assertTrue(verdict["ok"], verdict["why"])

    def test_a_program_that_builds_nothing_fails_every_check(self):
        noise, empty = run.sandbox("print('hello')\n"), run.sandbox("")
        for t in HOLDOUT:
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
