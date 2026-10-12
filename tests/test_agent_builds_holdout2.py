"""
Held-out set 2 is an instrument too, and it is frozen.

The controls of the other two sets, plus the rule set 1 taught: every literal a
check reads appears in its task's text, so a solution that does the task
another way is not failed for the author's wording.
"""

import hashlib
import importlib.util
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "evals" / "agent_builds"
if not HERE.is_dir():
    raise unittest.SkipTest("the agent-build evals are not part of this checkout")
sys.path.insert(0, str(HERE))

_spec = importlib.util.spec_from_file_location("agent_builds_run_holdout2", HERE / "run.py")
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)
from holdout2 import HOLDOUT2  # noqa: E402

BY_ID = {t.id: t for t in HOLDOUT2}

MISTAKES = [
    ("holdout2-skill-system", [('{"role": "system", "parts": ["Reply only in French."]},', "")]),
    ("holdout2-skill-receipt-items", [('print(len(data["items"]))', "print(3)")]),
    ("holdout2-skill-base64-image", [('{"type": "image", "source": {"kind": "base64", "mime": "image/png", "data": PNG}},',
                                      "PNG,")]),
    ("holdout2-skill-default-variable", [('print(greet.run(variables={"tone": "casual"}))', "print(greet.run())")]),
    ("holdout2-tool-validation", [('print(value, "error:", exc)', 'print(value, "error")')]),
    ("holdout2-tool-counter", [("counter.run(input=2)\n", "")]),
    ("holdout2-chain-translate-headline", [('(translate, "german"), ', ""), ("{german}", "{summary}"),
                                           ('print(result["german"])\n', "")]),
    ("holdout2-chain-stop-on-error", [('on_step_error="stop"', 'on_step_error="skip"')]),
    ("holdout2-chain-urgency-ticket", [('return f"filed ticket urgent={urgent}"', 'return "filed ticket"')]),
    ("holdout2-chain-approve-publish", [('signal={"approved": True}', 'signal={"approved": False}')]),
    ("holdout2-chain-slogan-upper", [("return text.upper()", "return text")]),
    ("holdout2-pool-ordered-translations", [("max_flows=2", "max_flows=3")]),
    ("holdout2-pool-shared-budget", [(", max_cost=0.10", "")]),
    # The stub fills the amount with 3, not the task's 20, so the threshold
    # itself cannot be told apart offline; the probe removes the gate instead.
    ("holdout2-agent-approve-small", [('permissions=PermissionPolicy({"financial": "approve"}),\n', ""),
                                      ("approve=lambda request: request.arguments.get(\"amount\", 0) <= 30", "")]),
    ("holdout2-agent-human-results", [('messages.append(tool_result_turn(call.id, "A human says: a coast redwood."))',
                                       "messages.append(tool_result_turn(call.id, agent.execute(call, state)[0]))")]),
    ("holdout2-agent-refuse-email", [("approve=lambda request: False", "approve=lambda request: True")]),
    ("holdout2-agent-tool-log", [('print([(e.name, e.payload.get("arguments")) for e in tracer.events if e.type == "tool_call.started"])',
                                  "print(len(tracer.events))")]),
    ("holdout2-agent-two-workers", [("team=[historian, chemist]", "")]),
    ("holdout2-agent-plan-first", [(', mode="waterfall", planner_model=Model("claude-sonnet-4-6")', "")]),
    ("holdout2-agent-token-ceiling", [("token_budget(2000)]", "token_budget(3000)]")]),
]

#: Literals a check reads, by task, each of which must appear in the task text.
LITERALS = {
    "holdout2-skill-system": ["capital of Canada", "French"],
    "holdout2-skill-receipt-items": ["Coffee"],
    "holdout2-skill-default-variable": ["formal", "casual"],
    "holdout2-tool-validation": ["example.com", "invalid email"],
    "holdout2-tool-counter": ["7"],
    "holdout2-chain-stop-on-error": ["inventory unavailable"],
    "holdout2-chain-urgency-ticket": ["urgent"],
    "holdout2-chain-slogan-upper": ["solar lamp", "campers"],
    "holdout2-pool-ordered-translations": ["Good morning", "Thank you", "See you soon", "Where is the station", "I like tea"],
    "holdout2-agent-token-ceiling": ["2000"],
}


class TestHeldOutSetTwo(unittest.TestCase):

    def test_twenty_tasks_in_the_same_groups(self):
        self.assertEqual(len(HOLDOUT2), 20)
        groups = [t.group for t in HOLDOUT2]
        self.assertEqual({g: groups.count(g) for g in set(groups)},
                         {"skill": 4, "tool": 2, "chain": 5, "pool": 2, "agent": 7})

    def test_the_file_is_frozen(self):
        recorded = (HERE / "HOLDOUT2.sha256").read_text().split()[0]
        actual = hashlib.sha256((HERE / "holdout2.py").read_bytes()).hexdigest()
        self.assertEqual(actual, recorded,
                         "holdout2.py changed after it was frozen; write a new set instead of updating the hash.")

    def test_no_task_repeats_another_set(self):
        from holdout import HOLDOUT
        from tasks import TASKS
        others = {t.text for t in TASKS + HOLDOUT}
        for t in HOLDOUT2:
            self.assertNotIn(t.text, others)

    def test_every_literal_a_check_reads_is_in_the_task_text(self):
        for case_id, literals in LITERALS.items():
            for literal in literals:
                with self.subTest(task=case_id, literal=literal):
                    self.assertIn(literal, BY_ID[case_id].text)

    def test_the_documents_an_agent_reads_do_not_mention_it(self):
        for name in ("SKILL.md", "llms.txt"):
            text = (ROOT / name).read_text()
            with self.subTest(document=name):
                self.assertNotIn("holdout", text.lower())
                for t in HOLDOUT2:
                    self.assertNotIn(t.text, text)

    def test_every_reference_passes_its_own_check(self):
        for t in HOLDOUT2:
            with self.subTest(task=t.id):
                verdict = t.check(run.sandbox(t.reference, t.prefer))
                self.assertTrue(verdict["ok"], verdict["why"])

    def test_a_program_that_builds_nothing_fails_every_check(self):
        noise, empty = run.sandbox("print('hello')\n"), run.sandbox("")
        for t in HOLDOUT2:
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
