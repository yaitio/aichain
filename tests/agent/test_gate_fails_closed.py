"""
The approval gate opens on a yes and nothing else.

``granted = bool(answer)`` approved a call on ``"denied"`` and on
``{"approved": False}``; a policy answering an unknown word meant "allowed";
``policy.rules`` could be changed after it was checked; and a callback that
raised was handled three ways — a check swallowed it, a ceiling, a nudge and
an approver raised out of ``run()`` with no result.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent, check, nudge, step_count                 # noqa: E402
from yait_aichain.tools._base import Tool                                      # noqa: E402
from yait_aichain.tools._permissions import PermissionPolicy                   # noqa: E402
from test_loop import _calls, _model, _text                                    # noqa: E402


class Risky(Tool):
    name = "risky"
    risk = "destructive"
    parameters = {"type": "object", "properties": {"x": {"type": "string"}}}

    def __init__(self, ran):
        self.ran = ran

    def run(self, x=""):
        self.ran.append(x)
        return "done"


SCRIPT = (_calls(("c1", "risky", {"x": "a"})), _text("ok"))


def _agent(approve, ran, **kw):
    return Agent(_model(*SCRIPT), tools=[Risky(ran)], verbose=0,
                 permissions=PermissionPolicy({"destructive": "approve"}), approve=approve, **kw)


class TestOnlyYesOpens(unittest.TestCase):

    def test_a_string_answer_is_a_failed_run_not_an_approval(self):
        ran = []
        result = _agent(lambda req: "denied", ran).run("go")
        self.assertEqual(ran, [])
        self.assertEqual(result.stopped_by, "condition_error")
        self.assertIn("approver", result.error)

    def test_a_dict_answer_does_not_approve(self):
        ran = []
        _agent(lambda req: {"approved": False}, ran).run("go")
        self.assertEqual(ran, [])

    def test_true_approves_and_false_refuses(self):
        ran = []
        _agent(lambda req: True, ran).run("go")
        self.assertEqual(ran, ["a"])
        ran = []
        _agent(lambda req: False, ran).run("go")
        self.assertEqual(ran, [])

    def test_an_unknown_decision_is_a_refusal(self):
        class Odd:
            def decide(self, tool):
                return "maybe"
        ran = []
        Agent(_model(*SCRIPT), tools=[Risky(ran)], verbose=0, permissions=Odd()).run("go")
        self.assertEqual(ran, [])

    def test_the_rules_cannot_change_after_the_check(self):
        policy = PermissionPolicy()
        with self.assertRaises(TypeError):
            policy.rules["destructive"] = "allow"

    def test_the_approver_sees_a_copy(self):
        ran = []

        def edit(req):
            req.arguments["x"] = "changed"
            return True
        _agent(edit, ran).run("go")
        self.assertEqual(ran, ["a"])


class TestCallbacksThatRaise(unittest.TestCase):

    def _run(self, cond):
        return Agent(_model(*SCRIPT), tools=[Risky([])], verbose=0,
                     stop_when=[step_count(5), cond]).run("go")

    def test_a_check_that_raises_ends_the_run_as_condition_error(self):
        def boom(state):
            raise RuntimeError("bad check")
        result = self._run(check(boom))
        self.assertEqual(result.stopped_by, "condition_error")
        self.assertIn("bad check", result.error)

    def test_a_nudge_that_raises_ends_the_run_as_condition_error(self):
        def boom(state):
            raise RuntimeError("bad nudge")
        self.assertEqual(self._run(nudge(boom, "x")).stopped_by, "condition_error")


class TestToolFailureCarriesItsTraceback(unittest.TestCase):

    def test_tool_call_ended_has_the_exception(self):
        class Broken(Tool):
            name = "risky"
            parameters = {"type": "object", "properties": {"x": {"type": "string"}}}

            def run(self, x=""):
                raise KeyError("missing")
        events = []
        Agent(_model(*SCRIPT), tools=[Broken()], verbose=0, hooks=[events.append]).run("go")
        ended = [e for e in events if e.type == "tool_call.ended"][0]
        self.assertEqual(ended.payload["exception"]["type"], "KeyError")
        self.assertIn("Traceback", ended.payload["exception"]["traceback"])


if __name__ == "__main__":
    unittest.main()
