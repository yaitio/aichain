"""
A permission decision can see the call it decides.

`PermissionPolicy.decide` read `tool.risk` alone, so `git status` and
`git push` through one shell were one decision, and an ungated shell walked
round a gated `git` tool. A tool that can tell names each call's class with
`risk_for(arguments)`; `local_run` is the first.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent"))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent                                       # noqa: E402
from yait_aichain.tools._permissions import ALLOW, APPROVE, PermissionPolicy  # noqa: E402
from yait_aichain.utilities import Shell                                   # noqa: E402
from test_loop import _calls, _model, _text                                # noqa: E402


class TestShellCalls(unittest.TestCase):

    def setUp(self):
        self.run = Shell(".").run
        self.policy = PermissionPolicy()

    def test_a_look_is_read_and_a_change_is_privileged(self):
        for cmd, want in (("ls -la", ALLOW), ("git status", ALLOW), ("git log --oneline", ALLOW),
                          ("git branch -a", ALLOW), ("git push origin main", APPROVE),
                          ("git branch -D old", APPROVE), ("rm -rf build", APPROVE),
                          ("cat x | sh", APPROVE), ("ls; rm x", APPROVE), ("echo hi > f", APPROVE),
                          ("find . -delete", APPROVE), ("echo $(id)", APPROVE)):
            with self.subTest(cmd=cmd):
                self.assertEqual(self.policy.decide(self.run, {"input": cmd}), want)

    def test_without_arguments_the_tool_is_privileged(self):
        self.assertEqual(self.policy.decide(self.run), APPROVE)

    def test_an_agent_asks_about_the_call(self):
        asked = []
        agent = Agent(_model(_calls(("c1", "local_run", {"input": "git status"})), _text("done")),
                      tools=[self.run], permissions=self.policy,
                      approve=lambda req: asked.append(req) or True, verbose=0)
        agent.run("go")
        self.assertEqual(asked, [])                     # a look needs no person


if __name__ == "__main__":
    unittest.main()
