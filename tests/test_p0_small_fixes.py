"""
Three one-line defects from the 4.0 audit's first tier, each pinned here.

* `Model(options=OPTS)` popped `extra` out of the caller's dict, so a second
  model built from the same options lost its provider fields;
* a REST path parameter of `..` walked the request out of its endpoint —
  `DELETE /projects/1/members/..` is `DELETE /projects/1`;
* `local_run` put the interpreter in front of any command ending in `.py`,
  so `python x.py` became `<python> python x.py`;
* a `Gate` approved on any truthy value, so the string "false" from a form
  ran the action, and it passed a policy as `write` whatever it wrapped.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

from yait_aichain import Model


class TestOptionsAreTheCallersDict(unittest.TestCase):

    def test_two_models_from_one_dict_both_get_extra(self):
        opts = {"temperature": 0.3, "extra": {"service_tier": "flex"}}
        a, b = Model("gpt-4o", api_key="k", options=opts), Model("gpt-4o", api_key="k", options=opts)
        self.assertEqual(opts, {"temperature": 0.3, "extra": {"service_tier": "flex"}})
        self.assertEqual(a.extra, b.extra)
        self.assertEqual(b.extra, {"service_tier": "flex"})


class TestARestPathStaysInItsEndpoint(unittest.TestCase):

    def test_dot_segments_and_empty_values_are_refused(self):
        from yait_aichain.utilities.rest import RestApiTool
        t = RestApiTool("drop_member", "Remove a member.", "DELETE",
                        "https://api.example.com/projects/{pid}/members/{mid}",
                        path_params=["pid", "mid"])
        t._http = MagicMock()
        for bad in ("..", ".", ""):
            with self.subTest(mid=bad):
                with self.assertRaisesRegex(ValueError, "cannot be"):
                    t.run({"pid": "1", "mid": bad})
        t._http.request.assert_not_called()


class TestLocalRunRewritesOnlyAScript(unittest.TestCase):

    def setUp(self):
        from yait_aichain.utilities.shell._run import LocalRunTool
        self.root = tempfile.mkdtemp()
        with open(os.path.join(self.root, "x.py"), "w") as fh:
            fh.write("print('ran')\n")
        self.tool = LocalRunTool(self.root)

    def _out(self, command, **options):
        result = self.tool.run(command, options or None)
        return result["stdout"].strip(), result["returncode"]

    def test_a_command_that_names_python_is_run_as_written(self):
        self.assertEqual(self._out(f"{sys.executable} x.py"), ("ran", 0))

    def test_a_command_that_reads_a_py_file_is_not_given_to_python(self):
        self.assertEqual(self._out("cat x.py"), ("print('ran')", 0))

    def test_a_bare_script_runs_under_the_interpreter(self):
        self.assertEqual(self._out("x.py"), ("ran", 0))

    def test_python_true_keeps_a_pipe(self):
        out, code = self._out("x.py | tr a-z A-Z", python=True)
        self.assertEqual((out, code), ("RAN", 0))


class TestAGateOpensOnTrueAlone(unittest.TestCase):

    def _gate(self):
        from yait_aichain.tools import Gate, tool

        @tool(risk="destructive")
        def wipe(table: str) -> str:
            """Drop a table."""
            ran.append(table)
            return "wiped"
        ran: list = []
        return Gate(wipe), ran

    def test_only_the_boolean_true_runs_it(self):
        gate, ran = self._gate()
        self.assertEqual(gate.run({"approved": True}, table="t"), "wiped")
        self.assertEqual(gate.run({"approved": False}, table="t")["skipped"], True)
        for signal in ({"approved": "false"}, {"approved": "yes"}, {"approved": 1}, True, {}):
            with self.subTest(signal=signal):
                with self.assertRaisesRegex(TypeError, "resumes with"):
                    gate.run(signal, table="t")
        self.assertEqual(ran, ["t"])

    def test_the_gate_carries_the_risk_of_what_it_guards(self):
        gate, _ = self._gate()
        self.assertEqual(gate.risk, "destructive")

    def test_a_plain_function_is_made_a_tool_at_construction(self):
        from yait_aichain.tools import Gate

        def send(to: str) -> str:
            """Send it."""
            return to
        self.assertEqual(Gate(send).run({"approved": True}, to="x"), "x")
        with self.assertRaises(TypeError):
            Gate(42)


if __name__ == "__main__":
    unittest.main()
