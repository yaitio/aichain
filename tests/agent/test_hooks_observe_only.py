"""
A hook observes; it does not steer, and it does not fail unheard.

A hook that masked a token in `payload["arguments"]` changed the argument the
tool ran with; a broken hook logged one DEBUG line; one `Agent(verbose=1)`
lowered the package logger for the whole process, so every other agent
printed its tool arguments too.
"""

import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import _events                                        # noqa: E402
from yait_aichain.agent import Agent                                    # noqa: E402
from test_loop import Echo, _calls, _model, _text                       # noqa: E402

SCRIPT = (_calls(("c1", "echo", {"value": "secret"})), _text("done"))


class TestHooks(unittest.TestCase):

    def test_a_hook_that_edits_the_payload_does_not_edit_the_call(self):
        sink = []

        def mask(ev):
            if ev.type == "tool_call.started":
                ev.payload["arguments"]["value"] = "***"
        Agent(_model(*SCRIPT), tools=[Echo(sink)], hooks=[mask], verbose=0).run("go")
        self.assertEqual(sink, ["secret"])

    def test_a_broken_hook_is_reported_once_with_its_traceback(self):
        def broken(ev):
            raise RuntimeError("tracer down")
        _events._FAILED.discard(id(broken))
        with self.assertLogs("yait_aichain.events", "WARNING") as logged:
            Agent(_model(*SCRIPT), tools=[Echo()], hooks=[broken], verbose=0).run("go")
        self.assertEqual(len(logged.records), 1)
        self.assertIsNotNone(logged.records[0].exc_info)

    def test_one_verbose_agent_leaves_the_package_logger_alone(self):
        pkg = logging.getLogger("yait_aichain")
        level, handlers = pkg.level, list(pkg.handlers)
        Agent(_model(*SCRIPT), tools=[Echo()], verbose=1)
        self.assertEqual((pkg.level, pkg.handlers), (level, handlers))


if __name__ == "__main__":
    unittest.main()
