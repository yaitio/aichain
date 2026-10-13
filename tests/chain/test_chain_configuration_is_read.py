"""
Everything a chain step is configured with is read, or refused.

A Skill step dropped input_map names the chain lacked and never filled a
template; a tool with no properties took any mapping and dropped it; an Agent
ignored input_map; an output_field typo gave None and success; options on a
non-agent step were read by nothing; an empty custom store with ``__len__``
was replaced by an in-memory one; ``[runner, "key"]`` from YAML failed only
in ``run()``; and ``len`` was called by keyword and failed.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain.agent import Agent                                  # noqa: E402
from yait_aichain.chain import Chain                                  # noqa: E402
from yait_aichain.models import Model                                 # noqa: E402
from yait_aichain.state import InMemoryStore                          # noqa: E402
from yait_aichain.tools._base import Tool                             # noqa: E402


class Seen(Tool):
    name = "seen"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    def run(self, text=""):
        return text


class Recorder:
    """A Skill-shaped step: records the variables it was run with."""
    name = "rec"

    def __init__(self):
        self.got = None

    def run(self, variables=None):
        self.got = dict(variables or {})
        return "ok"


class TestReadOrRefused(unittest.TestCase):

    def test_a_skill_steps_input_map_is_resolved(self):
        rec = Recorder()
        Chain(steps=[(rec, "out", {"topic": "Draw {subject} please"})]).run(variables={"subject": "cats"})
        self.assertEqual(rec.got["topic"], "Draw cats please")
        with self.assertRaisesRegex(ValueError, "missing"):
            Chain(steps=[(Recorder(), "out", {"topic": "missing"})]).run(variables={})

    def test_an_agent_step_refuses_input_map_and_unknown_options(self):
        agent = Agent(Model("gpt-4o", api_key="k"))
        with self.assertRaisesRegex(ValueError, "input_map"):
            Chain(steps=[(agent, "out", {"task": "q"})])
        with self.assertRaisesRegex(ValueError, "output_field"):
            Chain(steps=[(agent, "out", {}, {"output_field": "outptu"})])
        with self.assertRaisesRegex(ValueError, "options are"):
            Chain(steps=[(agent, "out", {}, {"taskkey": "q"})])

    def test_options_on_a_tool_step_are_refused(self):
        with self.assertRaisesRegex(ValueError, "nothing would read"):
            Chain(steps=[(Seen(), "out", {}, {"note": "x"})])

    def test_an_empty_store_of_ones_own_is_kept(self):
        class Mine(InMemoryStore):
            def __len__(self):
                return 0
        store = Mine()
        self.assertIs(Chain(steps=[(Seen(), "out")], store=store)._store, store)

    def test_a_list_step_is_a_step_and_a_non_runner_is_refused_now(self):
        out = Chain(steps=[[Seen(), "out"]]).run(variables={"text": "hi"})
        self.assertEqual(out["out"], "hi")
        with self.assertRaisesRegex(TypeError, "chain step"):
            Chain(steps=[(42, "out")])

    def test_a_positional_only_function_is_called_positionally(self):
        out = Chain(steps=[(len, "n", {"obj": "items"})]).run(variables={"items": [1, 2, 3]})
        self.assertEqual(out["n"], 3)


if __name__ == "__main__":
    unittest.main()
