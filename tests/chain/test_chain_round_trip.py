"""
A saved chain comes back field for field.

Mutation testing, 2026-09-13: in `Chain.save` and `Chain.load` together, 225
one-token changes survived the whole suite — a Skill's name, description,
output spec, variables and options could each be written as `None`, a step's
`input_map` and options read from the wrong key, and the `api_key` passed to
`load` ignored, and nothing failed. The existing smoke test proved a chain
saves and runs; it never compared what came back with what went in.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from yait_aichain import Agent, Chain, Model, Skill                      # noqa: E402
from yait_aichain.agent import cost_budget, step_count, token_budget     # noqa: E402
from yait_aichain.tools._base import Tool                                # noqa: E402


class Configured(Tool):
    name = "configured"
    description = "A tool with constructor arguments."
    parameters = {"type": "object", "properties": {"input": {"type": "string"}}}

    def __init__(self, factor=1):
        self.factor = factor

    def _serialise_init_args(self):
        return {"factor": self.factor}

    def run(self, input=None, options=None):
        return input * self.factor


@unittest.skipUnless(HAS_YAML, "pyyaml not installed")
class TestRoundTrip(unittest.TestCase):

    def setUp(self):
        skill = Skill(Model("gpt-4o", api_key="k"),
                      prompt="Summarise {text}",
                      output={"format": {"type": "json"}},
                      variables={"text": "default"},
                      options={"temperature": 0.3},
                      name="summariser", description="makes it short")
        agent = Agent(Model("claude-sonnet-4-6", api_key="k"), tools=[],
                      mode="waterfall", instructions="be brief",
                      stop_when=[step_count(4), token_budget(900), cost_budget(0.25)],
                      name="researcher", description="finds things")
        self.original = Chain(
            steps=[(skill, "summary"),
                   (Configured(factor=3), "tripled", {"input": "summary"}, {"note": "x"}),
                   (agent, "answer", {}, {"task_key": "summary"})],
            variables={"lang": "pt"}, name="pipeline", description="three kinds",
            on_step_error="skip")
        self.path = os.path.join(tempfile.mkdtemp(), "chain.yaml")
        self.original.save(self.path)
        self.loaded = Chain.load(self.path, api_key="loaded-key", trusted=True)

    def test_chain_fields(self):
        c = self.loaded
        self.assertEqual((c.name, c.description, c.variables, c.on_step_error),
                         ("pipeline", "three kinds", {"lang": "pt"}, "skip"))

    def test_step_keys_maps_and_options(self):
        got = [(key, input_map, kind, options) for _, key, input_map, kind, options in self.loaded._steps]
        want = [(key, input_map, kind, options) for _, key, input_map, kind, options in self.original._steps]
        self.assertEqual(got, want)

    def test_the_skill(self):
        s = self.loaded._steps[0][0]
        o = self.original._steps[0][0]
        self.assertEqual((s.name, s.description), ("summariser", "makes it short"))
        self.assertEqual(s._input, o._input)
        self.assertEqual(s._output, o._output)
        self.assertEqual(s.variables, {"text": "default"})
        self.assertEqual(s.options, {"temperature": 0.3})
        self.assertEqual(s.model.name, "gpt-4o")
        self.assertEqual(s.model.client.api_key, "loaded-key")

    def test_the_tool_keeps_its_constructor_arguments(self):
        t = self.loaded._steps[1][0]
        self.assertIsInstance(t, Configured)
        self.assertEqual(t.factor, 3)

    def test_the_agent(self):
        a = self.loaded._steps[2][0]
        self.assertEqual((a.model.name, a.mode, a.instructions, a.name, a.description),
                         ("claude-sonnet-4-6", "waterfall", "be brief", "researcher", "finds things"))
        self.assertEqual(a.model.client.api_key, "loaded-key")
        self.assertEqual([c.spec for c in a.stop_when],
                         [{"kind": "step_count", "value": 4}, {"kind": "token_budget", "value": 900},
                          {"kind": "cost_budget", "value": 0.25}])

    def test_the_loaded_chain_saves_to_the_same_file(self):
        again = os.path.join(tempfile.mkdtemp(), "again.yaml")
        self.loaded.save(again)
        with open(self.path) as a, open(again) as b:
            self.assertEqual(a.read(), b.read())


if __name__ == "__main__":
    unittest.main()
