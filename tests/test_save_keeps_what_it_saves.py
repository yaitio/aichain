"""
A saved Skill, agent or tool comes back as it was — or saving says why not.

A Skill came back at temperature 1.0 from 0, with no fallback, no retries and
no budget; an agent came back without its permission policy — a gate quietly
lost — and its Shell tools moved to the loader's working directory; a Gate
or an MCP tool was saved and then failed to load; and `yaml.dump` wrote
`!!python/tuple`, which the loader refused.
"""

import os
import tempfile
import unittest
import warnings

os.environ.setdefault("OPENAI_API_KEY", "k")

from yait_aichain import Agent, Chain, Model, PermissionPolicy, Skill


def _path():
    return os.path.join(tempfile.mkdtemp(), "saved.yaml")


class TestSkill(unittest.TestCase):

    def test_options_fallback_retries_and_budget_come_back(self):
        skill = Skill([Model("gpt-4o", api_key="k", options={"temperature": 0.0}),
                       Model("gpt-4o-mini", api_key="k")],
                      prompt="hi", max_retries=2, retry_delay=0.5, max_cost=0.25)
        path = _path()
        skill.save(path)
        back = Skill.load(path, api_key="k")
        self.assertEqual(back.model._asked, {"temperature": 0.0})
        self.assertEqual([m.name for m in back.models], ["gpt-4o", "gpt-4o-mini"])
        self.assertEqual((back.max_retries, back.retry_delay, back.max_cost.limit), (2, 0.5, 0.25))

    def test_a_tuple_is_written_as_plain_yaml(self):
        skill = Skill(Model("gpt-4o", api_key="k"), prompt="hi", variables={"pair": (1, 2)})
        path = _path()
        skill.save(path)
        self.assertNotIn("!!python", open(path).read())
        self.assertEqual(Skill.load(path, api_key="k").variables, {"pair": [1, 2]})

    def test_an_empty_file_is_said_to_be_one(self):
        path = _path()
        open(path, "w").close()
        with self.assertRaisesRegex(ValueError, "not a saved"):
            Skill.load(path)


class TestAgentInAChain(unittest.TestCase):

    def test_the_policy_and_the_tools_folder_come_back(self):
        from yait_aichain.utilities.shell._read import LocalReadTool
        folder = tempfile.mkdtemp()
        agent = Agent(Model("gpt-4o", api_key="k"), tools=[LocalReadTool(folder)],
                      permissions=PermissionPolicy({"external": "deny"}))
        path = _path()
        Chain(steps=[(agent, "answer")]).save(path)
        back = Chain.load(path, api_key="k", trusted=True)._steps[0][0]
        self.assertEqual(back.permissions.rules["external"], "deny")
        self.assertEqual(back.tools[0].root, os.path.realpath(folder))

    def test_an_approver_is_said_to_be_left_behind(self):
        agent = Agent(Model("gpt-4o", api_key="k"), permissions=PermissionPolicy(),
                      approve=lambda r: True)
        with self.assertWarnsRegex(RuntimeWarning, "approver"):
            Chain(steps=[(agent, "answer")]).save(_path())


class TestWhatCannotComeBackIsRefused(unittest.TestCase):

    def test_a_gate(self):
        from yait_aichain.tools import Gate, tool

        @tool(risk="read")
        def look(x: str) -> str:
            """Look."""
            return x
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(TypeError, "Gate"):
                Chain(steps=[(Gate(look), "x")]).save(_path())


if __name__ == "__main__":
    unittest.main()
