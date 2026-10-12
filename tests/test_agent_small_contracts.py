"""
Small contracts in the agent and the run document, each a defect once.

A check by class name passed a user's own class called `Utility` or `Skill`
for the library's; a plan given as one string was recorded one character per
item; a run whose steps were all done or skipped reported itself running; and
a dict that was not a run document raised a bare KeyError.
"""

import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from yait_aichain import Agent, Model, Skill
from yait_aichain.state import RunDocument, StepStatus


class TestTypeChecksAreTypes(unittest.TestCase):

    def test_a_class_merely_named_like_ours_is_not_mistaken_for_it(self):
        class Utility:
            name = "u"

        class Skill_:
            pass
        Skill_.__name__ = "Skill"
        with self.assertRaises(TypeError) as caught:
            Agent(Model("gpt-4o", api_key="k"), tools=[Skill_()])
        self.assertNotIn("A Skill is one call", str(caught.exception))
        with self.assertRaises(TypeError) as caught:
            Agent(Model("gpt-4o", api_key="k"), tools=[Utility()])
        self.assertNotIn("is a utility", str(caught.exception))

    def test_a_real_skill_is_named_as_one(self):
        skill = Skill(Model("gpt-4o", api_key="k"), prompt="hi")
        with self.assertRaisesRegex(TypeError, "A Skill is one call"):
            Agent(Model("gpt-4o", api_key="k"), tools=[skill])


class TestAPlanOfOneItem(unittest.TestCase):

    def test_a_string_is_one_item(self):
        agent = Agent(Model("gpt-4o", api_key="k"))
        state = agent.new_state()
        agent._do_plan({"items": "check the invoice"}, state)
        self.assertEqual(state["plan"], ["check the invoice"])


class TestRunDocument(unittest.TestCase):

    def test_done_and_skipped_is_done(self):
        doc = RunDocument.new("chain", ["a", "b"])
        doc.steps[0]["status"], doc.steps[1]["status"] = StepStatus.DONE, StepStatus.SKIPPED
        self.assertEqual(doc.status, StepStatus.DONE)

    def test_something_else_is_said_to_be_something_else(self):
        with self.assertRaisesRegex(ValueError, "not a run document"):
            RunDocument.from_dict({"kind": "chain"})


if __name__ == "__main__":
    unittest.main()
