"""
The last of the Consistency list, and two entries that measurement rewrote.

`PLAN.md` carried these since the June audit. Working through them, two turned
out to be wrong about the code rather than the code being wrong about itself —
which is worth a test each, because a stale plan item is a defect that costs
somebody an afternoon rediscovering that there is nothing to fix.
"""

import inspect
import json
import os
import pkgutil
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yait_aichain.tools as tools_pkg                              # noqa: E402
from yait_aichain import Model, Skill                               # noqa: E402
from yait_aichain.tools._base import Tool                           # noqa: E402


def _answering():
    m = Model("gpt-4o", api_key="k")
    m.client._post = MagicMock(return_value=json.dumps({
        "choices": [{"message": {"content": "hi there"}}],
        "usage": {"total_tokens": 3}}).encode())
    m.client._auth_headers = MagicMock(return_value={})
    return m


class TestThePromptShortcut(unittest.TestCase):
    """`Skill(model, prompt="...")` for the commonest shape there is: one
    user message, one text part. The long form stays the only way to say
    anything else, because a shortcut that grows options becomes a second
    input format, and two ways to say one thing is how docs and code drift."""

    def test_it_runs(self):
        skill = Skill(_answering(), prompt="Say hi to {name}")
        self.assertEqual(skill.run(variables={"name": "Ada"}), "hi there")

    def test_it_builds_exactly_one_user_message(self):
        skill = Skill(_answering(), prompt="Say hi")
        messages = skill._input["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["parts"][0]["text"], "Say hi")

    def test_both_at_once_is_refused(self):
        """Ambiguous rather than mergeable: which one should the model see?"""
        with self.assertRaises(ValueError):
            Skill(_answering(), input={"messages": []}, prompt="x")

    def test_neither_is_refused(self):
        with self.assertRaises(ValueError):
            Skill(_answering())

    def test_the_long_form_is_untouched(self):
        skill = Skill(_answering(),
                      input={"messages": [
                          {"role": "system", "parts": ["Be terse."]},
                          {"role": "user", "parts": ["Hi"]}]})
        self.assertEqual(len(skill._input["messages"]), 2)


class TestNamingIsNotTheInconsistencyItLookedLike(unittest.TestCase):
    """The plan said "bring camelCase classes to one style". Measured, the
    two styles encode two different things:

    * a **camelCase** class name is identical to the tool's `name` — the
      string the model sees;
    * a **PascalCase** one differs, and its wire name is snake_case.

    Renaming the camelCase ones would either change the wire name — breaking
    every prompt, eval and saved chain that references it — or open exactly
    the gap the PascalCase ones have. The real inconsistency is one layer
    down and is a decision, not a sweep: the *wire* names mix `convertToMD`
    with `vector_query`, so a model sees both styles in one tool list.
    """

    @staticmethod
    def _tools():
        found = {}
        for info in pkgutil.walk_packages(tools_pkg.__path__,
                                          prefix="yait_aichain.tools."):
            try:
                module = __import__(info.name, fromlist=["_"])
            except Exception:
                continue
            for obj in vars(module).values():
                if (inspect.isclass(obj) and issubclass(obj, Tool)
                        and obj is not Tool
                        and not obj.__name__.startswith("_")):
                    found[obj.__name__] = obj
        return found

    def test_a_camelcase_class_is_its_own_wire_name(self):
        for name, cls in sorted(self._tools().items()):
            if name[:1].islower():
                with self.subTest(tool=name):
                    self.assertEqual(getattr(cls, "name", None), name)

    def test_a_pascalcase_class_names_its_wire_form_separately(self):
        for name, cls in sorted(self._tools().items()):
            if name[:1].isupper():
                with self.subTest(tool=name):
                    self.assertNotEqual(getattr(cls, "name", ""), name)


class TestTheRestToolDropsNothing(unittest.TestCase):
    """The plan listed `response_field` and the REST auth token as silently
    dropped. Both are read and applied — the entry was stale, and a stale
    plan item costs somebody an afternoon rediscovering there is nothing to
    fix."""

    def test_response_field_is_consumed(self):
        from yait_aichain.tools.rest_api import RestApiTool
        source = inspect.getsource(RestApiTool)
        self.assertIn("self._response_field", source)
        self.assertIn("result.get(self._response_field", source)

    def test_auth_and_static_headers_are_applied(self):
        from yait_aichain.tools.rest_api import RestApiTool
        source = inspect.getsource(RestApiTool)
        self.assertIn("self._apply_auth(headers", source)
        self.assertIn("headers.update(self._static_headers)", source)


if __name__ == "__main__":
    unittest.main()
