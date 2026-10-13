"""
Every tool says what it risks; none inherits the default by omission.

A tool that declared nothing inherited `write`, and the shipped policy allows
`write`: `vector_delete` — a bulk delete by a filter the model writes — ran
without a question, as did transcription, which sends a local file out. And
`PermissionPolicy({"destructve": "deny"})`, misspelt, was a rule that matched
nothing and left `destructive` at its default in silence.
"""

import importlib
import inspect
import pkgutil
import unittest

import yait_aichain
from yait_aichain import PermissionPolicy
from yait_aichain.tools._base import Tool

#: Classes that set their risk per instance, from what they wrap or reach.
PER_INSTANCE = {"FunctionTool", "Gate", "MCPTool", "RestApiTool", "_VectorTool"}


class TestRisk(unittest.TestCase):

    def test_every_tool_class_declares_one(self):
        silent = []
        for info in pkgutil.walk_packages(yait_aichain.__path__, "yait_aichain."):
            try:
                module = importlib.import_module(info.name)
            except ImportError:
                continue
            for name, cls in vars(module).items():
                if (inspect.isclass(cls) and issubclass(cls, Tool) and cls is not Tool
                        and cls.__module__ == module.__name__ and name not in PER_INSTANCE):
                    origin = next(k for k in cls.__mro__ if "risk" in k.__dict__)
                    if origin is Tool:
                        silent.append(f"{module.__name__}.{name}")
        self.assertEqual(silent, [], "these inherit Tool's default risk without saying so")

    def test_the_destructive_ones_are_destructive(self):
        from yait_aichain.utilities.vectors import VectorDeleteTool
        self.assertEqual(VectorDeleteTool.risk, "destructive")

    def test_a_misspelt_class_is_refused(self):
        with self.assertRaisesRegex(ValueError, "destructve"):
            PermissionPolicy({"destructve": "deny"})


if __name__ == "__main__":
    unittest.main()
