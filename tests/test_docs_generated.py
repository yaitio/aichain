"""
Every table in the docs that has a source of truth is generated from it.

The 2.6.1 hand sweep went stale on four pages within a week. The model
registry page listed 67 of 90 models and a provider tuple of seven; the
environment page named four keys the library does not read and missed four it
does; the tools reference covered seven of thirty-two tools and documented a
`query=` keyword none of them accept. `parameters.md` was the exception,
because a run produces it. `scripts/docs.py` gives every other data table the
same property, and this test is what makes it a property rather than a habit.
"""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("docs_generator", ROOT / "scripts" / "docs.py")
docs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(docs)


class TestGeneratedPagesAreCurrent(unittest.TestCase):

    def test_every_page_equals_its_regeneration(self):
        for path, text in docs.targets().items():
            with self.subTest(page=path.relative_to(ROOT).as_posix()):
                self.assertTrue(path.exists(), "page missing — run python scripts/docs.py")
                self.assertEqual(path.read_text(), text,
                                 "stale — run python scripts/docs.py")

    def test_every_exported_tool_has_a_page(self):
        docs._check_paging()                    # raises SystemExit when one is missing

    def test_a_lost_marker_is_an_error(self):
        rel = "docs/primitives/pool.md"
        text = (ROOT / rel).read_text().replace("<!-- g:policy -->", "", 1)
        with self.assertRaises(SystemExit):
            docs.render(rel, text)


class TestTheNumbersAreTheRegistry(unittest.TestCase):
    """The headline counts are the thing a reader quotes."""

    def test_counts(self):
        from yait_aichain.models import registry
        readme = (ROOT / "README.md").read_text()
        self.assertIn(f"{len(registry.models())} models from "
                      f"{len(registry.providers())} cloud providers", readme)


if __name__ == "__main__":
    unittest.main()


class TestToolPagesCallToolsTheWayTheyAreCalled(unittest.TestCase):
    """The hand-written usage on the tool pages called every tool with keywords
    no tool accepts — `tool(query=…, max_results=5)`, `run(source=…)` — for the
    whole 2.x line, because a Tool is called with `input` and `options` and
    nothing checked the pages against that. Thirty-six calls, seven pages.

    Checked here: every call to a tool constructed in the same block passes only
    `input`/`options`; a literal `options` dict names only keys the tool's schema
    declares; a Chain step's input map names only parameters the tool has."""

    PAGES = sorted((ROOT / "docs" / "tools-reference").glob("*.md"))

    def _tools_in(self, tree):
        import ast
        import yait_aichain.tools as tools
        bound = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and hasattr(tools, node.value.func.id)):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bound[target.id] = getattr(tools, node.value.func.id)
        return bound

    @staticmethod
    def _schema(cls):
        import inspect
        import os
        from unittest import mock
        params = inspect.signature(cls.__init__).parameters
        if any(p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
               for p in list(params.values())[1:]):
            return None
        keys = {k: "doc" for k in ("OPENAI_API_KEY", "PERPLEXITY_API_KEY",
                                   "BRAVE_SEARCH_API_KEY", "SERPAPI_API_KEY",
                                   "XAI_API_KEY", "DASHSCOPE_API_KEY", "GOOGLE_AI_API_KEY")}
        with mock.patch.dict(os.environ, keys):
            return cls().parameters

    def test_calls(self):
        import ast
        import re
        import yait_aichain.tools as tools
        problems = []
        for page in self.PAGES:
            text = page.read_text()
            for block in re.findall(r"```python\n(.*?)```", text, re.S):
                try:
                    tree = ast.parse(block)
                except SyntaxError:
                    continue
                bound = self._tools_in(tree)
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    cls = None
                    f = node.func
                    if isinstance(f, ast.Name) and f.id in bound:
                        cls = bound[f.id]                       # tool(...)
                    elif (isinstance(f, ast.Attribute) and f.attr == "run"
                          and isinstance(f.value, ast.Name) and f.value.id in bound):
                        cls = bound[f.value.id]                 # tool.run(...)
                    elif (isinstance(f, ast.Call) and isinstance(f.func, ast.Name)
                          and hasattr(tools, f.func.id)):
                        cls = getattr(tools, f.func.id)         # Tool()(...)
                    if cls is not None:
                        schema = self._schema(cls)
                        for kw in node.keywords:
                            if kw.arg not in ("input", "options"):
                                problems.append(f"{page.name}:{node.lineno}: {kw.arg}=")
                            elif (kw.arg == "options" and schema and isinstance(kw.value, ast.Dict)):
                                declared = schema["properties"].get("options", {}).get("properties", {})
                                for k in kw.value.keys:
                                    if isinstance(k, ast.Constant) and k.value not in declared:
                                        problems.append(f"{page.name}:{node.lineno}: options[{k.value!r}]")
                    # (Tool(), "out", {"param": "var"}) inside Chain(steps=[...])
                    if isinstance(node.func, ast.Name) and node.func.id == "Chain":
                        for step in ast.walk(node):
                            if (isinstance(step, ast.Tuple) and len(step.elts) >= 3
                                    and isinstance(step.elts[0], ast.Call)
                                    and isinstance(step.elts[0].func, ast.Name)
                                    and hasattr(tools, step.elts[0].func.id)
                                    and isinstance(step.elts[2], ast.Dict)):
                                schema = self._schema(getattr(tools, step.elts[0].func.id))
                                if not schema:
                                    continue
                                for k in step.elts[2].keys:
                                    if isinstance(k, ast.Constant) and k.value not in schema["properties"]:
                                        problems.append(f"{page.name}:{step.lineno}: input map {k.value!r}")
        self.assertEqual(problems, [], "\n".join(problems))
