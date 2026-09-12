"""
The library does not import itself through the layout it had before 2.0.

`sttQwen()` could not be constructed: it imported `clients._families.qwen`,
a top-level package that stopped existing when the library moved under
`yait_aichain`. The suite never noticed, because `tests/conftest.py` used to
put the old layout back on `sys.path` — the test environment revived exactly
the thing that was broken. The same spelling sat in some eighty docstring
examples, which is where a reader (or an agent) copies imports from.

Two checks: no source line in the package imports the flat layout, live or in
a docstring; and every exported Tool whose constructor needs nothing but a key
can actually be built.
"""

import inspect
import os
import re
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
FLAT = re.compile(
    r"^\s*(from|import)\s+(tools|models|skills|chain|pool|agent|state|clients)(\.|\s|$)")

KEYS = {k: "test-key" for k in (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_AI_API_KEY", "XAI_API_KEY",
    "PERPLEXITY_API_KEY", "BRAVE_SEARCH_API_KEY", "SERPAPI_API_KEY",
    "DASHSCOPE_API_KEY")}


class TestNoFlatImports(unittest.TestCase):

    def test_no_source_line_imports_the_pre_2_0_layout(self):
        hits = []
        for path in sorted((ROOT / "yait_aichain").rglob("*.py")):
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if FLAT.match(line):
                    hits.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
        self.assertEqual(hits, [], "\n".join(hits))


class TestEveryKeyOnlyToolConstructs(unittest.TestCase):

    def test_construct(self):
        import yait_aichain.tools as tools
        from yait_aichain.tools import Tool
        built = 0
        with mock.patch.dict(os.environ, KEYS):
            for name in sorted(dir(tools)):
                cls = getattr(tools, name)
                if not (inspect.isclass(cls) and issubclass(cls, Tool)) or cls is Tool:
                    continue
                required = [p for p in list(inspect.signature(cls.__init__).parameters.values())[1:]
                            if p.default is p.empty
                            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
                if required:
                    continue                    # needs a store, a server, a tool
                with self.subTest(tool=name):
                    cls()
                    built += 1
        self.assertGreater(built, 20)

    def test_qwen_speech_resolves_a_region(self):
        from yait_aichain.tools import ttsQwen, sttQwen
        with mock.patch.dict(os.environ, KEYS):
            self.assertIn("dashscope", ttsQwen(region="us")._base_url())
            self.assertIn("dashscope", sttQwen()._BASE_URL)


if __name__ == "__main__":
    unittest.main()
