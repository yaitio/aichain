"""
tests.tools.test_tool
======================

Unit tests for tools._base.Tool and tools._base.ToolResult,
and for tools.markitdown.MarkItDownTool.

All tests are pure — no network calls.  MarkItDown is mocked where needed
so the tests run even if the library is not installed.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools._base     import Tool, ToolResult
from tools.markitdown import MarkItDownTool


# ---------------------------------------------------------------------------
# Helpers — minimal concrete tool for testing the base class
# ---------------------------------------------------------------------------

class _AddTool(Tool):
    """Adds two numbers.  Used to test base-class behaviour."""
    name        = "add"
    description = "Add two numbers."
    parameters  = {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First operand."},
            "b": {"type": "number", "description": "Second operand."},
        },
        "required": ["a", "b"],
    }

    def run(self, a, b, **_) -> float:
        return a + b


class _BrokenTool(Tool):
    """Always raises RuntimeError — used to test error handling."""
    name        = "broken"
    description = "Always fails."
    parameters  = {"type": "object", "properties": {}, "required": []}

    def run(self, **_):
        raise RuntimeError("intentional failure")


class _NoParamsTool(Tool):
    """Tool with no required parameters."""
    name        = "noop"
    description = "Does nothing."
    parameters  = {"type": "object", "properties": {}, "required": []}

    def run(self, **_) -> str:
        return "ok"


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

class TestToolResult(unittest.TestCase):

    def test_success_is_truthy(self):
        r = ToolResult(success=True, output="hello")
        self.assertTrue(r)

    def test_failure_is_falsy(self):
        r = ToolResult(success=False, output=None, error="oops")
        self.assertFalse(r)

    def test_success_output_accessible(self):
        r = ToolResult(success=True, output={"key": 42})
        self.assertEqual(r.output["key"], 42)

    def test_failure_error_accessible(self):
        r = ToolResult(success=False, output=None, error="bad input")
        self.assertEqual(r.error, "bad input")

    def test_error_defaults_to_none(self):
        r = ToolResult(success=True, output="x")
        self.assertIsNone(r.error)

    def test_repr_success(self):
        r = ToolResult(success=True, output="hello")
        self.assertIn("True", repr(r))
        self.assertIn("hello", repr(r))

    def test_repr_failure(self):
        r = ToolResult(success=False, output=None, error="msg")
        self.assertIn("False", repr(r))
        self.assertIn("msg", repr(r))

    def test_repr_truncates_long_output(self):
        r = ToolResult(success=True, output="x" * 200)
        self.assertIn("...", repr(r))


# ---------------------------------------------------------------------------
# Tool base class — class attributes and __repr__
# ---------------------------------------------------------------------------

class TestToolBase(unittest.TestCase):

    def test_name_stored(self):
        self.assertEqual(_AddTool().name, "add")

    def test_description_stored(self):
        self.assertIn("Add", _AddTool().description)

    def test_parameters_stored(self):
        t = _AddTool()
        self.assertIn("a", t.parameters["properties"])
        self.assertIn("b", t.parameters["properties"])

    def test_repr_contains_name(self):
        self.assertIn("add", repr(_AddTool()))

    def test_base_run_raises_not_implemented(self):
        base = object.__new__(Tool)
        base.name = "x"
        with self.assertRaises(NotImplementedError):
            base.run()


# ---------------------------------------------------------------------------
# Tool._validate
# ---------------------------------------------------------------------------

class TestToolValidate(unittest.TestCase):

    def test_valid_kwargs_pass(self):
        _AddTool()._validate({"a": 1, "b": 2})   # should not raise

    def test_missing_required_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _AddTool()._validate({"a": 1})        # missing "b"
        self.assertIn("b", str(ctx.exception))

    def test_error_mentions_tool_name(self):
        with self.assertRaises(ValueError) as ctx:
            _AddTool()._validate({})
        self.assertIn("add", str(ctx.exception))

    def test_no_required_always_passes(self):
        _NoParamsTool()._validate({})             # should not raise
        _NoParamsTool()._validate({"extra": 99})  # extra keys are fine


# ---------------------------------------------------------------------------
# Tool.__call__ — safe execution wrapper
# ---------------------------------------------------------------------------

class TestToolCall(unittest.TestCase):

    def test_success_returns_tool_result(self):
        result = _AddTool()(a=3, b=4)
        self.assertIsInstance(result, ToolResult)

    def test_success_result_is_truthy(self):
        self.assertTrue(_AddTool()(a=1, b=2))

    def test_success_output_correct(self):
        self.assertEqual(_AddTool()(a=10, b=5).output, 15)

    def test_success_error_is_none(self):
        self.assertIsNone(_AddTool()(a=1, b=1).error)

    def test_runtime_error_caught(self):
        result = _BrokenTool()()
        self.assertFalse(result)
        self.assertIsNotNone(result.error)
        self.assertIn("intentional", result.error)

    def test_validation_error_caught(self):
        result = _AddTool()(a=1)          # missing "b"
        self.assertFalse(result)
        self.assertIn("b", result.error)

    def test_never_raises(self):
        # __call__ must not propagate any exception
        try:
            _BrokenTool()()
            _AddTool()()                  # missing required params
        except Exception as exc:
            self.fail(f"__call__ raised unexpectedly: {exc}")

    def test_no_params_tool_succeeds(self):
        result = _NoParamsTool()()
        self.assertTrue(result)
        self.assertEqual(result.output, "ok")


# ---------------------------------------------------------------------------
# Tool.run — direct (unsafe) execution
# ---------------------------------------------------------------------------

class TestToolRun(unittest.TestCase):

    def test_run_returns_raw_output(self):
        self.assertEqual(_AddTool().run(a=2, b=3), 5)

    def test_run_propagates_exceptions(self):
        with self.assertRaises(RuntimeError):
            _BrokenTool().run()


# ---------------------------------------------------------------------------
# Tool.schema — OpenAI function-calling format
# ---------------------------------------------------------------------------

class TestToolSchema(unittest.TestCase):

    def setUp(self):
        self.schema = _AddTool().schema()

    def test_top_level_type_is_function(self):
        self.assertEqual(self.schema["type"], "function")

    def test_function_name_matches(self):
        self.assertEqual(self.schema["function"]["name"], "add")

    def test_function_description_present(self):
        self.assertIn("Add", self.schema["function"]["description"])

    def test_parameters_key_present(self):
        self.assertIn("parameters", self.schema["function"])

    def test_parameters_has_properties(self):
        self.assertIn("properties", self.schema["function"]["parameters"])

    def test_required_list_present(self):
        self.assertIn("required", self.schema["function"]["parameters"])
        self.assertIn("a", self.schema["function"]["parameters"]["required"])
        self.assertIn("b", self.schema["function"]["parameters"]["required"])


# ---------------------------------------------------------------------------
# MarkItDownTool — construction and parameter declaration
# ---------------------------------------------------------------------------

class TestMarkItDownToolConstruct(unittest.TestCase):

    def test_name(self):
        self.assertEqual(MarkItDownTool().name, "markitdown")

    def test_description_non_empty(self):
        self.assertTrue(MarkItDownTool().description)

    def test_source_is_required(self):
        self.assertIn("source", MarkItDownTool().parameters["required"])

    def test_output_path_is_optional(self):
        required = MarkItDownTool().parameters["required"]
        self.assertNotIn("output_path", required)

    def test_converter_lazily_initialised(self):
        tool = MarkItDownTool()
        self.assertIsNone(tool._md)   # not yet created

    def test_repr_no_llm(self):
        r = repr(MarkItDownTool())
        self.assertIn("markitdown", r)
        self.assertNotIn("llm_model", r)

    def test_repr_with_llm(self):
        tool = MarkItDownTool(llm_client=MagicMock(), llm_model="gpt-4o")
        self.assertIn("gpt-4o", repr(tool))


# ---------------------------------------------------------------------------
# MarkItDownTool — import error handling
# ---------------------------------------------------------------------------

class TestMarkItDownImportError(unittest.TestCase):

    def test_run_raises_import_error_when_not_installed(self):
        tool = MarkItDownTool()
        with patch.dict("sys.modules", {"markitdown": None}):
            tool._md = None   # force re-creation
            with self.assertRaises(ImportError) as ctx:
                tool.run(source="file.pdf")
        self.assertIn("pip install markitdown", str(ctx.exception))

    def test_call_returns_failure_on_import_error(self):
        tool = MarkItDownTool()
        with patch.dict("sys.modules", {"markitdown": None}):
            tool._md = None
            result = tool(source="file.pdf")
        self.assertFalse(result)
        self.assertIn("markitdown", result.error)


# ---------------------------------------------------------------------------
# MarkItDownTool — run() with mocked MarkItDown
# ---------------------------------------------------------------------------

def _mock_md_result(text: str, title: str = ""):
    r = MagicMock()
    r.text_content = text
    r.title        = title
    return r


def _make_tool_with_mock(text: str, title: str = "") -> tuple:
    """Return (tool, mock_md_instance) with a pre-configured converter."""
    mock_md = MagicMock()
    mock_md.convert.return_value = _mock_md_result(text, title)

    tool    = MarkItDownTool()
    tool._md = mock_md        # inject mock directly — skip import
    return tool, mock_md


class TestMarkItDownRun(unittest.TestCase):

    def test_returns_markdown_string(self):
        tool, _ = _make_tool_with_mock("# Hello\n\nWorld")
        result  = tool.run(source="doc.pdf")
        self.assertEqual(result, "# Hello\n\nWorld")

    def test_calls_convert_with_source(self):
        tool, mock_md = _make_tool_with_mock("text")
        tool.run(source="report.pdf")
        mock_md.convert.assert_called_once_with("report.pdf")

    def test_saves_to_file_when_output_path_given(self):
        tool, _ = _make_tool_with_mock("# Saved")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "output.md")
            tool.run(source="doc.pdf", output_path=out)
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "# Saved")

    def test_creates_parent_dirs_for_output_path(self):
        tool, _ = _make_tool_with_mock("text")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "deep", "nested", "output.md")
            tool.run(source="doc.pdf", output_path=out)
            self.assertTrue(os.path.exists(out))

    def test_no_file_written_without_output_path(self):
        tool, _ = _make_tool_with_mock("text")
        with tempfile.TemporaryDirectory() as tmpdir:
            # change cwd so any accidental writes land in tmpdir
            prev = os.getcwd()
            os.chdir(tmpdir)
            try:
                tool.run(source="doc.pdf")
            finally:
                os.chdir(prev)
            self.assertEqual(os.listdir(tmpdir), [])

    def test_call_style_returns_tool_result(self):
        tool, _ = _make_tool_with_mock("# OK")
        result  = tool(source="file.html")
        self.assertIsInstance(result, ToolResult)
        self.assertTrue(result)
        self.assertEqual(result.output, "# OK")

    def test_call_missing_source_returns_failure(self):
        tool, _ = _make_tool_with_mock("text")
        result  = tool()        # missing "source"
        self.assertFalse(result)
        self.assertIn("source", result.error)

    def test_schema_is_openai_compatible(self):
        schema = MarkItDownTool().schema()
        fn     = schema["function"]
        self.assertEqual(fn["name"], "markitdown")
        self.assertIn("source",      fn["parameters"]["properties"])
        self.assertIn("output_path", fn["parameters"]["properties"])
        self.assertEqual(fn["parameters"]["required"], ["source"])


# ---------------------------------------------------------------------------
# MarkItDownTool — real conversion (no mock, uses actual markitdown)
# ---------------------------------------------------------------------------

class TestMarkItDownReal(unittest.TestCase):
    """Integration-style tests that use the real markitdown library."""

    def _tool(self) -> MarkItDownTool:
        return MarkItDownTool()

    def test_converts_html_string_to_markdown(self):
        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            fh.write("<h1>Hello</h1><p>World</p>")
            path = fh.name
        try:
            result = self._tool().run(source=path)
            self.assertIn("Hello", result)
            self.assertIn("World", result)
        finally:
            os.unlink(path)

    def test_converts_plain_text_file(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            fh.write("Plain text content here.")
            path = fh.name
        try:
            result = self._tool().run(source=path)
            self.assertIn("Plain text", result)
        finally:
            os.unlink(path)

    def test_converter_instance_reused(self):
        tool = self._tool()
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w"
        ) as fh:
            fh.write("hello")
            path = fh.name
        try:
            tool.run(source=path)
            first_md  = tool._md
            tool.run(source=path)
            second_md = tool._md
            self.assertIs(first_md, second_md)   # same instance reused
        finally:
            os.unlink(path)

    def test_file_not_found_propagates(self):
        with self.assertRaises(Exception):
            self._tool().run(source="/definitely/does/not/exist.pdf")

    def test_call_file_not_found_returns_failure(self):
        result = self._tool()(source="/does/not/exist.pdf")
        self.assertFalse(result)
        self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main()
