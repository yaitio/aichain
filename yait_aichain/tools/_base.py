"""
tools._base
===========

``Tool`` — base class for all tool implementations.
``ToolResult`` — uniform return type for every tool call.

Design
------
Every concrete tool:

  1. Declares three class-level attributes:

       name        : str   — unique machine-readable identifier
       description : str   — one-sentence explanation (used by agents / LLMs)
       parameters  : dict  — JSON Schema ``object`` describing accepted inputs

  2. Implements ``run(input, options=None) -> Any`` — the actual logic.

That is the complete contract.  Everything else (validation, error wrapping,
OpenAI-compatible schema generation) is provided by this base class.

Two call styles
---------------
  * ``tool(input, options)``       → always returns a :class:`ToolResult`; never
    raises.  Exceptions from ``run()`` are caught and surfaced as
    ``ToolResult(success=False, error="...")``.

  * ``tool.run(input, options)``   → returns the raw output; propagates exceptions.
    Use this when you want normal Python error handling.

Parameters schema
-----------------
``parameters`` must follow JSON Schema's ``object`` shape so it can be
forwarded directly to OpenAI function-calling or Anthropic tool-use APIs::

    parameters = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "string",
                "description": "File path or URL to process.",
            },
            "options": {
                "type":        "object",
                "description": "Optional configuration.",
            },
        },
        "required": ["input"],
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """
    Uniform return type for every :class:`Tool` call.

    Attributes
    ----------
    success : bool
        ``True`` when ``run()`` completed without raising an exception.
    output  : Any
        The value returned by ``run()``.  ``None`` when *success* is
        ``False``.
    error   : str | None
        Human-readable error message.  ``None`` when *success* is ``True``.

    Examples
    --------
    ::

        result = tool(input="report.pdf")

        if result:                       # bool(result) == result.success
            print(result.output)
        else:
            print("Error:", result.error)
    """

    success: bool
    output:  Any
    error:   str | None = None

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        if self.success:
            preview = repr(self.output)
            if len(preview) > 60:
                preview = preview[:57] + "..."
            return f"ToolResult(success=True, output={preview})"
        return f"ToolResult(success=False, error={self.error!r})"


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class Tool:
    """
    Base class for all tool implementations.

    Subclasses must set the three class-level attributes and implement
    :meth:`run`.  Everything else is inherited.

    Class attributes
    ----------------
    name : str
        Unique, machine-readable identifier, e.g. ``"markitdown"``.
        Used as the function name in OpenAI / Anthropic schemas.

    description : str
        One-sentence description of what the tool does.  This is what an
        LLM reads to decide whether to use the tool.

    parameters : dict
        JSON Schema ``object`` describing the tool's accepted inputs.
        Must contain a ``"properties"`` key and optionally a ``"required"``
        list.  See module docstring for an example.

    Examples
    --------
    Defining a custom tool::

        from tools import Tool, ToolResult

        class ReverseTextTool(Tool):
            name        = "reverse_text"
            description = "Reverse the characters in a string."
            parameters  = {
                "type": "object",
                "properties": {
                    "input":   {"type": "string", "description": "Text to reverse."},
                    "options": {"type": "object", "description": "Unused."},
                },
                "required": ["input"],
            }

            def run(self, input: str, options=None) -> str:
                return input[::-1]

        tool   = ReverseTextTool()
        result = tool(input="hello")    # ToolResult(success=True, output='olleh')
        raw    = tool.run(input="hi")   # 'ih'  (raises on error)
    """

    name:        str  = ""
    description: str  = ""
    parameters:  dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, input=None, options=None) -> Any:
        """
        Execute the tool and return the raw result.

        Must be implemented by every subclass.  May raise any exception —
        callers that want safe execution should use ``__call__`` instead.

        Parameters
        ----------
        input : Any
            The primary input for the tool, matching the ``"input"`` key
            declared in ``self.parameters["properties"]``.
        options : dict | None
            Optional configuration dict.  Keys depend on the concrete tool.

        Returns
        -------
        Any
            Tool-specific output (string, dict, list, bytes, …).

        Raises
        ------
        NotImplementedError
            Always, from the base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement run()"
        )

    def __call__(self, input=None, options=None) -> ToolResult:
        """
        Validate inputs, run the tool, and return a :class:`ToolResult`.

        Unlike :meth:`run`, this method **never raises**.  Any exception
        (including ``ValueError`` from validation) is caught and returned
        as ``ToolResult(success=False, error=...)``.

        Parameters
        ----------
        input : Any
            Primary input for the tool.
        options : dict | None
            Optional configuration dict.

        Returns
        -------
        ToolResult
            ``success=True``  + raw output on success.
            ``success=False`` + error message on failure.
        """
        try:
            self._validate(input, options)
            output = self.run(input, options)
            return ToolResult(success=True, output=output)
        except Exception as exc:
            return ToolResult(success=False, output=None, error=str(exc))

    def schema(self) -> dict:
        """
        Return an OpenAI-compatible function-calling schema for this tool.

        The returned dict can be passed directly to the ``tools`` parameter
        of the OpenAI Chat Completions API or the Anthropic Messages API.

        Returns
        -------
        dict
            ``{"type": "function", "function": {"name": ..., "description":
            ..., "parameters": ...}}``.

        Example
        -------
        ::

            import openai
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Convert report.pdf"}],
                tools=[tool.schema()],
            )
        """
        return {
            "type": "function",
            "function": {
                "name":        self.name,
                "description": self.description,
                "parameters":  self.parameters,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, input, options) -> None:
        """
        Check that required parameters are satisfied.

        If ``"input"`` is listed in ``parameters["required"]`` and *input*
        is ``None``, a ``ValueError`` is raised.

        Raises
        ------
        ValueError
            When a required parameter is missing.
        """
        required = self.parameters.get("required", [])
        if "input" in required and input is None:
            raise ValueError(
                f"Tool '{self.name}' missing required parameter: 'input'"
            )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
