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

import warnings
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

class Record(dict):
    """
    A tool's result: a dict for code, and text for a model or a prompt.

    Three readers take the same result. Code reads the fields —
    ``found["results"]``. An agent shows the model ``str(record)``, the tool's
    own rendering, rather than the dict as JSON. A Skill that substitutes the
    result into its prompt gets that rendering too, since substitution is
    ``str(value)``. And a Chain stores a record under its step's key, where a
    plain dict is merged into the variables: a record is one result, and its
    keys (``results``, ``cost``) would collide between two steps.

    The rendering is not stored: a record written to a state store and read
    back is a plain dict.

    Parameters
    ----------
    data : dict
        The fields, for code.
    render : callable
        Turns them into the text a model is shown; ``str(record)`` calls it.
    """

    __slots__ = ("_render",)

    def __init__(self, data: dict, render) -> None:
        super().__init__(data)
        self._render = render

    def __str__(self) -> str:
        return self._render(self)

    def __copy__(self):
        return Record(dict(self), self._render)

    def __deepcopy__(self, memo):
        import copy
        return Record(copy.deepcopy(dict(self), memo), self._render)

    def __reduce__(self):            # pickled as the dict it is; a renderer is not data
        return (dict, (dict(self),))


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

        from yait_aichain.tools import Tool, ToolResult

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

    Returning a dict inside a Chain
    -------------------------------
    A step's output key names a **string** result. When ``run`` returns a
    dict, the keys are spread into the run's variables instead and the step
    name is not used::

        Chain(steps=[(CollectFacts(), "facts")])   # returns {"brand", "decks"}
        # accumulated: {"brand": ..., "decks": ...}   — no "facts" key

    Spreading is deliberate: it is what lets one step feed several named
    inputs downstream. What it costs is that two steps returning the same key
    overwrite each other silently, and a later step asking for ``facts`` gets
    a missing-argument error rather than a dict.

    Wrap the payload when the step name is what the next step expects::

        def run(self, ...) -> dict:
            return {"facts": {"brand": ..., "decks": ...}}
    """

    name:        str  = ""
    description: str  = ""
    parameters:  dict = {}

    # Risk class consulted by an Agent's PermissionPolicy before the tool runs
    # (1.4.4). Default ``"write"``; tag mutating/external tools accordingly —
    # see ``tools._permissions`` for the vocabulary. Enforcement is opt-in: with
    # no policy attached this attribute is inert.
    risk:        str  = "write"

    # Declares that this tool can park the run by raising ``state.Suspend``
    # (``Wait``, ``Gate``). An engine reads it *before* the run: a `Chain`
    # always has a store, an `Agent` must be given one, and a `Pool` cannot
    # park a fan-out at all. Declared rather than discovered, because the
    # alternative is finding out at the moment the pause is needed and the
    # place to keep it does not exist.
    suspends:    bool = False

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

    def __call__(self, input=None, options=None, **kwargs) -> ToolResult:
        """
        Validate inputs, run the tool, and return a :class:`ToolResult`.

        Unlike :meth:`run`, this method **never raises**.  Any exception
        (including ``ValueError`` from validation) is caught and returned
        as ``ToolResult(success=False, error=...)``.

        A tool with its own parameters — ``def run(self, email, options=None)``
        — is called by name, ``tool(email="a@b.c")``, the way an Agent or a
        Chain calls ``run(**kwargs)``. Until this, only ``input=`` and
        ``options=`` were accepted, so such a tool could not be called safely
        at all: ``tool(email=...)`` raised a bare ``TypeError``. `MCPTool`
        already forwarded keywords the same way.

        Parameters
        ----------
        input : Any
            Primary input for the tool.
        options : dict | None
            Optional configuration dict.
        **kwargs
            The tool's own parameters, by name.

        Returns
        -------
        ToolResult
            ``success=True``  + raw output on success.
            ``success=False`` + error message on failure.
        """
        try:
            if kwargs:
                if input is not None:
                    kwargs["input"] = input
                if options is not None:
                    kwargs["options"] = options
                missing = [k for k in self.parameters.get("required", []) if k not in kwargs]
                if missing:
                    raise ValueError(f"Tool '{self.name}' missing required parameter(s): "
                                     + ", ".join(repr(k) for k in missing))
                output = self.run(**kwargs)
            else:
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
        # A warning rather than a raise, and the asymmetry with `check_args`
        # is deliberate. There the caller is a model, the message goes back
        # into its attempt budget, and it retries; here the caller is a person
        # whose tool subclass may legitimately read an option its schema does
        # not advertise. Refusing their call over the library's reading of
        # their own schema would be the library knowing better. Saying so is
        # the most it is entitled to.
        if isinstance(options, dict):
            if unknown := self.unknown_keys({"options": options}):
                warnings.warn(self._unknown_message(unknown),
                              RuntimeWarning, stacklevel=3)

    def unknown_keys(self, kwargs: dict) -> list:
        """Keys the schema does not declare, as ``"options.recencyy"`` paths.

        The schemas were already there and nothing read them. `searchPerplexity`
        declares seven option keys — `max_results`, `recency`, `domains`,
        `country`, `language`, `after_date`, `before_date` — and
        `options={"recencyy": "day"}` passed validation, reached the tool, and
        was dropped on the floor: a search ran unfiltered and answered
        confidently. A typo cost a run and left no trace.

        One level into a declared object, and no further. That is where the
        gap was — the top level already fails loudly, because an unexpected
        keyword reaches `run(**kwargs)` and Python raises — and a schema is
        only authority over what it actually describes. A nested object with
        no `properties` of its own is a free-form dict by declaration, and is
        left alone.

        Parameters
        ----------
        kwargs : dict
            The arguments of a call.
        """
        declared = self.parameters.get("properties", {})
        unknown  = [k for k in kwargs if k not in declared]
        for key, spec in declared.items():
            nested = (spec or {}).get("properties")
            value  = kwargs.get(key)
            if nested and isinstance(value, dict):
                unknown += [f"{key}.{k}" for k in value if k not in nested]
        return sorted(unknown)

    def _unknown_message(self, unknown: list) -> str:
        declared = self.parameters.get("properties", {})
        known = sorted(declared)
        for key, spec in declared.items():
            for nested in sorted((spec or {}).get("properties") or {}):
                known.append(f"{key}.{nested}")
        return (f"Invalid call to tool '{self.name}': unknown "
                f"argument(s) {unknown}. Accepted: {known}.")

    def check_args(self, kwargs: dict) -> str | None:
        """
        Validate keyword arguments against ``parameters`` (1.4.4 tool-call
        repair). Returns a model-readable remediation message when a required
        argument is missing, or ``None`` when the call looks well-formed.

        Lightweight on purpose — required-presence only, no full JSON-Schema
        type enforcement (which would need an extra dependency). An Agent feeds
        the returned message back to the model as a structured observation so it
        can correct the call within the step's attempt budget.

        Parameters
        ----------
        kwargs : dict
            The arguments of a call.
        """
        required   = self.parameters.get("required", [])
        properties = self.parameters.get("properties", {})
        missing    = [k for k in required if k not in kwargs or kwargs.get(k) is None]
        if missing:
            return (
                f"Invalid call to tool '{self.name}': missing required "
                f"argument(s) {missing}. Expected parameters: "
                f"{sorted(properties) or 'see the tool schema'}."
            )
        # A name the schema does not carry is a mistake, not a request, and
        # the model can fix it inside the step's attempt budget — which is
        # exactly what this message is for. Missing is reported first because
        # it is the more basic failure.
        if unknown := self.unknown_keys(kwargs):
            return self._unknown_message(unknown)
        return None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# @tool — any function as a tool
# ---------------------------------------------------------------------------

import functools
import inspect as _inspect
import re as _re
import types as _types
import typing as _typing

#: What providers accept as a function name.
_TOOL_NAME = _re.compile(r"[A-Za-z0-9_-]{1,64}")
#: ``text: the text to count.`` or ``text (str): …`` — a parameter's line in a docstring.
_PARAM_LINE = _re.compile(r"^\s*\*{0,2}([A-Za-z_][A-Za-z0-9_]*)\*{0,2}\s*(?:\([^)]*\))?\s*:\s*(.+?)\s*$")
#: ``text : str`` — a NumPy-style parameter header; its description is the
#: indented lines under it, not the type after the colon.
_NUMPY_HEAD = _re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s+:\s*\S.*$")

_PRIMITIVE = {bool: "boolean", int: "integer", float: "number", str: "string"}


def _param_docs(doc: str) -> dict:
    """Each parameter's description, from a ``name: text`` line or from the
    NumPy form this library itself writes (``name : type`` over an indented
    description)."""
    lines = doc.splitlines()[1:]
    if any(_re.match(r"^\s*-{3,}\s*$", line) for line in lines):
        found, i = {}, 0
        while i < len(lines):
            head = _NUMPY_HEAD.match(lines[i])
            i += 1
            if not head:
                continue
            depth, body = len(head.group(1)), []
            while i < len(lines) and lines[i].strip() and \
                    len(lines[i]) - len(lines[i].lstrip()) > depth:
                body.append(lines[i].strip())
                i += 1
            if body:
                found[head.group(2)] = " ".join(body)
        return found
    return {m.group(1): m.group(2) for line in lines if (m := _PARAM_LINE.match(line))}


def _json_type(tp, where: str) -> dict:
    """The JSON Schema of one annotation — or a TypeError naming what is not supported."""
    if tp is _inspect.Parameter.empty or tp is _typing.Any:
        return {"type": "string"}
    origin, args = _typing.get_origin(tp), _typing.get_args(tp)
    if origin is _typing.Union or (hasattr(_types, "UnionType") and origin is _types.UnionType):
        rest = [a for a in args if a is not type(None)]
        if len(rest) == 1:
            return _json_type(rest[0], where)
        raise TypeError(f"{where}: a union of {rest} has no single JSON type; annotate one type, "
                        "or Optional of one")
    if origin is _typing.Literal:
        kinds = {type(a) for a in args}
        base = _json_type(kinds.pop(), where) if len(kinds) == 1 else {}
        return {**base, "enum": list(args)}
    if tp in _PRIMITIVE:
        return {"type": _PRIMITIVE[tp]}
    if tp in (list, tuple, set) or origin in (list, tuple, set):
        return {"type": "array", **({"items": _json_type(args[0], where)} if args else {})}
    if tp is dict or origin is dict:
        return {"type": "object"}
    raise TypeError(f"{where}: no JSON schema for {tp!r} — annotate with str, int, float, bool, "
                    "list, dict, Literal, or Optional of one of them")


class FunctionTool(Tool):
    """
    A function as a :class:`Tool` — what :func:`tool` returns.

    The function's name is the tool's ``name``, the first line of its
    docstring the ``description``, and its parameters the schema: types from
    the annotations, required where there is no default, each described by
    its ``name: text`` line in the docstring. ``run(**arguments)`` calls the
    function and returns what it returns; the function itself is
    ``.__wrapped__``.

    Parameters
    ----------
    fn : callable
        The function to wrap.
    name : str | None, optional
        The tool's name; the function's by default.
    description : str | None, optional
        What a model reads; the docstring's first line by default.
    risk : str, optional
        The risk class; ``"write"`` by default.
    """

    def __init__(self, fn, *, name: str | None = None, description: str | None = None,
                 risk: str = "write") -> None:
        from ._permissions import RISK_CLASSES
        if not callable(fn):
            raise TypeError(f"@tool wraps a function; got {type(fn).__name__}")
        # A partial is its function with some arguments fixed: the name and
        # the docstring are the function's, the signature the partial's.
        inner = fn
        while isinstance(inner, functools.partial):
            inner = inner.func
        label = getattr(inner, "__name__", type(inner).__name__)
        self.name = name or label
        if not _TOOL_NAME.fullmatch(self.name):
            raise ValueError(f"tool name {self.name!r} must be 1 to 64 letters, digits, '_' or '-' — "
                             "what providers accept; pass name=")
        doc = _inspect.getdoc(inner) or ""
        self.description = description or (doc.strip().splitlines()[0].strip() if doc.strip() else "")
        if not self.description:
            raise ValueError(f"@tool {label}: no description — a model decides whether to call a "
                             "tool by it. Give the function a docstring, or pass description=")
        if risk not in RISK_CLASSES:
            raise ValueError(f"@tool {label}: risk {risk!r} is not one of {sorted(RISK_CLASSES)}")
        self.risk = risk

        sig = _inspect.signature(fn)
        try:
            hints = _typing.get_type_hints(inner) if not isinstance(inner, type) else {}
        except NameError as exc:
            raise TypeError(f"@tool {label}: an annotation names something not defined where "
                            f"the function is ({exc}); import it, or annotate with a builtin") from None
        lines = _param_docs(doc)
        props, required = {}, []
        for p in sig.parameters.values():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                raise TypeError(f"@tool {label}: *{p.name} has no schema — a model passes named "
                                "arguments; declare each one")
            annotation = hints.get(p.name, p.annotation)
            if annotation is p.empty and type(p.default) in (*_PRIMITIVE, list, dict):
                annotation = type(p.default)      # `count=3` is an integer, not a string
            spec = _json_type(annotation, f"@tool {label}, parameter {p.name!r}")
            if p.name in lines:
                spec["description"] = lines[p.name]
            props[p.name] = spec
            if p.default is p.empty:
                required.append(p.name)
        self.parameters = {"type": "object", "properties": props, "required": required}
        self.__wrapped__ = fn
        self.__doc__ = inner.__doc__
        self._takes_options = "options" in sig.parameters

    def run(self, *args, **kwargs):  # type: ignore[override]
        # Tool.__call__ passes (input, options) positionally when no keyword is
        # given; a function without an `options` parameter does not take the None.
        if len(args) == 2 and args[1] is None and not self._takes_options:
            args = args[:1]
        return self.__wrapped__(*args, **kwargs)


def tool(fn=None, *, name: str | None = None, description: str | None = None,
         risk: str = "write"):
    """
    Make any function a tool — code, a chain and an agent call it alike.

    ::

        @tool(risk="read")
        def word_count(text: str, unique: bool = False) -> dict:
            \"\"\"Count the words in a text.

            text: the text to count.
            unique: count each word once.
            \"\"\"
            words = text.split()
            return {"words": len(set(words)) if unique else len(words)}

        word_count.run(text="to be or not to be")     # {"words": 6}
        Agent(model, tools=[word_count])              # the model sees word_count(text, unique)

    ``name``, ``description`` and ``risk`` are the tool's settings; the
    function's parameters are its arguments. **``risk`` defaults to
    ``"write"``**, as on every tool: the library cannot tell what a function
    does, and assuming it only reads would let a permission policy pass it
    unasked — saying ``risk="read"`` is the author's claim to make. Usable as
    ``@tool``, ``@tool(...)`` or ``tool(fn, ...)``.

    Parameters
    ----------
    fn : callable, optional
        The function; omitted when used as ``@tool(...)``.
    name : str | None, optional
        The tool's name; the function's by default.
    description : str | None, optional
        What a model reads; the docstring's first line by default.
    risk : str, optional
        The risk class a permission policy reads; ``"write"`` by default,
        because the library cannot tell what a function does.
    """
    if fn is None:
        return lambda f: FunctionTool(f, name=name, description=description, risk=risk)
    return FunctionTool(fn, name=name, description=description, risk=risk)
