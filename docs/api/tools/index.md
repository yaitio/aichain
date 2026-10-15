# API: Tools — the base

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../../tools/index.md).

<!-- g:api -->
[`tool`](#tool) · [`Tool`](#tool) · [`FunctionTool`](#functiontool) · [`ToolResult`](#toolresult) · [`Record`](#record) · [`media_part`](#media_part)

---

## `tool()`

```python
tool(
    fn=None,
    *,
    name: str | None = None,
    description: str | None = None,
    risk: str = 'write',
)
```

Make any function a tool — code, a chain and an agent call it alike.

```python
@tool(risk="read")
def word_count(text: str) -> int:
    """Count the words in a text.

    text: the text to count.
    """
    return len(text.split())
```

Usable as `@tool`, `@tool(...)` or `tool(fn, ...)`. A keyword-only
`run` (`def f(text, *, run: Run)`) is handed the run the call belongs
to and left out of the schema a model sees.

**Parameters**

- `fn` (*callable, optional*, default `None`) — The function; omitted when used as `@tool(...)`.
- `name` (*str | None, optional*, default `None`) — The tool's name; the function's by default.
- `description` (*str | None, optional*, default `None`) — What a model reads; the docstring's first line by default.
- `risk` (*str, optional*, default `'write'`) — The risk class a permission policy reads; `"write"` by default —
  the library cannot tell what a function does.

---

## `Tool`

```python
Tool()
```

Base class for all tool implementations: set the three class attributes
and implement `run`; everything else is inherited.

Class attributes
----------------
name : str
    Unique identifier; the function name a model sees.
description : str
    What the tool does; what a model reads to decide whether to use it.
parameters : dict
    JSON Schema `object` of the inputs: `"properties"` and optionally
    `"required"`.

**Examples**

Defining a custom tool:

```python
from yait_aichain.tools import Tool, ToolResult

class ReverseTextTool(Tool):
    name        = "reverse_text"
    description = "Reverse the characters in a string."
    parameters  = {"type": "object", "required": ["input"],
                   "properties": {"input": {"type": "string"}}}

    def run(self, input: str, options=None) -> str:
        return input[::-1]

tool   = ReverseTextTool()
result = tool(input="hello")    # ToolResult(success=True, output='olleh')
raw    = tool.run(input="hi")   # 'ih'  (raises on error)
```

### `Tool.run()`

```python
Tool.run(input=None, options=None) -> Any
```

Execute the tool and return the raw result; may raise — `__call__` does not.

**Parameters**

- `input` (*Any*, default `None`) — The primary input, the `"input"` key of `parameters`.
- `options` (*dict | None*, default `None`) — Optional configuration; keys depend on the tool.

**Returns**

- `Any` — Tool-specific output (string, dict, list, bytes, …).

**Raises**

- `NotImplementedError` — Always, from the base class.

### `Tool()`

```python
Tool(input=None, options=None, **kwargs) -> ToolResult
```

Validate inputs, run the tool, and return a `ToolResult`.

**Never raises**: any exception, validation included, comes back as
`ToolResult(success=False, error=...)` — a `Suspend` too, since only
an agent or a chain can wait. A tool with its own parameters is called
by name: `tool(email="a@b.c")`.

**Parameters**

- `input` (*Any*, default `None`) — Primary input for the tool.
- `options` (*dict | None*, default `None`) — Optional configuration dict.
- `**kwargs` — The tool's own parameters, by name.

**Returns**

- `ToolResult` — The output on success, the error message on failure.

### `Tool.schema()`

```python
Tool.schema() -> dict
```

Return an OpenAI-compatible function-calling schema for this tool.

**Returns**

- `dict` — `{"type": "function", "function": {"name", "description", "parameters"}}`
  — what a `tools=[...]` field of a chat API takes.

### `Tool.unknown_keys()`

```python
Tool.unknown_keys(kwargs: dict) -> list
```

Keys the schema does not declare, as `"options.recencyy"` paths.

Checked one level into a declared object; an object with no
`properties`, or an open map, accepts any key.

**Parameters**

- `kwargs` (*dict*) — The arguments of a call.

### `Tool.check_args()`

```python
Tool.check_args(kwargs: dict) -> str | None
```

Validate keyword arguments against `parameters`: a model-readable
message when a required argument is missing or a key is unknown,
`None` when the call looks well-formed. Types are not checked.

**Parameters**

- `kwargs` (*dict*) — The arguments of a call.

---

## `FunctionTool`

```python
FunctionTool(
    fn,
    *,
    name: str | None = None,
    description: str | None = None,
    risk: str = 'write',
)
```

A function as a `Tool` — what `tool` returns.

The function's name is the tool's `name`, the first line of its
docstring the `description`, and its parameters the schema: types from
the annotations, required where there is no default, each described by
its `name: text` line in the docstring. `run(**arguments)` calls the
function and returns what it returns; the function itself is
`.__wrapped__`.

**Parameters**

- `fn` (*callable*) — The function to wrap.
- `name` (*str | None, optional*, default `None`) — The tool's name; the function's by default.
- `description` (*str | None, optional*, default `None`) — What a model reads; the docstring's first line by default.
- `risk` (*str, optional*, default `'write'`) — The risk class; `"write"` by default.

### `FunctionTool.run()`

```python
FunctionTool.run(*args, **kwargs)
```

Execute the tool and return the raw result; may raise — `__call__` does not.

**Parameters**

- `input` (*Any*) — The primary input, the `"input"` key of `parameters`.
- `options` (*dict | None*) — Optional configuration; keys depend on the tool.

**Returns**

- `Any` — Tool-specific output (string, dict, list, bytes, …).

**Raises**

- `NotImplementedError` — Always, from the base class.

---

## `ToolResult`

```python
ToolResult(success: bool, output: Any, error: str | None = None)
```

Uniform return type for every `Tool` call.

**Attributes**

- `success` (*bool*) — `True` when `run()` completed without raising an exception.
- `output` (*Any*) — The value returned by `run()`.  `None` when *success* is
  `False`.
- `error` (*str | None*, default `None`) — Human-readable error message.  `None` when *success* is `True`.

**Examples**

```python
result = tool(input="report.pdf")

if result:                       # bool(result) == result.success
    print(result.output)
else:
    print("Error:", result.error)
```

---

## `Record`

```python
Record(data: dict, render)
```

A tool's result: a dict for code, and text for a model or a prompt.

Three readers take the same result. Code reads the fields —
`found["results"]`. An agent shows the model `str(record)`, the tool's
own rendering, rather than the dict as JSON. A Skill that substitutes the
result into its prompt gets that rendering too, since substitution is
`str(value)`. And a Chain stores a record under its step's key, where a
plain dict is merged into the variables: a record is one result, and its
keys (`results`, `cost`) would collide between two steps.

The rendering is not stored: a record written to a state store and read
back is a plain dict.

**Parameters**

- `data` (*dict*) — The fields, for code.
- `render` (*callable*) — Turns them into the text a model is shown; `str(record)` calls it.

---

## `media_part()`

```python
media_part(path: str, mime: str | None = None) -> dict
```

A file as a media part a tool can return: read once, now, as base64.

The type follows the MIME type — `image`, `audio`, `video`, else
`document`. Self-contained, so the history it enters means the same
thing in another process and after the file changes.

**Parameters**

- `path` (*str*) — The file.
- `mime` (*str | None, optional*, default `None`) — Its MIME type; guessed from the name and the first bytes when omitted.
<!-- /g:api -->
