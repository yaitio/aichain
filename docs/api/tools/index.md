# API: Tools — the base

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../../tools/index.md).

<!-- g:api -->
[`tool`](#tool) · [`Tool`](#tool) · [`FunctionTool`](#functiontool) · [`ToolResult`](#toolresult) · [`Record`](#record)

---

## `tool()`

```python
tool(
    fn=None,
    *,
    name: 'str | None' = None,
    description: 'str | None' = None,
    risk: str = 'write',
)
```

Make any function a tool — code, a chain and an agent call it alike.

```python
@tool(risk="read")
def word_count(text: str, unique: bool = False) -> dict:
    """Count the words in a text.

    text: the text to count.
    unique: count each word once.
    """
    words = text.split()
    return {"words": len(set(words)) if unique else len(words)}

word_count.run(text="to be or not to be")     # {"words": 6}
Agent(model, tools=[word_count])              # the model sees word_count(text, unique)
```

`name`, `description` and `risk` are the tool's settings; the
function's parameters are its arguments. **`risk` defaults to
`"write"`**, as on every tool: the library cannot tell what a function
does, and assuming it only reads would let a permission policy pass it
unasked — saying `risk="read"` is the author's claim to make. Usable as
`@tool`, `@tool(...)` or `tool(fn, ...)`.

**Parameters**

- `fn` (*callable, optional*, default `None`) — The function; omitted when used as `@tool(...)`.
- `name` (*str | None, optional*, default `None`) — The tool's name; the function's by default.
- `description` (*str | None, optional*, default `None`) — What a model reads; the docstring's first line by default.
- `risk` (*str, optional*, default `'write'`) — The risk class a permission policy reads; `"write"` by default,
  because the library cannot tell what a function does.

---

## `Tool`

```python
Tool()
```

Base class for all tool implementations.

Subclasses must set the three class-level attributes and implement
`run`.  Everything else is inherited.

Class attributes
----------------
name : str
    Unique, machine-readable identifier, e.g. `"markitdown"`.
    Used as the function name in OpenAI / Anthropic schemas.

description : str
    One-sentence description of what the tool does.  This is what an
    LLM reads to decide whether to use the tool.

parameters : dict
    JSON Schema `object` describing the tool's accepted inputs.
    Must contain a `"properties"` key and optionally a `"required"`
    list.  See module docstring for an example.

**Examples**

Defining a custom tool:

```python
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
```

Returning a dict inside a Chain
-------------------------------
A step's output key names a **string** result. When `run` returns a
dict, the keys are spread into the run's variables instead and the step
name is not used:

```python
Chain(steps=[(CollectFacts(), "facts")])   # returns {"brand", "decks"}
# accumulated: {"brand": ..., "decks": ...}   — no "facts" key
```

Spreading is deliberate: it is what lets one step feed several named
inputs downstream. What it costs is that two steps returning the same key
overwrite each other silently, and a later step asking for `facts` gets
a missing-argument error rather than a dict.

Wrap the payload when the step name is what the next step expects:

```python
def run(self, ...) -> dict:
    return {"facts": {"brand": ..., "decks": ...}}
```

### `Tool.run()`

```python
Tool.run(input=None, options=None) -> Any
```

Execute the tool and return the raw result.

Must be implemented by every subclass.  May raise any exception —
callers that want safe execution should use `__call__` instead.

**Parameters**

- `input` (*Any*, default `None`) — The primary input for the tool, matching the `"input"` key
  declared in `self.parameters["properties"]`.
- `options` (*dict | None*, default `None`) — Optional configuration dict.  Keys depend on the concrete tool.

**Returns**

- `Any` — Tool-specific output (string, dict, list, bytes, …).

**Raises**

- `NotImplementedError` — Always, from the base class.

### `Tool()`

```python
Tool(input=None, options=None, **kwargs) -> ToolResult
```

Validate inputs, run the tool, and return a `ToolResult`.

Unlike `run`, this method **never raises**.  Any exception
(including `ValueError` from validation) is caught and returned
as `ToolResult(success=False, error=...)`.

A tool with its own parameters — `def run(self, email, options=None)`
— is called by name, `tool(email="a@b.c")`, the way an Agent or a
Chain calls `run(**kwargs)`. Until this, only `input=` and
`options=` were accepted, so such a tool could not be called safely
at all: `tool(email=...)` raised a bare `TypeError`. `MCPTool`
already forwarded keywords the same way.

**Parameters**

- `input` (*Any*, default `None`) — Primary input for the tool.
- `options` (*dict | None*, default `None`) — Optional configuration dict.
- `**kwargs` — The tool's own parameters, by name.

**Returns**

- `ToolResult` — `success=True`  + raw output on success.
  `success=False` + error message on failure.

### `Tool.schema()`

```python
Tool.schema() -> dict
```

Return an OpenAI-compatible function-calling schema for this tool.

The returned dict can be passed directly to the `tools` parameter
of the OpenAI Chat Completions API or the Anthropic Messages API.

**Returns**

- `dict` — `{"type": "function", "function": {"name": ..., "description":
  ..., "parameters": ...}}`.

**Examples**

```python
import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Convert report.pdf"}],
    tools=[tool.schema()],
)
```

### `Tool.unknown_keys()`

```python
Tool.unknown_keys(kwargs: dict) -> list
```

Keys the schema does not declare, as `"options.recencyy"` paths.

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

**Parameters**

- `kwargs` (*dict*) — The arguments of a call.

### `Tool.check_args()`

```python
Tool.check_args(kwargs: dict) -> 'str | None'
```

Validate keyword arguments against `parameters` (1.4.4 tool-call
repair). Returns a model-readable remediation message when a required
argument is missing, or `None` when the call looks well-formed.

Lightweight on purpose — required-presence only, no full JSON-Schema
type enforcement (which would need an extra dependency). An Agent feeds
the returned message back to the model as a structured observation so it
can correct the call within the step's attempt budget.

**Parameters**

- `kwargs` (*dict*) — The arguments of a call.

---

## `FunctionTool`

```python
FunctionTool(
    fn,
    *,
    name: 'str | None' = None,
    description: 'str | None' = None,
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

Execute the tool and return the raw result.

Must be implemented by every subclass.  May raise any exception —
callers that want safe execution should use `__call__` instead.

**Parameters**

- `input` (*Any*) — The primary input for the tool, matching the `"input"` key
  declared in `self.parameters["properties"]`.
- `options` (*dict | None*) — Optional configuration dict.  Keys depend on the concrete tool.

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
<!-- /g:api -->
