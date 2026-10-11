# API: mcp

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../../utilities/index.md).

<!-- g:tool -->
## `MCP`

```python
MCP(
    server,
    *,
    filter: 'list | None' = None,
    headers: 'dict | None' = None,
    env: 'dict | None' = None,
    cwd: 'str | None' = None,
    transport: 'str | None' = None,
    connect_timeout: 'float | None' = 60.0,
    call_timeout: 'float | None' = None,
    risk: 'dict | None' = None,
)
```

An MCP server. `tools` and `tool(name)` give its tools; `call(name,
**arguments)` runs one from code. The server is asked for its tools on
first use, not here, so making the utility costs nothing.

### `MCP.tools`

The utility's tools, to hand an agent at once: `Agent(tools=kb.tools)`.

### `MCP.names`

The names of the server's tools.

### `MCP.TOOLS`

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### `MCP.tool()`

```python
MCP.tool(name: str) -> MCPTool
```

The server's tool *name*, or a KeyError naming the ones it has.

### `MCP.call()`

```python
MCP.call(name: str, **arguments)
```

Run the server's tool *name* with *arguments*; its result, or the error it raised.

---

## The classes behind it

What the factory returns, with the schema a model is shown.

### `MCPTool`

A single tool from an MCP server, exposed as a standard aichain Tool.

| | |
|---|---|
| Import | `from yait_aichain.utilities.mcp import MCPTool` |
| Risk class | `write` |

```python
MCPTool(
    name: str,
    description: str,
    server: str | dict | Any,
    *,
    parameters: dict | None = None,
    headers: dict | None = None,
    env: dict | None = None,
    cwd: str | None = None,
    transport: str | None = None,
    call_timeout: float | None = None,
    risk: str = 'privileged',
)
```

The call schema is built from the constructor arguments, so it is whatever the caller declares.
<!-- /g:tool -->
