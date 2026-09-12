<!-- g:tool -->
# `MCPTool`

A single tool from an MCP server, exposed as a standard aichain Tool.

| | |
|---|---|
| Import | `from yait_aichain.tools import MCPTool` |
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
)
```

The call schema is built from the constructor arguments, so it is whatever the caller declares.
<!-- /g:tool -->
