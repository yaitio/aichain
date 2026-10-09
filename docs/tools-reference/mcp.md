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
    call_timeout: float | None = None,
)
```

The call schema is built from the constructor arguments, so it is whatever the caller declares.
<!-- /g:tool -->

## `MCP` — a server as a utility

```python
from yait_aichain.utilities import MCP

deepwiki = MCP("https://mcp.deepwiki.com/mcp")
deepwiki.names                                       # ['ask_wiki_question', 'read_wiki_contents', ...]
deepwiki.tool("ask_wiki_question").run(repoName="yaitio/aichain", question="What is a Record?")
deepwiki.call("read_wiki_structure", repoName="yaitio/aichain")    # the same, from code
Agent(model, tools=deepwiki.tools)
```

The server is asked for its tools on first use, not when `MCP(...)` is made.
Each tool carries the name, description and schema the server gives it.
`MCPTools(server)` and `MCPTool` still work.
