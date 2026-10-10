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
