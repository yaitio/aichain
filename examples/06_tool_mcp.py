"""
06_tool_mcp.py — Connect to an MCP server, discover tools, run them.

Start the MCP server first:
    python your_newsapi_server.py

MCP server: newsapi on http://127.0.0.1:8009/mcp

Tools available:
    search_news        — search articles (requires: q, sources, or domains)
    get_top_headlines  — breaking news  (requires: q, country, category, or sources)
    get_sources        — list all available news sources
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tools import MCPTools

MCP_URL = "http://127.0.0.1:8009/mcp"

# ── Discover all tools on the server ─────────────────────────────────────────
tools = MCPTools(MCP_URL)
print(f"Connected: {MCP_URL}")
print(f"Tools found: {len(tools)}\n")
for t in tools:
    print(f"  {t.name:<25} {t.description.splitlines()[0]}")

# ── 1. Search news ────────────────────────────────────────────────────────────
print("\n── search_news: 'AI agents' ─────────────────────────────────────────")
tool_map = {t.name: t for t in tools}

search = tool_map["search_news"]
result = search.run(input={"q": "AI agents"})
print(result)

# ── 2. Top headlines ──────────────────────────────────────────────────────────
print("\n── get_top_headlines: technology ────────────────────────────────────")
headlines = tool_map["get_top_headlines"]
result = headlines.run(input={"category": "technology"})
print(result)

# ── 3. List sources ───────────────────────────────────────────────────────────
print("\n── get_sources ──────────────────────────────────────────────────────")
sources = tool_map["get_sources"]
result  = sources.run(input={})
print(result)
