"""
06_tool_mcp.py — An MCP server as a utility: discover its tools, call them.

Uses DeepWiki (Cognition), a public MCP server that answers questions about
GitHub repositories — no key, read-only. The server declares its tools; each
comes with the name, description and schema the server gives it.

Required env vars:
    (none)

Required packages:
    pip install "yait-aichain[mcp]"
"""

from yait_aichain.utilities import MCP

deepwiki = MCP("https://mcp.deepwiki.com/mcp")

# ── Discover the tools the server declares ───────────────────────────────────
print("Tools found:")
for t in deepwiki.tools:
    print(f"  {t.name:<22} {t.description.splitlines()[0]}")

# ── 1. One tool, called by its arguments ─────────────────────────────────────
print("\n── read_wiki_structure: anthropics/anthropic-sdk-python ──────────")
structure = deepwiki.tool("read_wiki_structure").run(repoName="anthropics/anthropic-sdk-python")
print(str(structure)[:600])

# ── 2. Another, from code, by name ───────────────────────────────────────────
print("\n── read_wiki_contents: pallets/flask ──────────────────────────────")
contents = deepwiki.call("read_wiki_contents", repoName="pallets/flask")
print(f"{len(str(contents)):,} characters; the first 400:")
print(str(contents)[:400])

# ask_wiki_question answers in prose, written by DeepWiki's own model — slower,
# and the stream can be cut by a proxy on the way:
#     deepwiki.call("ask_wiki_question", repoName="pallets/flask", question="How are blueprints registered?")

# An agent takes the server's tools, never the utility itself:
#     Agent(model, tools=deepwiki.tools)
