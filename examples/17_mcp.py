"""
17_mcp.py — An MCP server is a set of agent tools: discover them, call one.

Keys: none; DeepWiki is a public, read-only server. Needs pip install "yait-aichain[mcp]".
"""

from yait_aichain.utilities import MCP

deepwiki = MCP("https://mcp.deepwiki.com/mcp")

print("Tools found:")
for t in deepwiki.tools:
    print(f"  {t.name:<22} {t.risk:<10} {t.description.splitlines()[0][:60]}")

structure = deepwiki.call("read_wiki_structure", repoName="pallets/flask")
print(f"\nread_wiki_structure, pallets/flask:\n{str(structure)[:500]}")

# An agent takes them as they are: Agent(model, tools=deepwiki.tools)
