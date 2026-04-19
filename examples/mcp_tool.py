"""
examples/mcp_tool.py
=====================

MCPTool and MCPTools — runnable examples
=========================================

Demonstrates how to connect any MCP server to aichain using
``MCPTool`` and ``MCPTools``.

  1. **HTTP server** — connect to a public MCP endpoint, list tools,
     call one directly.
  2. **STDIO server** — launch a local MCP subprocess (requires
     ``npx @modelcontextprotocol/server-memory`` to be installed).
  3. **Agent integration** — register all MCP tools from a server into
     an Agent in three lines.
  4. **Chain integration** — use one MCP tool as a Chain step.
  5. **MCPTool direct** — build a single MCPTool without discovery.

Requirements
------------
  pip install fastmcp

Run::

    python examples/mcp_tool.py                   # all examples
    python examples/mcp_tool.py --scenario http
    python examples/mcp_tool.py --scenario stdio
    python examples/mcp_tool.py --scenario agent
    python examples/mcp_tool.py --scenario chain
    python examples/mcp_tool.py --scenario direct
"""

from __future__ import annotations

import os
import sys

# ── Resolve library root ──────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _LIB_DIR)

try:
    import dotenv
    dotenv.load_dotenv(os.path.join(_LIB_DIR, ".env"))
except ImportError:
    pass

from tools import MCPTool, MCPTools


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}\n")


def _check_fastmcp() -> bool:
    try:
        import fastmcp  # noqa: F401
        return True
    except ImportError:
        print("  ⚠  fastmcp is not installed.  Run: pip install fastmcp")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 1. HTTP server — discover and call
# ─────────────────────────────────────────────────────────────────────────────

def demo_http() -> None:
    """
    Connect to a public Streamable-HTTP MCP endpoint, list its tools,
    then call one.

    This demo uses the public MCP "everything" test server hosted at
    https://demo.mcp.run/sse — replace the URL with any MCP HTTP endpoint.
    """
    _section("1. HTTP server — discover tools and call one")

    if not _check_fastmcp():
        return

    # Any public or private MCP HTTP endpoint works here.
    # We use a well-known test server (SSE transport):
    SERVER_URL = "https://demo.mcp.run/sse"

    print(f"  Server: {SERVER_URL}\n")

    try:
        tools = MCPTools(SERVER_URL)
    except Exception as exc:
        print(f"  Could not connect: {exc}")
        print("  (This is expected when running offline or if the public demo is down.)")
        return

    print(f"  Found {len(tools)} tool(s):")
    for t in tools:
        print(f"    • {t.name}: {t.description or '(no description)'}")

    if not tools:
        print("  No tools found — nothing to call.")
        return

    # Call the first available tool with no arguments as a smoke-test
    first = tools[0]
    print(f"\n  Calling {first.name!r} with no arguments:")
    try:
        result = first.run()
        print(f"  → {result!r}")
    except Exception as exc:
        print(f"  Call raised (expected if required args missing): {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. STDIO server — subprocess
# ─────────────────────────────────────────────────────────────────────────────

def demo_stdio() -> None:
    """
    Launch a local MCP server as a subprocess via STDIO.

    Requires:  npm install -g @modelcontextprotocol/server-memory
    """
    _section("2. STDIO server — subprocess (MCP memory server)")

    if not _check_fastmcp():
        return

    import shutil
    if not shutil.which("npx"):
        print("  ⚠  npx not found — skipping STDIO demo.")
        print("     Install Node.js to run local MCP servers via npx.")
        return

    server_spec = {
        "command": "npx",
        "args":    ["-y", "@modelcontextprotocol/server-memory"],
    }
    print(f"  Server spec: {server_spec}\n")

    try:
        tools = MCPTools(server_spec)
    except Exception as exc:
        print(f"  Could not start server: {exc}")
        return

    print(f"  Found {len(tools)} tool(s):")
    for t in tools:
        print(f"    • {t.name}: {t.description or '(no description)'}")

    # Demo: store and retrieve a value
    store_tool    = next((t for t in tools if "store" in t.name.lower()  or "set" in t.name.lower()  or "create" in t.name.lower()),  None)
    retrieve_tool = next((t for t in tools if "retrieve" in t.name.lower() or "get" in t.name.lower() or "read" in t.name.lower()),  None)

    if store_tool and retrieve_tool:
        print(f"\n  Storing a test value via {store_tool.name!r}…")
        try:
            store_tool.run(input={"key": "greeting", "value": "Hello from aichain!"})
            print(f"  Retrieving via {retrieve_tool.name!r}…")
            result = retrieve_tool.run(input={"key": "greeting"})
            print(f"  → {result!r}")
        except Exception as exc:
            print(f"  Tool call raised: {exc}")
    else:
        print("\n  (Could not identify store/retrieve tools for demo.)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Agent integration
# ─────────────────────────────────────────────────────────────────────────────

def demo_agent() -> None:
    """
    Register all MCP tools from a server in an Agent — three lines.
    """
    _section("3. Agent integration")

    if not _check_fastmcp():
        return

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("  ⚠  ANTHROPIC_API_KEY not set — showing code pattern only.")
        print()
        print("  # Pattern:")
        print("  from agent import Agent")
        print("  from models import Model")
        print("  from tools import MCPTools")
        print()
        print('  agent = Agent(')
        print('      model = Model("claude-sonnet-4-6"),')
        print('      tools = MCPTools("https://your-mcp-server.com/mcp"),')
        print('  )')
        print('  result = agent.run("What can you search for?")')
        return

    SERVER_URL = "https://demo.mcp.run/sse"
    print(f"  Server: {SERVER_URL}")

    try:
        mcp_tools = MCPTools(SERVER_URL)
    except Exception as exc:
        print(f"  Could not connect: {exc}")
        return

    if not mcp_tools:
        print("  No tools found on demo server.")
        return

    from agent import Agent
    from models import Model

    print(f"  Registering {len(mcp_tools)} MCP tool(s) in an Agent …\n")

    agent = Agent(
        model = Model("claude-sonnet-4-6"),
        tools = mcp_tools,
    )

    result = agent.run("List the tools you have available and briefly describe each.")
    print(f"  Agent response:\n  {result.output}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Chain integration
# ─────────────────────────────────────────────────────────────────────────────

def demo_chain() -> None:
    """
    Use an MCP tool as one step in a Chain.

    Step 1: MCPTool "fetch_page" retrieves a URL
    Step 2: Skill (claude-haiku) summarises the content
    """
    _section("4. Chain integration  (MCP fetch + Skill summarise)")

    if not _check_fastmcp():
        return

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("  ⚠  ANTHROPIC_API_KEY not set — showing code pattern only.")
        print()
        print("  # Pattern:")
        print("  from chain import Chain")
        print("  from skill import Skill")
        print("  from models import Model")
        print("  from tools import MCPTools")
        print()
        print('  mcp_tools = MCPTools("https://your-mcp-server.com/mcp")')
        print('  fetch     = next(t for t in mcp_tools if t.name == "fetch")')
        print()
        print('  summarise = Skill(')
        print('      model         = Model("claude-haiku-4-5-20251001"),')
        print('      system_prompt = "Summarise the following in 3 bullet points.",')
        print('      output_format = "text",')
        print('  )')
        print()
        print('  chain = Chain(steps=[')
        print('      (fetch,     "page_content",  {"url": "page_url"}),')
        print('      (summarise, "summary",        {"input": "page_content"}),')
        print('  ])')
        print('  chain.run(variables={"page_url": "https://example.com"})')
        return

    # Use the public playwright/fetch MCP server if available
    SERVER_URL = "https://demo.mcp.run/sse"

    try:
        mcp_tools = MCPTools(SERVER_URL)
    except Exception as exc:
        print(f"  Could not connect: {exc}")
        return

    fetch_tool = next((t for t in mcp_tools if "fetch" in t.name.lower()), None)
    if not fetch_tool:
        print(f"  No 'fetch' tool found on {SERVER_URL}.")
        print(f"  Available: {[t.name for t in mcp_tools]}")
        return

    from chain import Chain
    from skill import Skill
    from models import Model

    summarise = Skill(
        model         = Model("claude-haiku-4-5-20251001"),
        system_prompt = "Summarise the following content in 3 concise bullet points.",
        output_format = "text",
    )

    chain = Chain(
        steps = [
            (fetch_tool, "page_content", {"url": "target_url"}),
            (summarise,  "summary",      {"input": "page_content"}),
        ],
        name = "mcp_fetch_summarise",
    )

    print(f"  Fetching https://example.com via MCP {fetch_tool.name!r} → summarising …\n")
    chain.run(variables={"target_url": "https://example.com"})
    print(f"  Summary:\n  {chain.accumulated.get('summary', '(not found)')}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MCPTool direct (no discovery)
# ─────────────────────────────────────────────────────────────────────────────

def demo_direct() -> None:
    """
    Build a single MCPTool without the discovery round-trip when you
    already know the tool name.
    """
    _section("5. MCPTool direct — skip discovery")

    if not _check_fastmcp():
        return

    print("  Building MCPTool directly (no MCPTools discovery call):\n")

    # Build without connecting — no API call yet
    tool = MCPTool(
        name        = "echo",
        description = "Echo the input back.",
        server      = "https://demo.mcp.run/sse",
    )

    print(f"  {tool!r}")
    print(f"  parameters: {tool.parameters}")
    print()

    print("  Calling tool.run(input={'message': 'hello'}) …")
    try:
        result = tool.run(input={"message": "hello"})
        print(f"  → {result!r}")
    except Exception as exc:
        print(f"  Call raised (tool may not exist on demo server): {exc}")

    print()
    print("  This pattern is useful when:")
    print("  • You know the server's tool names in advance.")
    print("  • You want to avoid the extra round-trip to list all tools.")
    print("  • You're building a Chain and only need one specific tool.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

_SCENARIOS: dict = {
    "http":   demo_http,
    "stdio":  demo_stdio,
    "agent":  demo_agent,
    "chain":  demo_chain,
    "direct": demo_direct,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MCPTool and MCPTools examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/mcp_tool.py                   # all\n"
            "  python examples/mcp_tool.py --scenario http\n"
            "  python examples/mcp_tool.py --scenario stdio\n"
            "  python examples/mcp_tool.py --scenario agent\n"
            "  python examples/mcp_tool.py --scenario chain\n"
            "  python examples/mcp_tool.py --scenario direct\n"
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=list(_SCENARIOS),
        help="Which scenario to run (default: all).",
    )
    args = parser.parse_args()

    print("\n══ MCPTool / MCPTools examples ════════════════════════════════════\n")
    print("  Requires:  pip install fastmcp")
    print()

    if args.scenario:
        _SCENARIOS[args.scenario]()
    else:
        for fn in _SCENARIOS.values():
            fn()

    print("\n══ Done ════════════════════════════════════════════════════════════\n")
