# Utilities

Some things your code works with are bigger than one call: a vector database,
a folder, an API with twenty endpoints, an MCP server. A utility is that thing
as one object. Your code calls its methods; some of its members are tools, and
those — only those — are what an agent gets.

```python
from yait_aichain.utilities import Vectors
from yait_aichain.tools import chunk, embed

kb = Vectors("qdrant", collection="docs", description="the product documentation",
             embedder=embed("openai", model="text-embedding-3-small"))
pieces = chunk(max_chars=800).run(text)                # [{"text": …}, …]
kb.create(dimension=1536)                              # methods: your code
kb.upsert([{"id": f"doc-{i}", **p} for i, p in enumerate(pieces)])   # each record needs an id
kb.query.run("how does caching work?", {"n": 5})       # a tool: .run, like any other

Agent(model, tools=[kb.query])                         # the agent searches, and can do nothing else
```

---

## A method or a tool

The syntax says which is which. Only a tool has `.run`:

| | Called as | Your code | A chain | An agent |
|---|---|---|---|---|
| A utility's method — `kb.upsert` | `kb.upsert(records)` | ✓ | ✓, as a step | ✗, refused by name |
| A utility's tool — `kb.query` | `kb.query.run(input, options)` | ✓ | ✓ | ✓ |

Administration stays with code. An agent given `kb.upsert` is refused when it
is made — "`Vectors.upsert` is a method of a utility, not a tool" — so a
model can never write to, or delete from, a store it was only meant to search.

A utility's `.tools` lists all its tools at once. Giving them one by one is the
way to give fewer: `[sh.browse, sh.read]` is an agent that reads a folder and
cannot write to it or run anything in it.

---

## The four

### Vectors — a vector database

```python
kb = Vectors("chroma", collection="tickets", description="last year's support tickets",
             embedder=embed("openai", model="text-embedding-3-small"))
```

| Methods | Tool |
|---|---|
| `create(dimension)`, `upsert(records)`, `fetch(ids)`, `delete(ids)`, `count()` | `query` — `input`, and `n`, `filter` per call |

Backends: `chroma`, `pinecone`, `qdrant`. `description` is required — it is how
the search tool tells a model what it searches, and an agent with two
knowledge bases cannot tell them apart without it. → [API: Vectors](../api/utilities/vectors.md)

### Shell — a folder

```python
sh = Shell("projects/demo")
sh.read.run("README.md", {"start_line": 1, "end_line": 40})
Agent(model, tools=[sh.browse, sh.read])               # read-only
```

| Tool | Risk | What it does |
|---|---|---|
| `browse`, `read` | `read` | look inside the folder |
| `write` | `write` | change a file inside the folder |
| `run` | `privileged` | run any command — the folder is where it **starts**, not a boundary |

`run` has the host process's rights: a command can use absolute paths, reach
the network, delete. With a [permission policy](../control/permissions.md) it
goes to a person before every command; without one, an agent runs whatever it
writes. Give it only behind a policy, or inside a container.

→ [API: Shell](../api/utilities/shell.md)

### REST — an API

```python
api = REST("https://api.example.com", description="The shop's order API.",
           auth={"type": "bearer", "env": "SHOP_TOKEN"})
get_order = api.endpoint("get_order", "GET", "/orders/{id}", path_params=["id"],
                         description="Return one order by its id.")
get_order.run(id=42)
```

The base URL, the auth and the headers are set once, for every endpoint, and
never passed with a call. Each endpoint is a tool, and the API's description
and the endpoint's are both required — a model has nothing else to go on.
→ [API: REST](../api/utilities/rest.md)

### MCP — a server

```python
deepwiki = MCP("https://mcp.deepwiki.com/mcp")
deepwiki.names                                         # the tools the server declares
deepwiki.call("read_wiki_structure", repoName="yaitio/aichain")   # from code
Agent(model, tools=deepwiki.tools)                     # or deepwiki.tool("ask_wiki_question")
```

The server is asked for its tools on first use, not when the utility is made.
Each tool keeps the name, description and schema the server gives it.
`connect_timeout` bounds reaching the server, `call_timeout` one call.

Each tool's risk class comes from the server's annotations — `readOnlyHint`
makes it `read`, `destructiveHint` makes it `destructive` — and is
`privileged` where the server says nothing, so a policy sends it to a person.
The annotations are the server's word; `risk={"ask_wiki_question": "read"}`
puts yours over it.
→ [API: MCP](../api/utilities/mcp.md)

---

## In a chain

A method is a step like any function: its parameters are filled from the
chain's variables by name.

```python
def numbered(pieces):
    return [{"id": f"doc-{i}", **p} for i, p in enumerate(pieces)]

chain = Chain(steps=[
    (chunk(max_chars=800), "pieces",  {"input": "text"}),
    (numbered,             "records"),                 # a function: reads `pieces`
    (kb.upsert,            "stored"),                  # a method: reads `records`, writes how many
])
```

A chain with a method in it cannot be saved to YAML — a method of a live
object is not data — and `Chain.save()` says so.
