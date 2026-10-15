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
knowledge bases cannot tell them apart without it.

| Backend | Where it is | Key | Also |
|---|---|---|---|
| `chroma` (or `chromadb`) | `url=` or `CHROMA_URL`, default `http://localhost:8000` | `CHROMA_API_KEY`, for Chroma Cloud | `CHROMA_TENANT`, `CHROMA_DATABASE` |
| `pinecone` | `index=` or `PINECONE_INDEX`; the collection is a namespace | `PINECONE_API_KEY`, required | `cloud`, `region` for a new index (`aws`, `us-east-1`) |
| `qdrant` | `url=` or `QDRANT_URL`, default `http://localhost:6333` | `QDRANT_API_KEY`, for Qdrant Cloud | |

Each is spoken over its REST API, with no vendor SDK installed. A record's
text is kept in the backend's metadata, so `fetch` and `query` give it back.
A `filter` is the backend's own filter format, passed through as it is. → [API: Vectors](../api/utilities/vectors.md)

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

The folder defaults to the working directory when the Shell is made. For
`browse`, `read` and `write` it is a boundary: a path is resolved against it,
symlinks and `..` followed, and one that lands outside raises
`PermissionError`.

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
An endpoint's risk class follows its method: `GET` is `read`, `DELETE` is
`destructive`, `POST`, `PUT` and `PATCH` are `external` — they change
something on a service the library does not own. `risk="read"` on a `POST`
that only searches says otherwise.

An endpoint declares where each field goes — `path_params` into `{field}` in
the path, `query_params` after `?`, `body_params` into the JSON body — and a
field the call does not give is left out. `run()` takes the values as a dict
or as keyword arguments, which is how a chain passes them; a field given both
ways is an error. A JSON response comes back as a `dict` or `list` (in a chain
a dict's keys become variables), anything else as a `str`, and
`response_field` picks one key out of a dict. A failed call raises the typed
`APIError` a model call would.

| `auth["type"]` | Sends | The secret from |
|---|---|---|
| `bearer` | `Authorization: Bearer <value>` | one of `token` (a value or a callable), `token_field` (a field your code or a chain fills, never the model — wrap it in `state.Secret` so a paused chain does not store it) or `env` |
| `cookie` | `Cookie: <cookie_name>=<value>`, `cookie_name` defaulting to `token` | as for `bearer` |
| `api_key` | the value in `header`, default `X-Api-Key` | as for `bearer` |
| `basic` | `Authorization: Basic …` | `username_env` and `password_env` |
| `query` | `?<param>=<value>` in the URL, `param` defaulting to `key` | as for `bearer` |
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
`connect_timeout` bounds reaching the server and listing its tools (60 s),
`call_timeout` one call (300 s); `None` waits indefinitely. `filter=[...]`
keeps only the tools named, and `prefix="wiki_"` names them `wiki_…` for an
agent holding two servers whose tools share names — `tool()`, `call()` and
`risk=` still take the server's own names.

| `server` | Transport |
|---|---|
| `"./server.py"`, `"python server.py …"`, `{"command": "npx", "args": [...]}` | a STDIO subprocess (`env=`, `cwd=`) |
| `"https://example.com/mcp"`, `{"url": "https://…"}` | Streamable HTTP (`headers=` for auth) |
| a URL ending in `/sse` | the legacy SSE transport |

It needs `pip install fastmcp`. The calls are synchronous: the library runs
one background event loop for every MCP tool, so a tool works from Jupyter or
inside a web server's loop as anywhere else, and a STDIO server is not
restarted for each call.

Each tool's risk class comes from the server's annotations — `readOnlyHint`
makes it `read`, `destructiveHint` makes it `destructive` — and is
`privileged` where the server says nothing, so a policy sends it to a person.
The annotations are the server's word; `risk={"ask_wiki_question": "read"}`
puts yours over it, and `trust_annotations=False` ignores them, so every tool
is `privileged` unless `risk=` says otherwise.
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
