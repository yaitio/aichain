## `Vectors` — the database as a utility

```python
from yait_aichain.tools import chunk, embed
from yait_aichain.utilities import Vectors

kb = Vectors("qdrant", collection="docs", description="the product documentation",
             embedder=embed("openai", model="text-embedding-3-small"))
kb.create(dimension=1536)
pieces = chunk(max_chars=800).run(text)             # no ids: a record needs one
kb.upsert([{"id": f"doc-{i}", **p} for i, p in enumerate(pieces)])   # a method: code and chain
kb.query.run("how does caching work?", {"n": 5})    # a tool: .run, and an agent can take it
Agent(model, tools=[kb.query])                      # the agent searches, and can do nothing else
```

`description` says what the database holds, and is required: the search tool's
own description says nothing of what is searched, and an agent with two
knowledge bases could not tell them apart. `kb.query` reads "Search the product
documentation — …".

`create`, `upsert`, `fetch`, `delete` and `count` are methods — called directly,
by code or as a chain step (`(kb.upsert, "count", {"records": "pieces"})`) — and
`query` is the one tool. Given `kb` or `kb.upsert`, an agent refuses it and names
`query`. The store underneath is `kb.store`; `VectorDB(...)` and `vectorQuery(store)`
are what it is built on and still work.
