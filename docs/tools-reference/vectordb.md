<!-- g:tool -->
# Vector store tools

## `vector_chunk` — `VectorChunkTool`

Split text into overlapping chunks for embedding and vector-store ingestion.

| | |
|---|---|
| Import | `from yait_aichain.tools import VectorChunkTool` |
| Risk class | `write` |

```python
VectorChunkTool(
    max_chars: int = 1500,
    overlap_chars: int = 150,
    merge_peers: bool = True,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | The text to chunk. |
| `options.max_chars` | `integer` |  | Maximum characters per chunk (default 1500). |
| `options.overlap_chars` | `integer` |  | Overlap in characters between consecutive chunks (default 150). |
| `options.merge_peers` | `boolean` |  | Merge adjacent small chunks with identical headings (default true). |

```python
result = VectorChunkTool()(
    input="…",
    options={"max_chars": …},
)
```

## `vector_upsert` — `VectorUpsertTool`

Insert or update records in a `VectorStore`.

| | |
|---|---|
| Import | `from yait_aichain.tools import VectorUpsertTool` |
| Risk class | `write` |

```python
VectorUpsertTool(
    store: VectorStore,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `list[object]` | ✓ | List of records to insert or update. |
| `options.collection` | `string` |  | Override the store's default collection / namespace. |

```python
result = VectorUpsertTool(store)(
    input="…",
    options={"collection": …},
)
```

## `vector_query` — `VectorQueryTool`

Semantic similarity search over a `VectorStore`.

| | |
|---|---|
| Import | `from yait_aichain.tools import VectorQueryTool` |
| Risk class | `write` |

```python
VectorQueryTool(
    store: VectorStore,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | The query text to search for semantically similar documents. |
| `options.n` | `integer` |  | Number of results to return (default 5). |
| `options.filter` | `object` |  | Provider-native metadata filter applied before ranking. |
| `options.collection` | `string` |  | Override the store's default collection / namespace. |

```python
result = VectorQueryTool(store)(
    input="…",
    options={"n": …},
)
```

## `vector_fetch` — `VectorFetchTool`

Retrieve records from a `VectorStore` by exact ID.

| | |
|---|---|
| Import | `from yait_aichain.tools import VectorFetchTool` |
| Risk class | `write` |

```python
VectorFetchTool(
    store: VectorStore,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `list[string]` | ✓ | List of document IDs to retrieve. |
| `options.collection` | `string` |  | Override the store's default collection / namespace. |

```python
result = VectorFetchTool(store)(
    input="…",
    options={"collection": …},
)
```

## `vector_delete` — `VectorDeleteTool`

Delete records from a `VectorStore`.

| | |
|---|---|
| Import | `from yait_aichain.tools import VectorDeleteTool` |
| Risk class | `write` |

```python
VectorDeleteTool(
    store: VectorStore,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `any` |  | List of document IDs to delete, or null to delete by filter only. |
| `options.filter` | `object` |  | Provider-native metadata filter for bulk deletion. |
| `options.collection` | `string` |  | Override the store's default collection / namespace. |

```python
result = VectorDeleteTool(store)(
    input="…",
    options={"filter": …},
)
```
<!-- /g:tool -->

## `Vectors` — the database as a utility

```python
from yait_aichain.tools import chunk, embed
from yait_aichain.utilities import Vectors

kb = Vectors("qdrant", collection="docs", embedder=embed("openai", model="text-embedding-3-small"))
kb.create(dimension=1536)
kb.upsert(chunk(max_chars=800).run(text))           # a method: code and chain
kb.query.run("how does caching work?", {"n": 5})    # a tool: .run, and an agent can take it
Agent(model, tools=[kb.query])                      # the agent searches, and can do nothing else
```

`create`, `upsert`, `fetch`, `delete` and `count` are methods — called directly,
by code or as a chain step (`(kb.upsert, "count", {"records": "pieces"})`) — and
`query` is the one tool. Given `kb` or `kb.upsert`, an agent refuses it and names
`query`. The store underneath is `kb.store`; `VectorDB(...)` and `vectorQuery(store)`
are what it is built on and still work.
