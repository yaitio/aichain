# API: vectors

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../../utilities/index.md).

<!-- g:tool -->
## `Vectors`

```python
Vectors(
    backend: str,
    collection: str,
    *,
    description: str,
    embedder=None,
    url: str | None = None,
    api_key: str | None = None,
    **settings,
)
```

A vector database: `create`, `upsert`, `fetch`, `delete` and
`count` for code; `query` a tool an agent can take.

```python
kb = Vectors("qdrant", collection="docs", description="the product documentation",
             embedder=embed("openai", model="text-embedding-3-small"))
kb.create(dimension=1536)
pieces = chunk(max_chars=800).run(text)                 # no ids: a record needs one
kb.upsert([{"id": f"doc-{i}", **p} for i, p in enumerate(pieces)])
kb.query.run("how does caching work?", {"n": 5})
Agent(model, tools=[kb.query])            # the agent searches, and can do nothing else
```

`description` says what the database holds, and is required: an agent
with two knowledge bases cannot tell them apart without it, and the search
tool's own description — "semantic similarity search" — says nothing of
what is searched. `query` reads "Search the product documentation — …".

`backend` is `chroma`, `pinecone` or `qdrant`; `collection`,
`embedder`, `url`, `api_key` and the backend's own settings
(Pinecone's `index`, `cloud`, `region`) are what `VectorDB` takes.
The store underneath is `kb.store`.

### `Vectors.create()`

```python
Vectors.create(dimension: int, metric: str = 'cosine')
```

Create the collection, *dimension* long, if it does not exist.

### `Vectors.upsert()`

```python
Vectors.upsert(records: list, batch_size: int | None = None) -> int
```

Insert or replace records — each `{"id", "text", …}`; returns how many.

### `Vectors.fetch()`

```python
Vectors.fetch(ids: list) -> list
```

The records with these ids.

### `Vectors.delete()`

```python
Vectors.delete(ids: list | None = None, filter: dict | None = None)
```

Remove records by id or by a metadata filter.

### `Vectors.count()`

```python
Vectors.count() -> int
```

How many records the collection holds.

## `chunk()`

```python
chunk(
    *,
    max_chars: int = 1500,
    overlap_chars: int | None = None,
    merge_peers: bool = True,
    name: str | None = None,
    description: str | None = None,
) -> yait_aichain.tools.chunk._chunk.Chunk
```

A chunking tool. The sizes are settings: a store is filled with pieces cut
one way, so they are fixed for the tool, not chosen per call.
`overlap_chars` defaults to a tenth of `max_chars` — 150 at the
default size.

```python
pieces = chunk(max_chars=800).run(long_text)    # [{"text", "chars", "metadata"}, …]
```

---

## The classes behind it

What the factory returns, with the schema a model is shown.

### `vector_chunk` — `Chunk`

Split text into overlapping chunks for embedding and vector-store ingestion.

| | |
|---|---|
| Import | `from yait_aichain.tools import Chunk` |
| Risk class | `read` |

```python
Chunk(
    max_chars: int = 1500,
    overlap_chars: int | None = None,
    merge_peers: bool = True,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | The text to chunk. |

```python
result = Chunk()(
    input="…",
)
```

### `vector_upsert` — `VectorUpsertTool`

Insert or update records in a `VectorStore`.

| | |
|---|---|
| Import | `from yait_aichain.utilities.vectors import VectorUpsertTool` |
| Risk class | `write` |

```python
VectorUpsertTool(
    store: VectorStore,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `list[object]` | ✓ | List of records to insert or update. |
| `options` | `object` |  | Upsert options. |

```python
result = VectorUpsertTool(store)(
    input="…",
)
```

### `vector_query` — `VectorQueryTool`

Semantic similarity search over a `VectorStore`.

| | |
|---|---|
| Import | `from yait_aichain.utilities.vectors import VectorQueryTool` |
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

```python
result = VectorQueryTool(store)(
    input="…",
    options={"n": …},
)
```

### `vector_fetch` — `VectorFetchTool`

Retrieve records from a `VectorStore` by exact ID.

| | |
|---|---|
| Import | `from yait_aichain.utilities.vectors import VectorFetchTool` |
| Risk class | `write` |

```python
VectorFetchTool(
    store: VectorStore,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `list[string]` | ✓ | List of document IDs to retrieve. |
| `options` | `object` |  | Fetch options. |

```python
result = VectorFetchTool(store)(
    input="…",
)
```

### `vector_delete` — `VectorDeleteTool`

Delete records from a `VectorStore`.

| | |
|---|---|
| Import | `from yait_aichain.utilities.vectors import VectorDeleteTool` |
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

```python
result = VectorDeleteTool(store)(
    input="…",
    options={"filter": …},
)
```
<!-- /g:tool -->
