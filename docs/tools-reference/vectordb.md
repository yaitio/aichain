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
