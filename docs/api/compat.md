# API: older names

Names from before the 3.x tool convention, kept so code written against them
still runs. New code reaches the same things through the factories and the
utilities — `embed(...)`, `rerank(...)`, `Vectors(...)`, `Shell(...)`,
`MCP(...)` — and each entry below says which. Generated from the code.

<!-- g:api -->
[`VectorDB`](#vectordb) · [`VectorStore`](#vectorstore) · [`VectorRecord`](#vectorrecord) · [`VectorBackend`](#vectorbackend) · [`ChromaBackend`](#chromabackend) · [`PineconeBackend`](#pineconebackend) · [`QdrantBackend`](#qdrantbackend) · [`Embedding`](#embedding) · [`EmbeddingResult`](#embeddingresult) · [`Reranker`](#reranker) · [`RerankResult`](#rerankresult) · [`TTS`](#tts) · [`STT`](#stt) · [`LocalTools`](#localtools) · [`MCPTools`](#mcptools) · [`localBrowse`](#localbrowse) · [`localRead`](#localread) · [`localWrite`](#localwrite) · [`localRun`](#localrun) · [`vectorChunk`](#vectorchunk) · [`vectorQuery`](#vectorquery) · [`vectorUpsert`](#vectorupsert) · [`vectorFetch`](#vectorfetch) · [`vectorDelete`](#vectordelete)

---

## `VectorDB()`

```python
VectorDB(
    provider: str,
    collection: str,
    *,
    embedder: object | None = None,
    url: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> yait_aichain.utilities.vectors._base.VectorStore
```

Return a `VectorStore` for *provider*.

**Parameters**

- `provider` (*str*) — Backend name: `"chroma"`, `"pinecone"`, or `"qdrant"`.
  Aliases: `"chromadb"`.
- `collection` (*str*) — Default collection (Chroma/Qdrant) or namespace (Pinecone) used for
  all operations.  Can be overridden per-call via `options["collection"]`.
- `embedder` (*Embedder | None, optional*, default `None`) — An `Embedder` instance used to auto-vectorise
  text on `upsert()` and `query()`.  When `None`, callers must
  supply pre-computed `vector` fields in their records.
- `url` (*str | None, optional*, default `None`) — Base URL of the server.  Overrides the provider's environment variable
  default (`CHROMA_URL`, `QDRANT_URL`).
- `api_key` (*str | None, optional*, default `None`) — API key.  Overrides the provider's environment variable
  (`CHROMA_API_KEY`, `PINECONE_API_KEY`, `QDRANT_API_KEY`).
- `**kwargs` — Additional provider-specific arguments forwarded to the backend
  constructor.  Common extras:
    `index`  (str)  — Pinecone index name
    `cloud`  (str)  — Pinecone cloud (default `"aws"`)
    `region` (str)  — Pinecone region (default `"us-east-1"`)

**Returns**

- `VectorStore` — Ready-to-use store with `create`, `query`, `upsert`, `fetch`,
  `delete`, and `count` methods.

**Raises**

- `ValueError` — When *provider* is not recognised.

**Examples**

```python
from yait_aichain.tools.vectordb  import VectorDB
from yait_aichain.tools.embedding import Embedding

store = VectorDB(
    "chroma",
    collection = "docs",
    embedder   = Embedding("openai/text-embedding-3-small"),
)
```

---

## `VectorStore`

```python
VectorStore(backend: VectorBackend, collection: str, embedder: Any | None = None)
```

Provider-agnostic vector store.

Wraps a `VectorBackend` and an optional `Embedder`.
Fixes the active `collection` so callers never repeat it.

**Parameters**

- `backend` (*VectorBackend*) — A concrete backend (Chroma, Pinecone, Qdrant, …).
- `collection` (*str*) — Default collection / namespace for all operations.
- `embedder` (*Embedder | None, optional*, default `None`) — When provided, text is automatically vectorised before storage and
  at query time.  When omitted, callers must supply pre-computed
  `vector` fields in their records.

**Examples**

With auto-embedding:

```python
from yait_aichain.tools.embedding import Embedding
from yait_aichain.tools.vectordb  import VectorDB

store = VectorDB(
    "chroma",
    collection = "knowledge_base",
    embedder   = Embedding("openai/text-embedding-3-small"),
)
store.upsert([{"id": "1", "text": "LLMs use KV cache …"}])
results = store.query("how does caching work?", n=3)
```

Without embedding (supply vectors yourself):

```python
store = VectorDB("qdrant", collection="dense_index")
store.upsert([{"id": "1", "text": "…", "vector": [0.1, 0.2, …]}])
```

### `VectorStore.create()`

```python
VectorStore.create(dimension: int, metric: str = 'cosine')
```

Create the collection in the backend.

**Parameters**

- `dimension` (*int*) — Vector size (must match the embedder's output dimension).
- `metric` (*`"cosine"` | `"euclidean"` | `"dot"`*, default `'cosine'`) — Distance metric used for similarity ranking.

### `VectorStore.query()`

```python
VectorStore.query(
    text: str,
    n: int = 5,
    filter: dict | None = None,
) -> list[VectorRecord]
```

Semantic similarity search.

Embeds *text* (requires an embedder) and returns the *n* most
similar records, sorted by descending similarity score.

**Parameters**

- `text` (*str          — query string*)
- `n` (*int          — number of results (default 5)*, default `5`)
- `filter` (*dict | None  — provider-native metadata filter*, default `None`)

**Returns**

- `list[VectorRecord]` — Records with `score` in [0, 1], highest = most similar, on
  every backend and every metric. `raw_score` carries what the
  backend actually returned. The one exception is the `dot`
  metric, which is unbounded and scale-dependent: it is passed
  through and says so, because a fabricated bound would look
  comparable when it is not.

### `VectorStore.upsert()`

```python
VectorStore.upsert(records: list[dict], batch_size: 'int | None' = None) -> int
```

Insert or update records.

Each dict must have:
  `id`       : str            — unique identifier
  `text`     : str            — document content
  `metadata` : dict           — optional payload / metadata
  `vector`   : list[float]    — optional pre-computed vector
                                  (required when no embedder is set)

Missing vectors are auto-computed in a single batched embedding call.
Writes are sent to the backend in batches of *batch_size* records
(default 50) so large ingests don't exceed provider request limits.

**Returns**

- `int` — Number of records upserted.

### `VectorStore.fetch()`

```python
VectorStore.fetch(ids: list[str]) -> list[VectorRecord]
```

Retrieve specific records by ID (no score).

### `VectorStore.delete()`

```python
VectorStore.delete(ids: list[str] | None = None, filter: dict | None = None)
```

Remove records.

Pass *ids* to delete specific records, *filter* to delete by
metadata condition, or both.  At least one must be provided and
non-empty: an empty list/dict is rejected rather than treated as
"match everything", because some backends (Pinecone, Qdrant) would
otherwise interpret it as a full-collection wipe.

### `VectorStore.count()`

```python
VectorStore.count() -> int
```

Return the total number of records in the collection.

---

## `VectorRecord`

```python
VectorRecord(
    id: str,
    text: str,
    score: float | None = None,
    metadata: dict = <factory>,
    raw_score: float | None = None,
)
```

Uniform data model for a single vector DB entry.

**Attributes**

- `id` (*str            — unique document identifier*)
- `text` (*str            — original text content*)
- `score` (*float | None   — similarity in [0, 1], higher is more similar*, default `None`)
- `raw_score: float | None   — what the backend returned, unconverted`
- `metadata` (*dict           — arbitrary key-value payload / metadata*, default `<factory>`)
- `**``score`` means one thing on every backend**, which it did not until`
- `2026-09-13. Chroma normalised its distances to [0, 1] and Qdrant and`
- `Pinecone handed their raw numbers through, so the same field carried a`
- `cosine similarity in [-1, 1] on one, a normalised score on another, and —`
- `for the euclidean metric — a *distance*, where **lower** is more similar.`
- `The docstring on `query()` promised "highest score = most similar" and it`
- `was false for two backends on one metric. Anyone comparing retrievers, or`
- `setting a threshold and moving a collection, was reading three scales as`
- `one.`
- ```raw_score`` keeps whatever the backend said. Normalising is a`
- `conversion, and a conversion that throws the original away cannot be`
- `checked.`

### `VectorRecord.to_dict()`

```python
VectorRecord.to_dict() -> dict
```

Return a plain dict (JSON-serialisable).

---

## `VectorBackend`

```python
VectorBackend(*, url: str, api_key: str | None = None, **_kwargs)
```

Abstract base class for vector DB provider REST adapters.

Subclasses must:
  1. Set `provider` (class attribute).
  2. Call `super().__init__(url=..., api_key=...)` in their `__init__`.
  3. Implement all six abstract methods.

Shared HTTP helpers (_get, _post, _put, _delete_req) handle JSON
encoding/decoding and error checking so subclasses stay concise.

Environment variables
---------------------
Each backend reads its URL and API key from environment variables when
the caller does not pass them explicitly.  Subclasses document the
specific variable names.

### `VectorBackend.create_collection()`

```python
VectorBackend.create_collection(collection: str, dimension: int, metric: str = 'cosine')
```

Create a new collection / index with the given vector dimension.

### `VectorBackend.query()`

```python
VectorBackend.query(
    collection: str,
    vector: list[float],
    n: int = 5,
    filter: dict | None = None,
) -> list[VectorRecord]
```

Return the *n* most similar records to *vector*.

### `VectorBackend.upsert()`

```python
VectorBackend.upsert(
    collection: str,
    records: list[VectorRecord],
    vectors: list[list[float]],
)
```

Insert or update *records* together with their *vectors*.

### `VectorBackend.fetch()`

```python
VectorBackend.fetch(collection: str, ids: list[str]) -> list[VectorRecord]
```

Return records matching *ids* (no score populated).

### `VectorBackend.delete()`

```python
VectorBackend.delete(
    collection: str,
    ids: list[str] | None = None,
    filter: dict | None = None,
)
```

Delete records by *ids* or metadata *filter* (at least one required).

### `VectorBackend.count()`

```python
VectorBackend.count(collection: str) -> int
```

Return the number of records in *collection*.

---

## `ChromaBackend`

```python
ChromaBackend(
    *,
    url: str | None = None,
    api_key: str | None = None,
    tenant: str | None = None,
    database: str | None = None,
    **kwargs,
)
```

VectorBackend for ChromaDB (REST API v2).

**Parameters**

- `url` (*str | None*, default `None`) — Base URL of the Chroma HTTP server.
  Defaults to the `CHROMA_URL` env var or `http://localhost:8000`.
- `api_key` (*str | None*, default `None`) — API key for Chroma Cloud.  Omit for local/self-hosted deployments.
  Defaults to `CHROMA_API_KEY` env var.
- `tenant` (*str | None*, default `None`) — Tenant name.  Defaults to `CHROMA_TENANT` env var or
  `default_tenant`.
- `database` (*str | None*, default `None`) — Database name.  Defaults to `CHROMA_DATABASE` env var or
  `default_database`.

**Examples**

Local server:

```python
from yait_aichain.tools.vectordb import VectorDB
from yait_aichain.tools.embedding import Embedding

store = VectorDB(
    "chroma",
    collection = "my_docs",
    embedder   = Embedding("openai/text-embedding-3-small"),
    url        = "http://localhost:8000",
)
```

Chroma Cloud:

```python
store = VectorDB(
    "chroma",
    collection = "my_docs",
    embedder   = Embedding("openai/text-embedding-3-small"),
    url        = "https://api.trychroma.com",
    api_key    = "ch-...",
)
```

### `ChromaBackend.create_collection()`

```python
ChromaBackend.create_collection(collection: str, dimension: int, metric: str = 'cosine')
```

Create a new collection / index with the given vector dimension.

### `ChromaBackend.query()`

```python
ChromaBackend.query(
    collection: str,
    vector: list[float],
    n: int = 5,
    filter: dict | None = None,
) -> list[VectorRecord]
```

Return the *n* most similar records to *vector*.

### `ChromaBackend.upsert()`

```python
ChromaBackend.upsert(
    collection: str,
    records: list[VectorRecord],
    vectors: list[list[float]],
)
```

Insert or update *records* together with their *vectors*.

### `ChromaBackend.fetch()`

```python
ChromaBackend.fetch(collection: str, ids: list[str]) -> list[VectorRecord]
```

Return records matching *ids* (no score populated).

### `ChromaBackend.delete()`

```python
ChromaBackend.delete(
    collection: str,
    ids: list[str] | None = None,
    filter: dict | None = None,
)
```

Delete records by *ids* or metadata *filter* (at least one required).

### `ChromaBackend.count()`

```python
ChromaBackend.count(collection: str) -> int
```

Return the number of records in *collection*.

---

## `PineconeBackend`

```python
PineconeBackend(
    *,
    index: str | None = None,
    api_key: str | None = None,
    url: str | None = None,
    cloud: str = 'aws',
    region: str = 'us-east-1',
    metric: str = 'cosine',
    **kwargs,
)
```

VectorBackend for Pinecone (serverless REST API).

**Parameters**

- `index` (*str*, default `None`) — Name of the Pinecone index.  Defaults to `PINECONE_INDEX` env var.
- `api_key` (*str | None*, default `None`) — Pinecone API key.  Defaults to `PINECONE_API_KEY` env var.
- `url` (*str | None*, default `None`) — Index host URL (`https://<index-host>`).  When omitted the host
  is auto-discovered from the control plane on the first data call.
- `cloud` (*str*, default `'aws'`) — Cloud for new index creation (default `"aws"`).
- `region` (*str*, default `'us-east-1'`) — Region for new index creation (default `"us-east-1"`).

**Examples**

```python
from yait_aichain.tools.vectordb import VectorDB
from yait_aichain.tools.embedding import Embedding

store = VectorDB(
    "pinecone",
    collection = "my_namespace",
    embedder   = Embedding("openai/text-embedding-3-small"),
    index      = "my-index",
)
store.upsert([{"id": "1", "text": "hello world"}])
results = store.query("greeting", n=3)
```

### `PineconeBackend.create_collection()`

```python
PineconeBackend.create_collection(
    collection: str,
    dimension: int,
    metric: str = 'cosine',
)
```

Create a serverless Pinecone index.

In Pinecone the index is shared; *collection* becomes a namespace
inside it.  This method creates the index if it does not exist.
HTTP 409 (already exists) is treated as success.

### `PineconeBackend.query()`

```python
PineconeBackend.query(
    collection: str,
    vector: list[float],
    n: int = 5,
    filter: dict | None = None,
) -> list[VectorRecord]
```

Return the *n* most similar records to *vector*.

### `PineconeBackend.upsert()`

```python
PineconeBackend.upsert(
    collection: str,
    records: list[VectorRecord],
    vectors: list[list[float]],
)
```

Insert or update *records* together with their *vectors*.

### `PineconeBackend.fetch()`

```python
PineconeBackend.fetch(collection: str, ids: list[str]) -> list[VectorRecord]
```

Return records matching *ids* (no score populated).

### `PineconeBackend.delete()`

```python
PineconeBackend.delete(
    collection: str,
    ids: list[str] | None = None,
    filter: dict | None = None,
)
```

Delete records by *ids* or metadata *filter* (at least one required).

### `PineconeBackend.count()`

```python
PineconeBackend.count(collection: str) -> int
```

Return the number of records in *collection*.

---

## `QdrantBackend`

```python
QdrantBackend(
    *,
    url: str | None = None,
    api_key: str | None = None,
    metric: str = 'cosine',
    **kwargs,
)
```

VectorBackend for Qdrant (REST API).

**Parameters**

- `url` (*str | None*, default `None`) — Base URL of the Qdrant server.
  Defaults to the `QDRANT_URL` env var or `http://localhost:6333`.
- `api_key` (*str | None*, default `None`) — API key for Qdrant Cloud.  Omit for local/self-hosted deployments.
  Defaults to `QDRANT_API_KEY` env var.

**Examples**

Local server:

```python
from yait_aichain.tools.vectordb import VectorDB
from yait_aichain.tools.embedding import Embedding

store = VectorDB(
    "qdrant",
    collection = "knowledge_base",
    embedder   = Embedding("openai/text-embedding-3-small"),
    url        = "http://localhost:6333",
)
```

Qdrant Cloud:

```python
store = VectorDB(
    "qdrant",
    collection = "knowledge_base",
    embedder   = Embedding("openai/text-embedding-3-small"),
    url        = "https://xyz.us-east4-0.gcp.cloud.qdrant.io",
    api_key    = "qdrant-...",
)
```

### `QdrantBackend.create_collection()`

```python
QdrantBackend.create_collection(collection: str, dimension: int, metric: str = 'cosine')
```

Create a new collection / index with the given vector dimension.

### `QdrantBackend.query()`

```python
QdrantBackend.query(
    collection: str,
    vector: list[float],
    n: int = 5,
    filter: dict | None = None,
) -> list[VectorRecord]
```

Return the *n* most similar records to *vector*.

### `QdrantBackend.upsert()`

```python
QdrantBackend.upsert(
    collection: str,
    records: list[VectorRecord],
    vectors: list[list[float]],
)
```

Insert or update *records* together with their *vectors*.

### `QdrantBackend.fetch()`

```python
QdrantBackend.fetch(collection: str, ids: list[str]) -> list[VectorRecord]
```

Return records matching *ids* (no score populated).

### `QdrantBackend.delete()`

```python
QdrantBackend.delete(
    collection: str,
    ids: list[str] | None = None,
    filter: dict | None = None,
)
```

Delete records by *ids* or metadata *filter* (at least one required).

### `QdrantBackend.count()`

```python
QdrantBackend.count(collection: str) -> int
```

Return the number of records in *collection*.

---

## `Embedding()`

```python
Embedding(
    model: str,
    *,
    api_key: str | None = None,
    **defaults,
) -> yait_aichain.tools.embed._base.Embedder
```

Return an `Embedder` instance for *model*.

**Parameters**

- `model` (*str*) — Model identifier.  May be a bare name (`"voyage-3-large"`) or a
  provider-prefixed string (`"voyage/voyage-3-large"`).
- `api_key` (*str | None, optional*, default `None`) — Override the provider's environment variable.
- `dimensions, batch_size` — Settings of the embedder — the vectors' length, where the model can
  shorten them, and the most texts in one request. A call takes only
  `input_type`.

**Returns**

- `Embedder` — A concrete `Embedder` subclass ready to call `embed`.

**Raises**

- `ValueError` — When the provider cannot be inferred from *model*.

---

## `EmbeddingResult`

```python
EmbeddingResult(
    embeddings: list[list[float]],
    model: str,
    provider: str,
    dimensions: int,
    total_tokens: int | None = None,
    metadata: dict = <factory>,
)
```

Uniform return type for every `Embedder` call.

**Attributes**

- `embeddings` (*list[list[float]]*) — One float vector per input text, in the same order as the input.
- `model` (*str*) — The model name actually used (may differ from the requested name when
  a provider applies an alias).
- `provider` (*str*) — Canonical provider identifier: `"openai"`, `"cohere"`,
  `"voyage"`, or `"google"`.
- `dimensions` (*int*) — Length of each embedding vector.  `0` when *embeddings* is empty.
- `total_tokens` (*int | None*, default `None`) — Total input tokens consumed, as reported by the provider.
  `None` when the provider does not expose token counts.
- `metadata` (*dict*, default `<factory>`) — Provider-specific extras (e.g. Cohere's billed units, Voyage's usage
  object).  Empty dict when nothing extra is returned.

**Examples**

```python
result = embedder.embed(["hello world", "foo bar"])
print(len(result.embeddings))      # 2
print(len(result.embeddings[0]))   # e.g. 1536
print(result.dimensions)           # 1536
print(result.total_tokens)         # e.g. 4
```

---

## `Reranker()`

```python
Reranker(
    model: str,
    *,
    api_key: str | None = None,
    **kwargs,
) -> yait_aichain.tools.rerank._base.RerankBase
```

Return a `RerankBase` instance for *model*.

**Parameters**

- `model` (*str*) — Model identifier.  Accepts `"provider/model"` (unambiguous) or
  a bare model name resolved by heuristic.

  Supported provider prefixes:
    `"cohere/"`  — Cohere Rerank
    `"voyage/"`  — Voyage AI Rerank
    `"qwen/"`    — Alibaba DashScope GTE-Rerank
- `api_key` (*str | None, optional*, default `None`) — Override the provider's environment variable.
- `**kwargs` — Extra arguments forwarded to the backend constructor.
  Common extras:
    `region` (str) — DashScope region for Qwen (`"ap"`, `"us"`, `"cn"`, `"hk"`)

**Returns**

- `RerankBase` — A concrete reranker ready to call `rerank`.

**Raises**

- `ValueError` — When the provider cannot be inferred from *model*.

**Examples**

```python
from yait_aichain.tools.reranking import Reranker

r1 = Reranker("cohere/rerank-v3.5")
r2 = Reranker("voyage/rerank-2")
r3 = Reranker("qwen/gte-rerank")

# Bare model name
r4 = Reranker("rerank-v3.5")         # → Cohere
r5 = Reranker("gte-rerank")           # → Qwen
```

---

## `RerankResult`

```python
RerankResult(
    results: list[dict],
    model: str,
    provider: str,
    total_tokens: int | None = None,
    metadata: dict = <factory>,
)
```

Uniform return type for every `RerankBase` call.

**Attributes**

- `results` (*list[dict]*) — Reranked documents, sorted by descending relevance score.
  Each item:
    `index`    int   — position in the original input list
    `text`     str   — document text
    `score`    float — relevance score (higher = more relevant)
    All original dict fields from the input are preserved when
    documents were passed as dicts.
- `model` (*str*)
- `provider` (*str*)
- `total_tokens` (*int | None*, default `None`)
- `metadata` (*dict*, default `<factory>`) — Provider-specific extras (e.g. Cohere billed units).

**Examples**

```python
result = reranker.rerank("how does caching work?", documents)
for item in result:
    print(item["score"], item["text"][:60])

texts = result.texts()    # list[str], top to bottom
scores = result.scores()  # list[float]
```

### `RerankResult.texts()`

```python
RerankResult.texts() -> list[str]
```

Return document texts in ranked order.

### `RerankResult.scores()`

```python
RerankResult.scores() -> list[float]
```

Return relevance scores in ranked order.

---

## `TTS()`

```python
TTS(
    provider: str,
    *,
    api_key: str | None = None,
) -> yait_aichain.tools.speech._tts.convertToSpeech
```

The 3.4 name of `speech` — `TTS("openai")` is `speech("openai")`.

---

## `STT()`

```python
STT(
    provider: str,
    *,
    api_key: str | None = None,
) -> yait_aichain.tools.transcribe._stt.convertToText
```

The 3.4 name of `transcribe` — `STT("openai")` is `transcribe("openai")`.

---

## `LocalTools()`

```python
LocalTools(root_dir: str | None = None) -> list
```

Return all four local tools pre-configured with the same sandbox root.

**Parameters**

- `root_dir` (*str | None*, default `None`) — Sandbox root directory.  Defaults to `os.getcwd()`.

**Returns**

- `list` — `[LocalBrowseTool, LocalReadTool, LocalWriteTool, LocalRunTool]`

**Examples**

```python
from yait_aichain import Agent, Model
from yait_aichain.tools import LocalTools

agent = Agent(Model("claude-haiku-5-5"), tools=LocalTools("/my/project"))
# the same as tools=Shell("/my/project").tools, which new code uses
```

---

## `MCPTools()`

```python
MCPTools(
    server: 'str | dict | Any',
    *,
    filter: 'list[str] | None' = None,
    headers: 'dict | None' = None,
    env: 'dict | None' = None,
    cwd: 'str | None' = None,
    transport: 'str | None' = None,
    timeout: 'float | None' = 60.0,
    call_timeout: 'float | None' = None,
) -> 'list[MCPTool]'
```

Connect to an MCP server, discover all exposed tools, and return one
`MCPTool` per tool.

The discovery connection is opened and closed within this call.  Each
returned `MCPTool` opens its own connection independently when
`run()` is called later.

`timeout` caps the whole discovery (connect + handshake + list_tools);
a hung server raises `TimeoutError` instead of blocking forever.
Pass `timeout=None` to wait indefinitely.

**Parameters**

- `server` (*str | dict | object*) — MCP server specification.

  **String forms (transport auto-detected):**

  * `"./server.py"` or `"python ./server.py --flag"`
    → STDIO subprocess.
  * `"https://api.example.com/mcp"`
    → Streamable HTTP.
  * `"https://api.example.com/sse"` (URL ends with `/sse`)
    → SSE (legacy).

  **Dict forms (explicit):**

  * `{"command": "python", "args": ["server.py"]}`
    → STDIO.
  * `{"command": "npx", "args": ["@scope/server", "/path"], "env": {...}}`
    → STDIO with env vars.
  * `{"url": "https://api.example.com/mcp"}`
    → HTTP.
  * `{"url": "https://...", "headers": {"Authorization": "Bearer ..."}}`
    → HTTP with auth.
- `filter` (*list[str] | None, optional*, default `None`) — When provided, only return tools whose names appear in this list.
  All other tools are silently skipped.
- `headers` (*dict | None, optional*, default `None`) — HTTP headers for HTTP / SSE transports.  Merged with any headers
  inside a dict-style *server* spec.
- `env` (*dict | None, optional*, default `None`) — Additional environment variables for STDIO transports.  Merged
  with the current process environment and any `env` inside a
  dict-style *server* spec.
- `cwd` (*str | None, optional*, default `None`) — Working directory for STDIO subprocess servers.
- `transport` (*str | None, optional*, default `None`) — Force a specific transport: `"http"`, `"sse"`, or `"stdio"`.
  Auto-detected from *server* when omitted.

**Returns**

- `list[MCPTool]` — One `MCPTool` per tool exposed by the server (or filtered
  subset), ordered as returned by the server.

**Raises**

- `ImportError` — When `fastmcp` is not installed (`pip install fastmcp`).
- `RuntimeError` — On connection failure.

**Examples**

```python
from yait_aichain.tools import MCPTools
from yait_aichain.agent import Agent
from yait_aichain.models import Model

# ── HTTP server ────────────────────────────────────────────────
tools = MCPTools("https://api.example.com/mcp")
for t in tools:
    print(f"{t.name}: {t.description}")

# ── HTTP with auth ─────────────────────────────────────────────
tools = MCPTools(
    "https://api.example.com/mcp",
    headers={"Authorization": "Bearer sk-..."},
)

# ── STDIO subprocess ───────────────────────────────────────────
tools = MCPTools({"command": "python", "args": ["./server.py"]})

# ── npx server ────────────────────────────────────────────────
tools = MCPTools(
    {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-brave-search"]},
    env={"BRAVE_API_KEY": "..."},
)

# ── Only specific tools ────────────────────────────────────────
tools = MCPTools(
    "https://api.example.com/mcp",
    filter=["search_web", "get_weather"],
)

# ── Register in an Agent ───────────────────────────────────────
agent = Agent(
    model = Model("claude-sonnet-4-6"),
    tools = MCPTools("https://api.example.com/mcp"),
)
```

---

## `localBrowse()`

```python
localBrowse(root_dir: str | None = None) -> LocalBrowseTool
```

Return a `LocalBrowseTool` sandboxed to *root_dir*.

---

## `localRead()`

```python
localRead(root_dir: str | None = None) -> LocalReadTool
```

Return a `LocalReadTool` sandboxed to *root_dir*.

---

## `localWrite()`

```python
localWrite(root_dir: str | None = None) -> LocalWriteTool
```

Return a `LocalWriteTool` sandboxed to *root_dir*.

---

## `localRun()`

```python
localRun(root_dir: str | None = None) -> LocalRunTool
```

Return a `LocalRunTool` sandboxed to *root_dir*.

---

## `vectorChunk()`

```python
vectorChunk(
    max_chars: int = 1500,
    overlap_chars: 'int | None' = None,
    merge_peers: bool = True,
) -> VectorChunkTool
```

Return a configured `VectorChunkTool`.

**Parameters**

- `max_chars` (*int  — maximum characters per chunk (default 1500)*, default `1500`)
- `overlap_chars` (*int  — overlap in characters (default 150)*, default `None`)
- `merge_peers` (*bool — merge adjacent small chunks with identical headings*, default `True`)

**Examples**

```python
from yait_aichain.tools.vectordb import vectorChunk

chunker = vectorChunk(max_chars=800, overlap_chars=80)
chunks  = chunker.run(my_text)
```

---

## `vectorQuery()`

```python
vectorQuery(store: VectorStore) -> VectorQueryTool
```

Return a `VectorQueryTool` bound to *store*.

---

## `vectorUpsert()`

```python
vectorUpsert(store: VectorStore) -> VectorUpsertTool
```

Return a `VectorUpsertTool` bound to *store*.

---

## `vectorFetch()`

```python
vectorFetch(store: VectorStore) -> VectorFetchTool
```

Return a `VectorFetchTool` bound to *store*.

---

## `vectorDelete()`

```python
vectorDelete(store: VectorStore) -> VectorDeleteTool
```

Return a `VectorDeleteTool` bound to *store*.
<!-- /g:api -->
