# API: embed

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../../tools/retrieval.md).

<!-- g:tool -->
## `embed()`

```python
embed(
    service: str,
    *,
    model: str,
    dimensions: int | None = None,
    batch_size: int | None = None,
    api_key: str | None = None,
    name: str | None = None,
    description: str | None = None,
) -> yait_aichain.tools.embed._base.Embedder
```

An embedding tool for *service* — one of `SERVICES` — and *model*.

The model, the vectors' `dimensions` and the `batch_size` are
settings, not options: vectors from two models, or of two lengths, cannot
share a store, so the tool is fixed to one. `name` and `description`
default to the provider's; `api_key` to its environment variable.

```python
vectors = embed("openai", model="text-embedding-3-small").run(["a text", "another"])
vectors.embeddings[0][:3], vectors.dimensions
str(vectors)          # what a model reads: "2 vector(s) of 1536 dimensions …"
```

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

## The classes behind it

What the factory returns, with the schema a model is shown.

### `embedOpenAI` — `EmbeddingOpenAI`

Embedder for the OpenAI Embeddings API.

| | |
|---|---|
| Import | `from yait_aichain.tools import EmbeddingOpenAI` |
| Risk class | `read` |
| Key | `OPENAI_API_KEY` |

```python
EmbeddingOpenAI(
    model: str,
    *,
    api_key: str | None = None,
    name: str | None = None,
    description: str | None = None,
    dimensions: int | None = None,
    batch_size: int | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `list[string]` | ✓ | The texts to embed — or one text. |
| `options.input_type` | `string` |  | What the texts are, for providers that embed the two differently. One of `query`, `document`. |

```python
result = EmbeddingOpenAI("<model>")(
    input="…",
    options={"input_type": …},
)
```

### `embedCohere` — `EmbeddingCohere`

Embedder for the Cohere Embed API (v2).

| | |
|---|---|
| Import | `from yait_aichain.tools import EmbeddingCohere` |
| Risk class | `read` |
| Key | `COHERE_API_KEY` |

```python
EmbeddingCohere(
    model: str,
    *,
    api_key: str | None = None,
    name: str | None = None,
    description: str | None = None,
    dimensions: int | None = None,
    batch_size: int | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `list[string]` | ✓ | The texts to embed — or one text. |
| `options.input_type` | `string` |  | What the texts are, for providers that embed the two differently. One of `query`, `document`. |

```python
result = EmbeddingCohere("<model>")(
    input="…",
    options={"input_type": …},
)
```

### `embedVoyage` — `EmbeddingVoyage`

Embedder for the Voyage AI Embeddings API.

| | |
|---|---|
| Import | `from yait_aichain.tools import EmbeddingVoyage` |
| Risk class | `read` |
| Key | `VOYAGE_API_KEY` |

```python
EmbeddingVoyage(
    model: str,
    *,
    api_key: str | None = None,
    name: str | None = None,
    description: str | None = None,
    dimensions: int | None = None,
    batch_size: int | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `list[string]` | ✓ | The texts to embed — or one text. |
| `options.input_type` | `string` |  | What the texts are, for providers that embed the two differently. One of `query`, `document`. |

```python
result = EmbeddingVoyage("<model>")(
    input="…",
    options={"input_type": …},
)
```

### `embedGoogle` — `EmbeddingGoogle`

Embedder for the Google AI (Gemini) Embeddings API.

| | |
|---|---|
| Import | `from yait_aichain.tools import EmbeddingGoogle` |
| Risk class | `read` |
| Key | `GOOGLE_AI_API_KEY` or `GOOGLE_API_KEY` |

```python
EmbeddingGoogle(
    model: str,
    *,
    api_key: str | None = None,
    name: str | None = None,
    description: str | None = None,
    dimensions: int | None = None,
    batch_size: int | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `list[string]` | ✓ | The texts to embed — or one text. |
| `options.input_type` | `string` |  | What the texts are, for providers that embed the two differently. One of `query`, `document`. |

```python
result = EmbeddingGoogle("<model>")(
    input="…",
    options={"input_type": …},
)
```

### `embedQwen` — `EmbeddingQwen`

Embedder for the Alibaba DashScope Embeddings API.

| | |
|---|---|
| Import | `from yait_aichain.tools import EmbeddingQwen` |
| Risk class | `read` |
| Key | `DASHSCOPE_API_KEY` |

```python
EmbeddingQwen(
    model: str,
    *,
    api_key: str | None = None,
    region: str | None = None,
    **defaults,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `list[string]` | ✓ | The texts to embed — or one text. |
| `options.input_type` | `string` |  | What the texts are, for providers that embed the two differently. One of `query`, `document`. |

```python
result = EmbeddingQwen("<model>")(
    input="…",
    options={"input_type": …},
)
```
<!-- /g:tool -->
