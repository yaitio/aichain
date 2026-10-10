# API: embed

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../../tools/embed.md).

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

---

## The classes behind it

What the factory returns, with the schema a model is shown.

### `embedOpenAI` — `EmbeddingOpenAI`

Embedder for the OpenAI Embeddings API.

| | |
|---|---|
| Import | `from yait_aichain.tools import EmbeddingOpenAI` |
| Risk class | `read` |
| Also exported as | `OpenAIEmbedder` |
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

### `embedCohere` — `CohereEmbedder`

Embedder for the Cohere Embed API (v2).

| | |
|---|---|
| Import | `from yait_aichain.tools import CohereEmbedder` |
| Risk class | `read` |
| Also exported as | `EmbeddingCohere` |
| Key | `COHERE_API_KEY` |

```python
CohereEmbedder(
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
result = CohereEmbedder("<model>")(
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
| Also exported as | `VoyageEmbedder` |
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
| Also exported as | `GoogleEmbedder` |
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
