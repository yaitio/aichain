<!-- g:tool -->
# Embeddings: text → vectors

## `embedOpenAI` — `EmbeddingOpenAI`

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

## `embedCohere` — `CohereEmbedder`

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

## `embedVoyage` — `EmbeddingVoyage`

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

## `embedGoogle` — `EmbeddingGoogle`

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

## `embedQwen` — `EmbeddingQwen`

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

## One tool, five providers

```python
from yait_aichain.tools import embed

vectors = embed("openai", model="text-embedding-3-small").run(["a text", "another"],
                                                              {"input_type": "document"})
vectors.embeddings[0][:3], vectors.dimensions, vectors.total_tokens
str(vectors)        # what a model reads: "2 vector(s) of 1536 dimensions from openai …"
```

`embed(service, model=…)` makes the tool — `openai`, `cohere`, `voyage`,
`google` or `qwen`. **The model is a setting, not an option:** vectors from two
models cannot be compared, so a store is filled by one, and the tool is fixed
to it. `run()` returns an `EmbeddingResult` — the vectors, the model, the
dimensions, the tokens — whose text is a one-line summary: an agent, or a
Skill prompt that substitutes it, reads that and never the numbers.

An agent rarely needs this tool: it reaches embeddings through a vector store's
search. It is a tool so that code, a chain and an agent call it alike.
`Embedding("openai/text-embedding-3-small")` is the factory it replaced and
still works.
