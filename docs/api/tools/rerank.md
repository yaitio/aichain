# API: rerank

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../../tools/retrieval.md).

<!-- g:tool -->
## `rerank()`

```python
rerank(
    service: str,
    *,
    model: str,
    api_key: str | None = None,
    name: str | None = None,
    description: str | None = None,
) -> yait_aichain.tools.rerank._base.RerankBase
```

A reranking tool for *service* — one of `SERVICES` — and *model*.

```python
top = rerank("cohere", model="rerank-v3.5").run(docs, {"query": "how does caching work?", "top_n": 5})
```

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

## The classes behind it

What the factory returns, with the schema a model is shown.

### `rerankCohere` — `RerankCohere`

Reranker for the Cohere Rerank API (v2).

| | |
|---|---|
| Import | `from yait_aichain.tools import RerankCohere` |
| Risk class | `read` |
| Key | `COHERE_API_KEY` |

```python
RerankCohere(
    model: str,
    *,
    api_key: str | None = None,
    name: str | None = None,
    description: str | None = None,
    **kwargs,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `list[string]` | ✓ | The documents to put in order — texts, or records with a 'text' field. |
| `options.query` | `string` | ✓ | The question the documents should answer. |
| `options.top_n` | `integer` |  | How many to return, best first (default: all). |

```python
result = RerankCohere("<model>")(
    input="…",
    options={"query": …},
)
```

### `rerankVoyage` — `RerankVoyage`

Reranker for the Voyage AI Rerank API.

| | |
|---|---|
| Import | `from yait_aichain.tools import RerankVoyage` |
| Risk class | `read` |
| Key | `VOYAGE_API_KEY` |

```python
RerankVoyage(
    model: str,
    *,
    api_key: str | None = None,
    name: str | None = None,
    description: str | None = None,
    **kwargs,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `list[string]` | ✓ | The documents to put in order — texts, or records with a 'text' field. |
| `options.query` | `string` | ✓ | The question the documents should answer. |
| `options.top_n` | `integer` |  | How many to return, best first (default: all). |

```python
result = RerankVoyage("<model>")(
    input="…",
    options={"query": …},
)
```

### `rerankQwen` — `RerankQwen`

Reranker for the Alibaba DashScope Text-Rerank API.

| | |
|---|---|
| Import | `from yait_aichain.tools import RerankQwen` |
| Risk class | `read` |
| Key | `DASHSCOPE_API_KEY` |

```python
RerankQwen(
    model: str,
    *,
    api_key: str | None = None,
    region: str | None = None,
    **kwargs,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `list[string]` | ✓ | The documents to put in order — texts, or records with a 'text' field. |
| `options.query` | `string` | ✓ | The question the documents should answer. |
| `options.top_n` | `integer` |  | How many to return, best first (default: all). |

```python
result = RerankQwen("<model>")(
    input="…",
    options={"query": …},
)
```
<!-- /g:tool -->
