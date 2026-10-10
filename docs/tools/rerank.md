<!-- g:tool -->
# Reranking: documents → in order

## `rerankCohere` — `RerankCohere`

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

## `rerankVoyage` — `RerankVoyage`

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

## `rerankQwen` — `RerankQwen`

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

## One tool, three providers

```python
from yait_aichain.tools import rerank

top = rerank("cohere", model="rerank-v3.5").run(docs, {"query": "how does caching work?", "top_n": 5})
top[0]      # {"index", "text", "score", …} — the best first; a record's own fields are kept
```

`rerank(service, model=…)` makes the tool — `cohere`, `voyage` or `qwen`. It
takes documents as texts or as records with a `text` field — what a vector
store's search returns — and puts them in order of how well they answer
`query`. Thirty candidates in, the five that answer the question out: an agent
can use it after a search of its own. `Reranker("rerank-v3.5")` is the factory
it replaced and still works.
