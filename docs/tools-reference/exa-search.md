<!-- g:tool -->
# `searchExa`

Web search powered by the Exa Search API — results with the passages that answer the query.

| | |
|---|---|
| Import | `from yait_aichain.tools import searchExa` |
| Risk class | `write` |
| Key | `EXA_API_KEY` |

```python
searchExa(
    api_key: str | None = None,
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | The search query, in natural language. |
| `options.max_results` | `integer` |  | Number of results to return (1–100, default 10). |
| `options.recency` | `string` |  | Only pages published within this window. One of `hour`, `day`, `week`, `month`, `year`. |
| `options.country` | `string` |  | Two-letter ISO country code of the user, e.g. 'US'. |
| `options.domains` | `list[string]` |  | Restrict results to these domains (or domain/path prefixes). |
| `options.exclude_domains` | `list[string]` |  | Leave out results from these domains. |
| `options.category` | `string` |  | Focus on one kind of page. One of `company`, `publication`, `news`, `personal site`, `financial report`, `people`. |
| `options.search_type` | `string` |  | 'auto' (default) balances quality and speed; 'instant'/'fast' trade quality for latency; the 'deep' modes search iteratively and cost more. One of `instant`, `fast`, `auto`, `deep-lite`, `deep`, `deep-reasoning`. |
| `options.start_date` | `string` |  | Only pages published after this ISO 8601 date. |
| `options.end_date` | `string` |  | Only pages published before this ISO 8601 date. |
| `options.text_chars` | `integer` |  | Also return each page's text, cut to this many characters. Off by default — highlights usually carry the answer. |

```python
result = searchExa()(
    input="…",
    options={"max_results": …},
)
```
<!-- /g:tool -->
