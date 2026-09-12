<!-- g:tool -->
# `searchOpenAI` — `OpenAIWebSearchTool`

Web search powered by OpenAI's built-in `web_search_preview` tool.

| | |
|---|---|
| Import | `from yait_aichain.tools import OpenAIWebSearchTool` |
| Risk class | `write` |
| Also exported as | `searchOpenAI` |
| Key | `OPENAI_API_KEY` |

```python
OpenAIWebSearchTool(
    api_key: str | None = None,
    model: str = 'gpt-4o',
)
```

| Name | Type | Required | Description |
|---|---|---|---|
| `input` | `string` | ✓ | The question or search query to answer using live web search. |
| `options.context_size` | `string` |  | How much web context to gather before answering. 'low' is fast and cheap; 'high' is thorough but uses more tokens. Defaults to 'medium'. One of `low`, `medium`, `high`. |
| `options.domains` | `list[string]` |  | Restrict the search to these domains only (e.g. ["openai.com", "arxiv.org"]). Subdomains are automatically included. |
| `options.country` | `string` |  | Two-letter ISO 3166-1 country code to bias results (e.g. 'US', 'GB'). Part of the approximate user location. |
| `options.timezone` | `string` |  | IANA timezone string to help with time-sensitive queries (e.g. 'America/New_York'). Part of the approximate user location. |
| `options.model` | `string` |  | Override the OpenAI model for this call. Defaults to the model passed at construction time. |

```python
result = OpenAIWebSearchTool()(
    input="…",
    options={"context_size": …},
)
```
<!-- /g:tool -->

Live web search **synthesized by an OpenAI model in one call**. Unlike the
raw-results tools (Brave, Perplexity, SerpAPI), it returns a prose answer with
inline citations — the model decides which sources to consult.

```python
from yait_aichain.tools import OpenAIWebSearchTool

tool   = OpenAIWebSearchTool()
result = tool(input="nuclear fusion breakthroughs 2025")

print(result.output)
```

---

## Output shape

Two-section plain text:

```
[Answer]
The latest fusion energy developments include NIF achieving… [1]
Commonwealth Fusion expects a commercial reactor by 2030… [2]

[Sources]
[1] NIF Achieves Ignition — Science Daily
    https://www.sciencedaily.com/releases/…
[2] Commonwealth Fusion Systems — Company Blog
    https://cfs.energy/news/…
```

If the model returns no answer, the output is the literal string `(no answer returned)`.

---

## Usage

### Basic

```python
tool = OpenAIWebSearchTool()
text = tool.run(input="Latest EU AI Act amendments 2025")
```

### Deep search, restricted to domains

```python
tool = OpenAIWebSearchTool()
text = tool.run(input="GPT-5 benchmark results",
                options={"context_size": "high",
                         "domains": ["openai.com", "arxiv.org"]})
```

### Location bias

```python
tool = OpenAIWebSearchTool()
text = tool.run(input="best hospitals near downtown",
                options={"country": "US", "timezone": "America/Chicago"})
```

### A different model

```python
tool = OpenAIWebSearchTool(model="gpt-4.1")          # for every call
tool.run(input="latest AI safety research 2025",
         options={"model": "gpt-4o"})                  # or for one
```

---

## When to prefer this over the raw search tools

| Goal | Tool |
|---|---|
| "Give me a synthesized, cited answer" | **`searchOpenAI`** |
| "Give me a list of URLs I can fetch" | `searchBrave` / `searchSerp` |
| "Give me snippets I can reason over without another fetch" | `searchPerplexity` |

---

## Notes

- This tool bills Responses-API tokens *plus* the built-in web-search fee.
- `context_size` is the biggest cost / quality lever.

---

## See also

- [`searchPerplexity`](perplexity-search.md), [`searchBrave`](brave-search.md), [`searchSerp`](serp-api.md)
