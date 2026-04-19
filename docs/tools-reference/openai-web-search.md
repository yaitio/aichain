# `openai_web_search` — `OpenAIWebSearchTool`

Live web search **synthesized by an OpenAI model in a single call**. Unlike raw-results tools (Brave, Perplexity, SerpAPI), this one returns a prose answer with inline citations — the model decides which sources to consult and weaves them into a single response.

```python
from tools import OpenAIWebSearchTool

tool   = OpenAIWebSearchTool()
result = tool(query="nuclear fusion breakthroughs 2025")

print(result.output)
```

---

## Requirements

| | |
|---|---|
| Env var | `OPENAI_API_KEY` |
| Endpoint | `POST https://api.openai.com/v1/responses` |
| Built-in tool | `web_search_preview` |

Supported models: `gpt-4o` (default), `gpt-4.1`, `gpt-4.1-mini`, `o3`, `o4-mini`. Not supported on `gpt-4.1-nano`. GPT-5 models intentionally avoided by default.

---

## Constructor

```python
OpenAIWebSearchTool(api_key: str | None = None, model: str = "gpt-4o")
```

---

## Parameters

| Name | Type | Required | Default | Notes |
|---|---|---|---|---|
| `query` | `string` | ✓ | — | Natural-language question. |
| `search_context_size` | `string` | | `medium` | `low` / `medium` / `high`. |
| `allowed_domains` | `list[string]` | | — | Restrict to these domains (subdomains included). |
| `country` | `string` | | — | ISO 3166-1 alpha-2 for location bias. |
| `timezone` | `string` | | — | IANA timezone (e.g. `America/New_York`). |

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
text = tool.run(query="Latest EU AI Act amendments 2025")
```

### Deep search + domain restriction

```python
text = tool.run(
    query               = "GPT-5 benchmark results",
    search_context_size = "high",
    allowed_domains     = ["openai.com", "arxiv.org"],
)
```

### Location bias

```python
text = tool.run(
    query    = "best hospitals near downtown",
    country  = "US",
    timezone = "America/Chicago",
)
```

### Using a different model

```python
tool = OpenAIWebSearchTool(model="gpt-4.1")
tool.run(query="latest AI safety research 2025")
```

---

## When to prefer this over the raw search tools

| Goal | Tool |
|---|---|
| "Give me a synthesized, cited answer" | **openai_web_search** |
| "Give me a list of URLs I can fetch" | `brave_search` / `serp_api_search` |
| "Give me snippets I can reason over without another fetch" | `perplexity_search` |

---

## Notes

- This tool bills OpenAI Responses-API tokens *plus* the built-in web-search fee.
- The `search_context_size` knob is the biggest cost / quality lever.

---

## See also

- [`perplexity_search`](perplexity-search.md), [`brave_search`](brave-search.md), [`serp_api_search`](serp-api.md)
