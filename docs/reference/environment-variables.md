# Environment variables

All keys the library reads from the environment, grouped by provider and tool. None are required at import time — each is resolved at the first API call that needs it.

API keys are **never** written to YAML when saving Skills or Chains; they are always resolved from the environment at load time.

---

## LLM providers

These keys are required to use the corresponding `Model(...)` prefix or direct subclass.

| Variable | Provider | Where to get it |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI | <https://platform.openai.com/api-keys> |
| `ANTHROPIC_API_KEY` | Anthropic | <https://console.anthropic.com/settings/keys> |
| `GOOGLE_AI_API_KEY` | Google AI (Gemini) | <https://aistudio.google.com/app/apikey> |
| `XAI_API_KEY` | xAI (Grok) | <https://console.x.ai/> |
| `PERPLEXITY_API_KEY` | Perplexity | <https://www.perplexity.ai/settings/api> |
| `MOONSHOT_API_KEY` | Kimi (Moonshot AI) | <https://platform.kimi.ai/> |
| `DEEPSEEK_API_KEY` | DeepSeek | <https://platform.deepseek.com/api_keys> |

### How the `Model` factory resolves keys

```python
Model("gpt-4o")                     # reads OPENAI_API_KEY
Model("claude-opus-4-6")            # reads ANTHROPIC_API_KEY
Model("gemini-2.5-pro")             # reads GOOGLE_AI_API_KEY
Model("grok-3")                     # reads XAI_API_KEY
Model("sonar-pro")                  # reads PERPLEXITY_API_KEY
Model("kimi-k2.5")                  # reads MOONSHOT_API_KEY
Model("deepseek-chat")              # reads DEEPSEEK_API_KEY

Model("gpt-4o", api_key="sk-…")     # explicit override — env var ignored
```

---

## Built-in tools

| Variable | Tool | Notes |
|---|---|---|
| `PERPLEXITY_API_KEY` | `PerplexitySearchTool` | Same key as the Perplexity LLM provider. |
| `BRAVE_SEARCH_API_KEY` | `BraveSearchTool` | Brave's subscription token. |
| `SERPAPI_API_KEY` | `SerpApiTool` | Note: `SERPAPI_` prefix (not `SERP_API_`). |
| `OPENAI_API_KEY` | `OpenAIWebSearchTool` | Same key as the OpenAI LLM provider. |
| `DEEPL_API_KEY` | `DeepLTranslateTool`, `DeepLRephraseTool` | Keys ending with `:fx` use the free endpoint automatically. |
| `LATE_API_KEY` | `LateAccountsTool`, `LatePublishTool` | <https://getlate.dev/dashboard/api-keys> |

Tools with no API requirements: `MarkItDownTool`, `MistletoeTool`, `WeasyprintTool`, `SectionContextTool`.

---

## Minimal setup by use case

### Chat / text generation only

```bash
# Pick whichever providers you use
export OPENAI_API_KEY="sk-…"
export ANTHROPIC_API_KEY="sk-ant-…"
export GOOGLE_AI_API_KEY="AIza…"
export XAI_API_KEY="xai-…"
export PERPLEXITY_API_KEY="pplx-…"
export MOONSHOT_API_KEY="sk-…"      # Kimi
export DEEPSEEK_API_KEY="sk-…"      # DeepSeek
```

### Research agent (search + fetch)

```bash
export ANTHROPIC_API_KEY="sk-ant-…"    # orchestrator
export BRAVE_SEARCH_API_KEY="BSA-…"    # or PERPLEXITY_API_KEY / SERPAPI_API_KEY
# MarkItDownTool needs no key
```

### Social publishing agent

```bash
export ANTHROPIC_API_KEY="sk-ant-…"
export LATE_API_KEY="late-…"
export OPENAI_API_KEY="sk-…"           # if also generating images
```

### Document pipeline (translate → PDF)

```bash
export OPENAI_API_KEY="sk-…"           # or any text model key
export DEEPL_API_KEY="…:fx"            # free tier
# MistletoeTool and WeasyprintTool need no key
```

---

## `.env` file (recommended for development)

The library does **not** load `.env` files automatically. Use `python-dotenv` or your shell:

```bash
# .env
OPENAI_API_KEY=sk-…
ANTHROPIC_API_KEY=sk-ant-…
DEEPL_API_KEY=…:fx
LATE_API_KEY=late-…
```

```python
from dotenv import load_dotenv
load_dotenv()
```

Or in the shell:

```bash
set -a && source .env && set +a
```

---

## Notes

- All tools raise `ValueError` at **construction time** if a required key is missing — you get a clear error before any API call is made.
- Pass `api_key=` explicitly to any `Model(...)` call or tool constructor to override the env var for that instance.
- When saving a Chain or Skill to YAML, keys are **never** included in the file.

---

## See also

- [Installation](../getting-started/installation.md) — which packages to install alongside which keys.
- [Model registry](model-registry.md) — which model belongs to which provider.
