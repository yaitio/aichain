# Environment variables

All keys the library reads from the environment, grouped by provider and tool. None are required at import time — each is resolved at the first API call that needs it.

API keys are **never** written to YAML when saving Skills or Chains; they are always resolved from the environment at load time.

---

## LLM providers

These keys are required to use the corresponding `Model(...)` prefix or direct subclass.

<!-- g:env-providers -->
| Variable | Provider | Where to get it |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic | <https://console.anthropic.com/settings/keys> |
| `BFL_API_KEY` | Black Forest Labs (FLUX) | — |
| `DEEPSEEK_API_KEY` | DeepSeek | <https://platform.deepseek.com/api_keys> |
| `GOOGLE_AI_API_KEY` | Google AI | <https://aistudio.google.com/app/apikey> |
| `MOONSHOT_API_KEY` | Kimi (Moonshot AI) | <https://platform.kimi.ai/> |
| `OPENAI_API_KEY` | OpenAI | <https://platform.openai.com/api-keys> |
| `PERPLEXITY_API_KEY` | Perplexity | <https://www.perplexity.ai/settings/api> |
| `DASHSCOPE_API_KEY` | Qwen (DashScope) | <https://dashscope.aliyuncs.com> |
| `RECRAFT_API_TOKEN` | Recraft | — |
| `REVE_API_KEY` | Reve | — |
| `XAI_API_KEY` | xAI | <https://console.x.ai/> |
<!-- /g:env-providers -->

Private servers are the exception to "required": the `private/` prefix works with
no key at all (see [Private models](../getting-started/private-models.md)).

| Variable | Provider | Meaning |
|---|---|---|
| `PRIVATE_BASE_URL` | Private (vLLM/Ollama/…) | Server URL when not `http://localhost:8000` — e.g. a remote GPU box |
| `PRIVATE_API_KEY` | Private (vLLM/Ollama/…) | Only if the server was started with `--api-key`; otherwise unset |

### How the `Model` factory resolves keys

<!-- g:env-resolve -->
```python
Model("claude-fable-5")        # reads ANTHROPIC_API_KEY
Model("flux-2-pro")            # reads BFL_API_KEY
Model("deepseek-chat")         # reads DEEPSEEK_API_KEY
Model("gemini-2.5-flash")      # reads GOOGLE_AI_API_KEY
Model("kimi-k2-0905-preview")  # reads MOONSHOT_API_KEY
Model("gpt-4o")                # reads OPENAI_API_KEY
Model("sonar")                 # reads PERPLEXITY_API_KEY
Model("QwQ-32B")               # reads DASHSCOPE_API_KEY
Model("recraft-vectorize")     # reads RECRAFT_API_TOKEN
Model("reve-image")            # reads REVE_API_KEY
Model("grok-3")                # reads XAI_API_KEY

Model("gpt-5.5", api_key="sk-…")    # explicit — the environment is not read
```
<!-- /g:env-resolve -->

---

## Built-in tools

<!-- g:env-tools -->
| Variable | Read by | Same key as a model provider |
|---|---|---|
| `BRAVE_SEARCH_API_KEY` | `searchBrave` | — |
| `COHERE_API_KEY` | `CohereEmbedder`, `EmbeddingCohere`, `RerankCohere` | — |
| `DASHSCOPE_API_KEY` | `EmbeddingQwen`, `RerankQwen`, `sttQwen`, `ttsQwen` | yes |
| `GOOGLE_AI_API_KEY` | `EmbeddingGoogle`, `GoogleEmbedder`, `sttGoogle`, `ttsGoogle` | yes |
| `GOOGLE_API_KEY` | `EmbeddingGoogle`, `GoogleEmbedder`, `sttGoogle`, `ttsGoogle` | — |
| `OPENAI_API_KEY` | `EmbeddingOpenAI`, `OpenAIEmbedder`, `searchOpenAI`, `sttOpenAI`, `ttsOpenAI` | yes |
| `PERPLEXITY_API_KEY` | `searchPerplexity` | yes |
| `SERPAPI_API_KEY` | `searchSerp` | — |
| `VOYAGE_API_KEY` | `EmbeddingVoyage`, `RerankVoyage`, `VoyageEmbedder` | — |
| `XAI_API_KEY` | `sttXAI`, `ttsXAI` | yes |
<!-- /g:env-tools -->

A tool not listed reads no key.

---

## Minimal setup by use case

### Every key the library reads

Set only the ones you use.

<!-- g:exports -->
```bash
export ANTHROPIC_API_KEY="…"
export BFL_API_KEY="…"
export DEEPSEEK_API_KEY="…"
export GOOGLE_AI_API_KEY="…"
export MOONSHOT_API_KEY="…"
export OPENAI_API_KEY="…"
export PERPLEXITY_API_KEY="…"
export DASHSCOPE_API_KEY="…"
export RECRAFT_API_TOKEN="…"
export REVE_API_KEY="…"
export XAI_API_KEY="…"
export BRAVE_SEARCH_API_KEY="…"
export COHERE_API_KEY="…"
export GOOGLE_API_KEY="…"
export SERPAPI_API_KEY="…"
export VOYAGE_API_KEY="…"
```
<!-- /g:exports -->

### Research agent (search + fetch)

```bash
export ANTHROPIC_API_KEY="sk-ant-…"    # orchestrator
export BRAVE_SEARCH_API_KEY="BSA-…"    # or PERPLEXITY_API_KEY / SERPAPI_API_KEY
# MarkItDownTool needs no key
```

---

## `.env` file (recommended for development)

The library does **not** load `.env` files automatically. Use `python-dotenv` or your shell:

```bash
# .env
OPENAI_API_KEY=sk-…
ANTHROPIC_API_KEY=sk-ant-…
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
