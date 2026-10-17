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
| `KLING_API_KEY` | Kling | <https://kling.ai/document-api/api/get-started/authentication> |
| `KREA_API_KEY` | Krea | — |
| `META_API_KEY` | Meta (Muse) | <https://dev.meta.ai> |
| `MINIMAX_API_KEY` | MiniMax | <https://platform.minimax.io> |
| `MISTRAL_API_KEY` | Mistral AI | <https://console.mistral.ai/api-keys> |
| `OPENAI_API_KEY` | OpenAI | <https://platform.openai.com/api-keys> |
| `PERPLEXITY_API_KEY` | Perplexity | <https://www.perplexity.ai/settings/api> |
| `DASHSCOPE_API_KEY` | Qwen (DashScope) | <https://dashscope.aliyuncs.com> |
| `RECRAFT_API_TOKEN` | Recraft | — |
| `STEPFUN_API_KEY` | StepFun | <https://platform.stepfun.ai> |
| `XAI_API_KEY` | xAI | <https://console.x.ai/> |
| `MIMO_API_KEY` | Xiaomi (MiMo) | <https://platform.xiaomimimo.com> |
| `ZAI_API_KEY` | Z.ai (GLM) | <https://z.ai/manage-apikey/apikey-list> |
| `OPENROUTER_API_KEY` | OpenRouter | <https://openrouter.ai/settings/keys> |
<!-- /g:env-providers -->

Private servers are the exception to "required": the `private/` prefix works with
no key at all (see [Private models](../models/private-models.md)).

| Variable | Provider | Meaning |
|---|---|---|
| `PRIVATE_BASE_URL` | Private (vLLM/Ollama/…) | Server URL when not `http://localhost:8000` — e.g. a remote GPU box |
| `PRIVATE_API_KEY` | Private (vLLM/Ollama/…) | Only if the server was started with `--api-key`; otherwise unset |

### How the `Model` factory resolves keys

<!-- g:env-resolve -->
```python
Model("claude-fable-5")           # reads ANTHROPIC_API_KEY
Model("flux-2-flex")              # reads BFL_API_KEY
Model("deepseek-chat")            # reads DEEPSEEK_API_KEY
Model("gemini-3-flash-preview")   # reads GOOGLE_AI_API_KEY
Model("kimi-k2.6")                # reads MOONSHOT_API_KEY
Model("kling-3.0")                # reads KLING_API_KEY
Model("krea-2-large")             # reads KREA_API_KEY
Model("muse-spark-1.1")           # reads META_API_KEY
Model("MiniMax-M2")               # reads MINIMAX_API_KEY
Model("codestral-2508")           # reads MISTRAL_API_KEY
Model("gpt-4.1")                  # reads OPENAI_API_KEY
Model("sonar")                    # reads PERPLEXITY_API_KEY
Model("qwen-flash")               # reads DASHSCOPE_API_KEY
Model("recraft-vectorize")        # reads RECRAFT_API_TOKEN
Model("step-3.5-flash")           # reads STEPFUN_API_KEY
Model("grok-3")                   # reads XAI_API_KEY
Model("mimo-v2.6-flash")          # reads MIMO_API_KEY
Model("glm-4.5")                  # reads ZAI_API_KEY
Model("openrouter/vendor/model")  # reads OPENROUTER_API_KEY

Model("gpt-5.5", api_key="sk-…")    # explicit — the environment is not read
```
<!-- /g:env-resolve -->

---

## Built-in tools

<!-- g:env-tools -->
| Variable | Read by | Same key as a model provider |
|---|---|---|
| `BRAVE_SEARCH_API_KEY` | `searchBrave` | — |
| `COHERE_API_KEY` | `EmbeddingCohere`, `RerankCohere` | — |
| `DASHSCOPE_API_KEY` | `EmbeddingQwen`, `RerankQwen`, `ttsQwen` | yes |
| `EXA_API_KEY` | `fetchExa`, `searchExa` | — |
| `FIRECRAWL_API_KEY` | `fetchFirecrawl`, `searchFirecrawl` | — |
| `GOOGLE_AI_API_KEY` | `EmbeddingGoogle`, `sttGoogle`, `ttsGoogle` | yes |
| `GOOGLE_API_KEY` | `EmbeddingGoogle`, `sttGoogle`, `ttsGoogle` | — |
| `GOOGLE_CLOUD_API_KEY` | `sttGoogle`, `ttsGoogle` | — |
| `OPENAI_API_KEY` | `EmbeddingOpenAI`, `decisionOpenAI`, `sttOpenAI`, `ttsOpenAI` | yes |
| `PARALLEL_API_KEY` | `fetchParallel`, `searchParallel` | — |
| `PERPLEXITY_API_KEY` | `searchPerplexity` | yes |
| `TAVILY_API_KEY` | `fetchTavily`, `searchTavily` | — |
| `TYPESAFE_API_KEY` | `decisionTypeSafe` | — |
| `VOYAGE_API_KEY` | `EmbeddingVoyage`, `RerankVoyage` | — |
| `XAI_API_KEY` | `ttsXAI` | yes |
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
export KLING_API_KEY="…"
export KREA_API_KEY="…"
export META_API_KEY="…"
export MINIMAX_API_KEY="…"
export MISTRAL_API_KEY="…"
export OPENAI_API_KEY="…"
export PERPLEXITY_API_KEY="…"
export DASHSCOPE_API_KEY="…"
export RECRAFT_API_TOKEN="…"
export STEPFUN_API_KEY="…"
export XAI_API_KEY="…"
export MIMO_API_KEY="…"
export ZAI_API_KEY="…"
export OPENROUTER_API_KEY="…"
export BRAVE_SEARCH_API_KEY="…"
export COHERE_API_KEY="…"
export EXA_API_KEY="…"
export FIRECRAWL_API_KEY="…"
export GOOGLE_API_KEY="…"
export GOOGLE_CLOUD_API_KEY="…"
export PARALLEL_API_KEY="…"
export TAVILY_API_KEY="…"
export TYPESAFE_API_KEY="…"
export VOYAGE_API_KEY="…"
```
<!-- /g:exports -->

### Research agent (search + fetch)

```bash
export ANTHROPIC_API_KEY="sk-ant-…"    # orchestrator
export BRAVE_SEARCH_API_KEY="BSA-…"    # or PERPLEXITY_API_KEY
# fetch("local") needs no key
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

- [Installation](../start/installation.md) — which packages to install alongside which keys.
- [Model registry](model-registry.md) — which model belongs to which provider.
