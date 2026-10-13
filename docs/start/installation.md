# Installation

```bash
pip install yait-aichain
```

One dependency comes with it, `urllib3`, which most Python environments
already have — and on Python 3.10 `tomli`, since the standard library reads
TOML only from 3.11. Nothing else is installed until you use what needs it — a
library meant to run in a serverless function cannot bring a hundred
megabytes it might not use.

## Extras, when you need them

| You use | Install | Brings |
|---|---|---|
| `Skill.save()` / `Chain.save()` and `load()` | `pip install "yait-aichain[yaml]"` | `pyyaml` |
| the `MCP` utility | `pip install "yait-aichain[mcp]"` | `fastmcp` |
| `fetch("local")` reading a PDF | `pip install "yait-aichain[fetch]"` | `pypdf` |
| transcribing a file over a provider's size limit | `pip install "yait-aichain[audio]"` | `pydub` (and `audioop-lts` on 3.13+) |
| all of them | `pip install "yait-aichain[all]"` | |

`fetch("local")` reads HTML with the standard library; until 4.0 it brought
MarkItDown, and through it `magika` and `onnxruntime` — tens of megabytes.

A vector store's own client — Chroma, Pinecone, Qdrant — is installed only if
you use that backend. Converting files to Markdown, HTML or PDF left the
library in 3.8.0; the cookbooks show it done with MarkItDown, mistletoe and
WeasyPrint directly.

## Keys

Set keys only for the providers and services you use — or none at all: a model
on your own server needs no key and sends nothing outside your network.
→ [Private models](../models/private-models.md)

<!-- g:install-keys -->
| Service | Environment variable | Get a key |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | <https://console.anthropic.com/settings/keys> |
| Black Forest Labs (FLUX) | `BFL_API_KEY` | — |
| DeepSeek | `DEEPSEEK_API_KEY` | <https://platform.deepseek.com/api_keys> |
| Google AI | `GOOGLE_AI_API_KEY` | <https://aistudio.google.com/app/apikey> |
| Kimi (Moonshot AI) | `MOONSHOT_API_KEY` | <https://platform.kimi.ai/> |
| Meta (Muse) | `META_API_KEY` | <https://dev.meta.ai> |
| MiniMax | `MINIMAX_API_KEY` | <https://platform.minimax.io> |
| Mistral AI | `MISTRAL_API_KEY` | <https://console.mistral.ai/api-keys> |
| OpenAI | `OPENAI_API_KEY` | <https://platform.openai.com/api-keys> |
| Perplexity | `PERPLEXITY_API_KEY` | <https://www.perplexity.ai/settings/api> |
| Qwen (DashScope) | `DASHSCOPE_API_KEY` | <https://dashscope.aliyuncs.com> |
| Recraft | `RECRAFT_API_TOKEN` | — |
| StepFun | `STEPFUN_API_KEY` | <https://platform.stepfun.ai> |
| xAI | `XAI_API_KEY` | <https://console.x.ai/> |
| Xiaomi (MiMo) | `MIMO_API_KEY` | <https://platform.xiaomimimo.com> |
| Z.ai (GLM) | `ZAI_API_KEY` | <https://z.ai/manage-apikey/apikey-list> |
| Brave Search | `BRAVE_SEARCH_API_KEY` | <https://brave.com/search/api> |
| Cohere | `COHERE_API_KEY` | <https://dashboard.cohere.com> |
| TypeSafe | `TYPESAFE_API_KEY` | <https://console.typesafe.ai> |
| Exa | `EXA_API_KEY` | <https://dashboard.exa.ai/api-keys> |
| Firecrawl | `FIRECRAWL_API_KEY` | <https://www.firecrawl.dev/app/api-keys> |
| Tavily | `TAVILY_API_KEY` | <https://app.tavily.com> |
| Parallel | `PARALLEL_API_KEY` | <https://platform.parallel.ai> |
| Voyage | `VOYAGE_API_KEY` | <https://dash.voyageai.com> |
<!-- /g:install-keys -->

The library does not read `.env` files by itself:

```bash
set -a && source .env && set +a
```

Keys are never written by `save()`; a saved Skill or Chain names its model, and
the key is read from the environment again when it is loaded. Every variable,
and which tool reads which: [Environment variables](../reference/environment-variables.md).

## Check it works

```bash
python -c "from yait_aichain import Model, Skill; print('ok')"
python examples/01_skill.py          # one real call; needs ANTHROPIC_API_KEY
```

Python 3.10 or later; the library is tested on 3.10 through 3.14.
