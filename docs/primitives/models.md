# Model

A **Model** is the provider gateway. You ask for a model by name; the library
resolves the provider, attaches a configured HTTP client, and gives you back one
object that every other primitive (Skill, Chain, Agent, embeddings) can use.
Swapping providers is a one-word change.

---

## Quick start

```python
import os
from yait_aichain.models import Model

m = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY"))
```

The same `Model` drops into any primitive — change the name to change provider,
nothing else:

```python
Model("gpt-4o")              # OpenAI
Model("gemini-2.5-flash")    # Google
Model("grok-3")              # xAI
Model("qwen-max")            # Qwen
```

▶ One prompt, three providers: [`examples/02_skill_models.py`](../../examples/02_skill_models.py) ·
Full catalogue: [model registry](../reference/model-registry.md) · Deep dive ↓

---

## Common gotchas

- **The name picks the provider.** `Model("gpt-4o")` is OpenAI because `gpt-`
  maps to OpenAI. There are *no* per-provider classes — one `Model`, resolved
  from data.
- **The API key comes from the environment** by default — each provider has its
  own variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …). Pass `api_key=` to
  override. Missing both → `ValueError` at construction.
- **A name that matches no prefix raises.** For a custom or fine-tuned name, give
  an explicit `provider/` prefix: `Model("openai/my-ft-name")`.
- **The registry is reference data, not a gatekeeper.** Asking a text model for
  an image won't be blocked locally — it fails when the provider rejects it. Use
  the registry to *discover* what works, not to validate.

---

## Reference

### How a model is resolved

`Model("name")` does three lookups, all data-driven:

1. **Provider** — from a well-known name prefix (table below), or an explicit
   `provider/` prefix. Exposed afterwards as `model._provider`.
2. **Settings & price** — from that provider's data file
   `yait_aichain/models/providers/<provider>.toml` (defaults, endpoints,
   capabilities, prices).
3. **Wire format** — handled by the provider's *family client*. Five clients
   cover all eight providers; you never see them directly.

There is a single `Model` class. Everything provider-specific (URL, auth header,
request/response shape, the reasoning knob) lives in data + the family client —
which is exactly why swapping models is one line.

| Prefix / pattern | Provider |
|---|---|
| `claude-` | Anthropic |
| `gemini-` | Google AI |
| `grok-` | xAI |
| `sonar`, `r1-1776` | Perplexity |
| `kimi-` | Kimi (Moonshot) |
| `deepseek-` | DeepSeek |
| `qwen`, `qwq`, `wanx`, `wan<digit>` | Qwen (DashScope) |
| `gpt-`, `dall-e-`, `chatgpt-image-`, `text-embedding-`, `whisper-`, `tts-`, `o<digit>` | OpenAI |

### Constructor

```python
Model(name, options=None, client_options=None, api_key=None)
```

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Model name; the prefix selects the provider. `provider/name` forces a provider for an unlisted name. |
| `options` | `dict` \| `None` | Generation parameters (see below). Missing keys fall back to the provider's data defaults. |
| `client_options` | `dict` \| `None` | HTTP client tuning (see below). |
| `api_key` | `str` \| `None` | Overrides the provider's environment variable. |

### `options` — generation parameters

One vocabulary for every provider; a provider silently ignores a key it doesn't
use.

| Key | Type | Notes |
|---|---|---|
| `temperature` | `float` | Sampling temperature. |
| `max_tokens` | `int` | Maximum output tokens. |
| `top_p` | `float` | Nucleus sampling mass. |
| `top_k` | `int` | Top-K sampling (provider-dependent). |
| `cache_control` | `bool` | Enable provider-level prompt caching. |
| `reasoning` | `None`\|`"low"`\|`"medium"`\|`"high"` | Universal reasoning depth (below). |

### Universal reasoning

`reasoning` is one knob mapped to each provider's native thinking controls — so
you raise "reasoning depth" the same way everywhere:

| Provider | Translated to |
|---|---|
| Anthropic | `{"type": "enabled", "budget_tokens": N}` — low=4k, medium=10k, high=20k. Temperature forced to 1.0 when active. |
| Google AI | `thinkingConfig.thinkingBudget` — low=2 048, medium=8 192, high=24 576. |
| OpenAI | `reasoning_effort` on o-series / GPT-5.x reasoners. Plain GPT models ignore it. |
| xAI | `reasoning_effort` for `grok-3-mini` / `-fast`. Other grok models think natively and ignore it. |
| Kimi | `{"thinking": {"type": "enabled"}}` — all levels enable it (no budget knob). Temperature forced to 1.0. |
| DeepSeek | Name switch — `"high"` routes to `deepseek-reasoner`; `"low"`/`"medium"` stay on `deepseek-chat`. |
| Perplexity | Not used. |

```python
Model("claude-opus-4-8", options={"reasoning": "high"})    # 20k thinking tokens
Model("gemini-2.5-pro",  options={"reasoning": "high"})    # Gemini thinking budget
Model("deepseek-chat",   options={"reasoning": "high"})    # routes to deepseek-reasoner
```

### `client_options` — HTTP / transport

| Key | Type | Notes |
|---|---|---|
| `url` | `str` | Override the base URL (proxies, gateways, Azure-style endpoints). |
| `timeout` | `urllib3.Timeout` | Custom connect / read timeout. |
| `retries` | `urllib3.Retry` | Transport-level retry policy. |
| `proxy` | `dict` | `{"url": "http://proxy:3128", "username": …, "password": …}`. Or set `HTTPS_PROXY` / `HTTP_PROXY` in the environment — that applies to tool traffic too, not just model calls. |
| `region` | `str` | **Qwen only** — DashScope region (`ap` default, `us`, `cn`, `hk`); also via `DASHSCOPE_REGION`. |

```python
Model("gpt-4o", client_options={"proxy": {"url": "http://corp-proxy:3128"}})
Model("qwen-max", client_options={"region": "us"})
```

### The registry — discovering models

The registry is **reference data**. Query it to discover what the library ships
and is tested with (10 providers, 77 models):

```python
from yait_aichain.models import registry

registry.models(task="text-to-image")
# ['chatgpt-image-latest', 'gemini-3.1-flash-image', 'gpt-image-2',
#  'gpt-image-1.5', 'grok-imagine-image-pro', 'wan2.2-t2i-flash', ...]

registry.providers(task="text-to-image")   # ['openai', 'google', 'xai', 'qwen']
registry.tasks("gpt-4o")                    # ['image-to-text', 'text-to-text']
registry.is_supported("gpt-image-2", "text-to-image")   # True
```

Tasks: `text-to-text`, `text-to-image`, `image-to-text`, `image-to-image`. To
diff the registry against a provider's live roster (new/removed models), call
`registry.refresh("openai")`.

### Image editing (image-to-image)

Editing an image is just **an input image part plus an image output** — the same
`Skill` you use for generation, with the model named accordingly. The provider is
a one-word swap.

Two flavours, depending on the model:

- **Instruction edit — preserves the subject** (OpenAI `gpt-image-*`, Google
  Gemini image, xAI `grok-imagine-*`, Qwen `qwen-image-edit`, BFL `flux-kontext-*`):
  follows the prompt while keeping the original object — e.g. "place *this*
  product on a marble counter" keeps your product. This is what you want for
  product photography and scene composition.
- **Whole-image variation — not subject-preserving** (Recraft `imageToImage`):
  a `strength`-controlled img2img transform of the *entire* image; great for
  restyling a picture, but it will not lift your object into a new scene
  (low `strength` stays near the original, high `strength` invents a new one).

```python
from yait_aichain import Model, Skill

Skill(
    model  = Model("gpt-image-1.5"),   # → gemini-3.1-flash-image / grok-imagine-image / qwen-image-edit
    input  = {"messages": [{"role": "user", "parts": [
        {"type": "image", "source": {"kind": "file", "path": "product.png"}},
        "Place this product on a marble kitchen counter, soft morning light",
    ]}]},
    output = {"modalities": ["image"], "format": {"type": "image"}},
).run()      # → {"base64": ..., "mime_type": ..., "url": ..., "revised_prompt": ...}
```

A media source of `{"kind": "file", "path": "..."}` is read and base64-encoded for
you (MIME inferred); `{"kind": "base64", "data": ..., "mime": ...}` and
`{"kind": "url", "url": ...}` work too. Discover edit-capable models with
`registry.models(task="image-to-image")`.

### Provider reference

All providers expose the same interface; the table is for discovery. Full,
current lists live in `models/providers/*.toml` and the
[model registry](../reference/model-registry.md).

| Provider | Env var | Representative models |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-5.5`, `gpt-5.4`, `gpt-4o`, `chatgpt-image-latest`, `gpt-image-2` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-fable-5`, `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` |
| Google AI | `GOOGLE_AI_API_KEY` | `gemini-3.1-pro-preview`, `gemini-2.5-pro`, `gemini-2.5-flash` |
| xAI | `XAI_API_KEY` | `grok-4-0709`, `grok-4-fast-reasoning`, `grok-3`, `grok-imagine-image-pro` |
| Perplexity | `PERPLEXITY_API_KEY` | `sonar-pro`, `sonar`, `sonar-reasoning-pro`, `sonar-deep-research` |
| Kimi | `MOONSHOT_API_KEY` | `kimi-k2.7-code`, `kimi-k2.6`, `kimi-k2.5`, `kimi-k2-thinking` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat`, `deepseek-reasoner` |
| Qwen | `DASHSCOPE_API_KEY` | `qwen3-max`, `qwen-max`, `qwen-vl-max`, `wan2.2-t2i-flash`, `qwen-image-edit` |
| Recraft | `RECRAFT_API_TOKEN` | `recraftv4_1`, `recraftv3`, `recraftv3_vector` *(image only)* |
| BFL (FLUX) | `BFL_API_KEY` | `flux-2-pro`, `flux-pro-1.1`, `flux-kontext-pro`, `flux-kontext-max` *(image only)* |

### What a Model does (and doesn't)

A Model does exactly two things:

- **`to_request(messages, output) → (path, body)`** — universal template →
  provider-native request.
- **`from_response(response, output) → str | dict`** — raw response → clean value.

It knows nothing about prompts, variables, retries, or pipelines — those belong
to [Skill](skills.md) and [Chain](chain.md). That isolation is the whole point.

---

## See also

- [Skill](skills.md) — bind a Model to a prompt and run it.
- [Chain](chain.md) — sequence model calls with automatic variable flow.
- [Agent](../agents/overview.md) — let a model plan and call tools.
- [Model registry](../reference/model-registry.md) — the full model catalogue.
