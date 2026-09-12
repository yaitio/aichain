# Model registry

A complete, machine-queryable inventory of every model officially supported by the library, organised by provider and task.

The registry is **reference data only** — the `Model` factory accepts any valid model name regardless of whether it appears here. Use the registry for discovery, documentation, and light validation in application code.

```python
from yait_aichain.models import registry

registry.models(provider="anthropic")
registry.providers(task="text-to-image")
registry.tasks("gpt-4o")
registry.is_supported("gpt-4o", "image-to-text")
registry.accepts("openai")                   # the options this provider has a control for
```

---

<!-- g:registry-models -->
## Tasks

| Task | What it is |
|---|---|
| `text-to-text` | Text prompt → text response. Chat, reasoning, code. |
| `text-to-image` | Text prompt → image. |
| `image-to-text` | Image (+ optional text) → text response. |
| `image-to-image` | Image (+ instruction) → edited image. |

## Providers

`anthropic`, `bfl`, `deepseek`, `google`, `kimi`, `openai`, `perplexity`, `qwen`, `recraft`, `reve`, `xai`.

The `private` provider is registry-less by design: its catalogue belongs to the server you run (vLLM, Ollama, LM Studio, …), so any name it serves works and none is listed here. See [Private models](../getting-started/private-models.md).

| Provider | `text-to-text` | `text-to-image` | `image-to-text` | `image-to-image` |
|---|---|---|---|---|
| Anthropic | 8 | — | 8 | — |
| Black Forest Labs (FLUX) | — | 6 | — | 2 |
| DeepSeek | 2 | — | — | — |
| Google AI | 6 | 3 | 6 | 3 |
| Kimi (Moonshot AI) | 8 | — | 2 | — |
| OpenAI | 11 | 7 | 11 | 7 |
| Perplexity | 5 | — | — | — |
| Qwen (DashScope) | 9 | 2 | 3 | 3 |
| Recraft | — | 8 | — | 3 |
| Reve | — | 1 | — | 1 |
| xAI | 7 | 1 | 7 | 1 |

---

## Anthropic

`provider = "anthropic"` · key `ANTHROPIC_API_KEY`

### `text-to-text`

```
claude-fable-5             # $10 in / $50 out per 1M tokens
claude-haiku-4-5-20251001  # $1 in / $5 out per 1M tokens
claude-opus-4-6            # $5 in / $25 out per 1M tokens
claude-opus-4-7            # $5 in / $25 out per 1M tokens
claude-opus-4-8            # $5 in / $25 out per 1M tokens
claude-opus-5              # $5 in / $25 out per 1M tokens
claude-sonnet-4-6          # $3 in / $15 out per 1M tokens
claude-sonnet-5            # $3 in / $15 out per 1M tokens
```

### `image-to-text`

```
claude-fable-5             # $10 in / $50 out per 1M tokens
claude-haiku-4-5-20251001  # $1 in / $5 out per 1M tokens
claude-opus-4-6            # $5 in / $25 out per 1M tokens
claude-opus-4-7            # $5 in / $25 out per 1M tokens
claude-opus-4-8            # $5 in / $25 out per 1M tokens
claude-opus-5              # $5 in / $25 out per 1M tokens
claude-sonnet-4-6          # $3 in / $15 out per 1M tokens
claude-sonnet-5            # $3 in / $15 out per 1M tokens
```

---

## Black Forest Labs (FLUX)

`provider = "bfl"` · key `BFL_API_KEY`

### `text-to-image`

```
flux-2-pro
flux-dev
flux-kontext-max
flux-kontext-pro
flux-pro-1.1
flux-pro-1.1-ultra
```

### `image-to-image`

```
flux-kontext-max
flux-kontext-pro
```

---

## DeepSeek

`provider = "deepseek"` · key `DEEPSEEK_API_KEY`

### `text-to-text`

```
deepseek-chat      # $0.27 in / $1.1 out per 1M tokens
deepseek-reasoner  # $0.55 in / $2.19 out per 1M tokens
```

---

## Google AI

`provider = "google"` · key `GOOGLE_AI_API_KEY`

### `text-to-text`

```
gemini-2.5-flash               # $0.3 in / $2.5 out per 1M tokens
gemini-2.5-flash-lite          # $0.1 in / $0.4 out per 1M tokens
gemini-2.5-pro                 # $1.25 in / $10 out per 1M tokens
gemini-3-flash-preview         # $0.5 in / $3 out per 1M tokens
gemini-3.1-flash-lite-preview  # $0.25 in / $1.5 out per 1M tokens
gemini-3.1-pro-preview         # $2 in / $12 out per 1M tokens
```

### `text-to-image`

```
gemini-2.5-flash-image
gemini-3-pro-image
gemini-3.1-flash-image
```

### `image-to-text`

```
gemini-2.5-flash               # $0.3 in / $2.5 out per 1M tokens
gemini-2.5-flash-lite          # $0.1 in / $0.4 out per 1M tokens
gemini-2.5-pro                 # $1.25 in / $10 out per 1M tokens
gemini-3-flash-preview         # $0.5 in / $3 out per 1M tokens
gemini-3.1-flash-lite-preview  # $0.25 in / $1.5 out per 1M tokens
gemini-3.1-pro-preview         # $2 in / $12 out per 1M tokens
```

### `image-to-image`

```
gemini-2.5-flash-image
gemini-3-pro-image
gemini-3.1-flash-image
```

---

## Kimi (Moonshot AI)

`provider = "kimi"` · key `MOONSHOT_API_KEY`

### `text-to-text`

```
kimi-k2-0905-preview    # $0.6 in / $2.5 out per 1M tokens
kimi-k2-thinking
kimi-k2-thinking-turbo
kimi-k2-turbo-preview
kimi-k2.5
kimi-k2.6               # $0.95 in / $4 out per 1M tokens
kimi-k2.7-code          # $0.95 in / $4 out per 1M tokens
kimi-k3                 # $3 in / $15 out per 1M tokens
```

### `image-to-text`

```
kimi-k2.5
kimi-k2.6  # $0.95 in / $4 out per 1M tokens
```

---

## OpenAI

`provider = "openai"` · key `OPENAI_API_KEY`

### `text-to-text`

```
gpt-4o         # $2.5 in / $10 out per 1M tokens
gpt-4o-mini    # $0.15 in / $0.6 out per 1M tokens
gpt-5.4        # $2.5 in / $15 out per 1M tokens
gpt-5.4-mini   # $0.75 in / $4.5 out per 1M tokens
gpt-5.4-nano   # $0.2 in / $1.25 out per 1M tokens
gpt-5.4-pro    # $30 in / $180 out per 1M tokens
gpt-5.5        # $5 in / $30 out per 1M tokens
gpt-5.5-pro    # $30 in / $180 out per 1M tokens
gpt-5.6-luna   # $0.2 in / $1.2 out per 1M tokens
gpt-5.6-sol    # $5 in / $30 out per 1M tokens
gpt-5.6-terra  # $2 in / $12 out per 1M tokens
```

### `text-to-image`

```
chatgpt-image-latest    # $5 in / $30 out per 1M tokens
gpt-image-1
gpt-image-1-mini        # $2 in / $8 out per 1M tokens
gpt-image-1.5           # $5 in / $32 out per 1M tokens
gpt-image-2             # $5 in / $30 out per 1M tokens
gpt-image-2.5-flare     # $5 in / $30 out per 1M tokens
gpt-image-2.5-sunburst  # $5 in / $30 out per 1M tokens
```

### `image-to-text`

```
gpt-4o         # $2.5 in / $10 out per 1M tokens
gpt-4o-mini    # $0.15 in / $0.6 out per 1M tokens
gpt-5.4        # $2.5 in / $15 out per 1M tokens
gpt-5.4-mini   # $0.75 in / $4.5 out per 1M tokens
gpt-5.4-nano   # $0.2 in / $1.25 out per 1M tokens
gpt-5.4-pro    # $30 in / $180 out per 1M tokens
gpt-5.5        # $5 in / $30 out per 1M tokens
gpt-5.5-pro    # $30 in / $180 out per 1M tokens
gpt-5.6-luna   # $0.2 in / $1.2 out per 1M tokens
gpt-5.6-sol    # $5 in / $30 out per 1M tokens
gpt-5.6-terra  # $2 in / $12 out per 1M tokens
```

### `image-to-image`

```
chatgpt-image-latest    # $5 in / $30 out per 1M tokens
gpt-image-1
gpt-image-1-mini        # $2 in / $8 out per 1M tokens
gpt-image-1.5           # $5 in / $32 out per 1M tokens
gpt-image-2             # $5 in / $30 out per 1M tokens
gpt-image-2.5-flare     # $5 in / $30 out per 1M tokens
gpt-image-2.5-sunburst  # $5 in / $30 out per 1M tokens
```

---

## Perplexity

`provider = "perplexity"` · key `PERPLEXITY_API_KEY`

### `text-to-text`

```
sonar                # $1 in / $1 out per 1M tokens
sonar-deep-research  # $2 in / $8 out per 1M tokens
sonar-pro            # $3 in / $15 out per 1M tokens
sonar-reasoning      # $1 in / $5 out per 1M tokens
sonar-reasoning-pro  # $2 in / $8 out per 1M tokens
```

---

## Qwen (DashScope)

`provider = "qwen"` · key `DASHSCOPE_API_KEY`

### `text-to-text`

```
QwQ-32B
qwen-max         # $1.6 in / $6.4 out per 1M tokens
qwen-plus        # $0.4 in / $1.2 out per 1M tokens
qwen-turbo       # $0.05 in / $0.2 out per 1M tokens
qwen3-235b-a22b  # $0.7 in / $2.8 out per 1M tokens
qwen3-32b
qwen3-72b
qwen3-max        # $1.2 in / $6 out per 1M tokens
qwen3-vl-plus    # $0.2 in / $1.6 out per 1M tokens
```

### `text-to-image`

```
wan2.2-t2i-flash
wan2.2-t2i-plus
```

### `image-to-text`

```
qwen-vl-max    # $0.8 in / $3.2 out per 1M tokens
qwen-vl-plus   # $0.21 in / $0.63 out per 1M tokens
qwen3-vl-plus  # $0.2 in / $1.6 out per 1M tokens
```

### `image-to-image`

```
qwen-image-edit
qwen-image-edit-max
qwen-image-edit-plus
```

---

## Recraft

`provider = "recraft"` · key `RECRAFT_API_TOKEN`

### `text-to-image`

```
recraftv3
recraftv3_vector
recraftv4_1
recraftv4_1_pro
recraftv4_1_utility
recraftv4_1_utility_pro
recraftv4_1_utility_vector
recraftv4_1_vector
```

### `image-to-image`

```
recraft-vectorize
recraftv3
recraftv3_vector
```

---

## Reve

`provider = "reve"` · key `REVE_API_KEY`

### `text-to-image`

```
reve-image
```

### `image-to-image`

```
reve-image
```

---

## xAI

`provider = "xai"` · key `XAI_API_KEY`

### `text-to-text`

```
grok-3                     # $3 in / $15 out per 1M tokens
grok-3-fast
grok-3-mini                # $0.3 in / $0.5 out per 1M tokens
grok-4-0709                # $3 in / $15 out per 1M tokens
grok-4-1-fast-reasoning    # $0.2 in / $0.5 out per 1M tokens
grok-4-fast-non-reasoning  # $0.2 in / $0.5 out per 1M tokens
grok-4-fast-reasoning      # $0.2 in / $0.5 out per 1M tokens
```

### `text-to-image`

```
grok-imagine-image
```

### `image-to-text`

```
grok-3                     # $3 in / $15 out per 1M tokens
grok-3-fast
grok-3-mini                # $0.3 in / $0.5 out per 1M tokens
grok-4-0709                # $3 in / $15 out per 1M tokens
grok-4-1-fast-reasoning    # $0.2 in / $0.5 out per 1M tokens
grok-4-fast-non-reasoning  # $0.2 in / $0.5 out per 1M tokens
grok-4-fast-reasoning      # $0.2 in / $0.5 out per 1M tokens
```

### `image-to-image`

```
grok-imagine-image
```
<!-- /g:registry-models -->

---

## Query helpers (`models.registry`)

### `models(provider=None, task=None) → list[str]`

Sorted, deduplicated list of model names. Filter by provider and/or task.

```python
registry.models()                                   # everything
registry.models(task="text-to-image")
registry.models(provider="openai", task="text-to-text")
```

Raises `ValueError` for an unknown provider or task.

### `providers(task=None) → list[str]`

Provider names that have at least one registered model, optionally limited to those supporting a task.

```python
registry.providers()
registry.providers(task="image-to-text")
```

Providers are returned in the order of `PROVIDERS`.

### `accepts(provider) → frozenset[str] | None`

The universal options a provider has a control for at all, from its data —
not what a particular model takes on a particular call, which is an
adaptation reported when it happens. `None` when the provider declares nothing.

### `refresh(provider, api_key=None) → dict`

Diff the registry against the provider's live model list — what the provider
serves that the data does not list, and what the data lists that it no longer
serves. Makes one network call.

### `tasks(model_name) → list[str]`

Tasks supported by a given model across all providers. Empty list when the model is not in the registry — **no error is raised**.

```python
registry.tasks("gpt-4o")        # ["image-to-text", "text-to-text"]
registry.tasks("unknown-model") # []
```

### `is_supported(model_name, task=None) → bool`

Quick membership check, optionally scoped to a task.

```python
registry.is_supported("grok-imagine-image")                   # True
registry.is_supported("grok-imagine-image", "text-to-image")  # True
registry.is_supported("grok-imagine-image", "text-to-text")   # False
registry.is_supported("unknown-model")                        # False
```

---

## Constants

Importable from `models.registry`:

<!-- g:registry-constants -->
| Name | Value |
|---|---|
| `TASKS` | `('text-to-text', 'text-to-image', 'image-to-text', 'image-to-image')` |
| `PROVIDERS` | `('anthropic', 'bfl', 'deepseek', 'google', 'kimi', 'openai', 'perplexity', 'qwen', 'recraft', 'reve', 'xai')` |
| `OPTIONS` | `('temperature', 'top_p', 'top_k', 'max_tokens', 'reasoning', 'cache_control', 'cache_ttl')` |
<!-- /g:registry-constants -->

---

## See also

- [Model factory](../primitives/models.md) — how model names are routed to provider clients.
- [Environment variables](environment-variables.md) — which key each provider needs.
- [Tools reference](../tools-reference/index.md) — all built-in tools.
