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

`anthropic`, `bfl`, `deepseek`, `google`, `kimi`, `krea`, `meta`, `minimax`, `mistral`, `openai`, `perplexity`, `qwen`, `recraft`, `stepfun`, `xai`, `xiaomi`, `zai`.

The `private` provider is registry-less by design: its catalogue belongs to the server you run (vLLM, Ollama, LM Studio, …), so any name it serves works and none is listed here. See [Private models](../models/private-models.md).

`openrouter` is registry-less too: it serves its own catalogue, named after the prefix — `Model("openrouter/vendor/model")` — and bills in the response, key `OPENROUTER_API_KEY`.

| Provider | `text-to-text` | `text-to-image` | `image-to-text` | `image-to-image` |
|---|---|---|---|---|
| Anthropic | 14 | — | 14 | — |
| Black Forest Labs (FLUX) | — | 11 | — | 2 |
| DeepSeek | 4 | — | — | — |
| Google AI | 8 | 5 | 8 | 5 |
| Kimi (Moonshot AI) | 4 | — | 1 | — |
| Krea | — | 4 | — | 1 |
| Meta (Muse) | 5 | — | 5 | — |
| MiniMax | 8 | — | 1 | — |
| Mistral AI | 8 | — | 6 | — |
| OpenAI | 20 | 7 | 20 | 7 |
| Perplexity | 4 | — | — | — |
| Qwen (DashScope) | 13 | 2 | 3 | 3 |
| Recraft | — | 9 | — | 3 |
| StepFun | 4 | — | 2 | — |
| xAI | 14 | 3 | 13 | 3 |
| Xiaomi (MiMo) | 3 | — | 3 | — |
| Z.ai (GLM) | 11 | — | 2 | — |

---

## Anthropic

`provider = "anthropic"` · key `ANTHROPIC_API_KEY`

### `text-to-text`

```
claude-fable-5              # $10 in / $50 out per 1M tokens
claude-fable-5-1            # $10 in / $50 out per 1M tokens
claude-haiku-4-5-20251001   # $1 in / $5 out per 1M tokens
claude-haiku-5-5            # $0.1 in / $0.5 out per 1M tokens
claude-opus-4-5-20251101    # $5 in / $25 out per 1M tokens
claude-opus-4-6             # $5 in / $25 out per 1M tokens
claude-opus-4-7             # $5 in / $25 out per 1M tokens
claude-opus-4-8             # $5 in / $25 out per 1M tokens
claude-opus-5               # $5 in / $25 out per 1M tokens
claude-opus-5-5             # $4 in / $20 out per 1M tokens
claude-sonnet-4-5-20250929  # $3 in / $15 out per 1M tokens
claude-sonnet-4-6           # $3 in / $15 out per 1M tokens
claude-sonnet-5             # $2 in / $10 out per 1M tokens
claude-sonnet-5-5           # $2 in / $10 out per 1M tokens
```

### `image-to-text`

```
claude-fable-5              # $10 in / $50 out per 1M tokens
claude-fable-5-1            # $10 in / $50 out per 1M tokens
claude-haiku-4-5-20251001   # $1 in / $5 out per 1M tokens
claude-haiku-5-5            # $0.1 in / $0.5 out per 1M tokens
claude-opus-4-5-20251101    # $5 in / $25 out per 1M tokens
claude-opus-4-6             # $5 in / $25 out per 1M tokens
claude-opus-4-7             # $5 in / $25 out per 1M tokens
claude-opus-4-8             # $5 in / $25 out per 1M tokens
claude-opus-5               # $5 in / $25 out per 1M tokens
claude-opus-5-5             # $4 in / $20 out per 1M tokens
claude-sonnet-4-5-20250929  # $3 in / $15 out per 1M tokens
claude-sonnet-4-6           # $3 in / $15 out per 1M tokens
claude-sonnet-5             # $2 in / $10 out per 1M tokens
claude-sonnet-5-5           # $2 in / $10 out per 1M tokens
```

---

## Black Forest Labs (FLUX)

`provider = "bfl"` · key `BFL_API_KEY`

### `text-to-image`

```
flux-2-flex
flux-2-klein-4b
flux-2-klein-9b
flux-2-max
flux-2-pro          # per_image $0.03
flux-3-image
flux-dev
flux-kontext-max    # per_image $0.08
flux-kontext-pro    # per_image $0.04
flux-pro-1.1        # per_image $0.04
flux-pro-1.1-ultra  # per_image $0.06
```

### `image-to-image`

```
flux-kontext-max  # per_image $0.08
flux-kontext-pro  # per_image $0.04
```

---

## DeepSeek

`provider = "deepseek"` · key `DEEPSEEK_API_KEY`

### `text-to-text`

```
deepseek-chat      # $0.27 in / $1.1 out per 1M tokens
deepseek-flash     # $0.3 in / $1.2 out per 1M tokens
deepseek-reasoner  # $0.55 in / $2.19 out per 1M tokens
deepseek-v4-pro    # $1.32 in / $3.96 out per 1M tokens
```

---

## Google AI

`provider = "google"` · key `GOOGLE_AI_API_KEY`

### `text-to-text`

```
gemini-2.5-flash        # $0.3 in / $2.5 out per 1M tokens
gemini-2.5-pro          # $1.25 in / $10 out per 1M tokens
gemini-3-flash-preview  # $0.5 in / $3 out per 1M tokens
gemini-3.1-flash-lite   # $0.25 in / $1.5 out per 1M tokens
gemini-3.1-pro-preview  # $2 in / $12 out per 1M tokens
gemini-3.5-flash-lite   # $0.3 in / $2.5 out per 1M tokens
gemini-3.6-flash        # $0.75 in / $3.75 out per 1M tokens
gemini-3.8-flash        # $0.75 in / $3.75 out per 1M tokens
```

### `text-to-image`

```
gemini-2.5-flash-image       # per_image $0.039
gemini-3-pro-image           # per_image $0.134
gemini-3.1-flash-image       # per_image $0.067
gemini-3.1-flash-lite-image  # per_image $0.0336
gemini-nano-banana-2.1       # per_image $0.0336
```

### `image-to-text`

```
gemini-2.5-flash        # $0.3 in / $2.5 out per 1M tokens
gemini-2.5-pro          # $1.25 in / $10 out per 1M tokens
gemini-3-flash-preview  # $0.5 in / $3 out per 1M tokens
gemini-3.1-flash-lite   # $0.25 in / $1.5 out per 1M tokens
gemini-3.1-pro-preview  # $2 in / $12 out per 1M tokens
gemini-3.5-flash-lite   # $0.3 in / $2.5 out per 1M tokens
gemini-3.6-flash        # $0.75 in / $3.75 out per 1M tokens
gemini-3.8-flash        # $0.75 in / $3.75 out per 1M tokens
```

### `image-to-image`

```
gemini-2.5-flash-image       # per_image $0.039
gemini-3-pro-image           # per_image $0.134
gemini-3.1-flash-image       # per_image $0.067
gemini-3.1-flash-lite-image  # per_image $0.0336
gemini-nano-banana-2.1       # per_image $0.0336
```

---

## Kimi (Moonshot AI)

`provider = "kimi"` · key `MOONSHOT_API_KEY`

### `text-to-text`

```
kimi-k2.6                 # $0.95 in / $4 out per 1M tokens
kimi-k2.7-code            # $0.95 in / $4 out per 1M tokens
kimi-k2.7-code-highspeed  # $1.9 in / $8 out per 1M tokens
kimi-k3                   # $3 in / $15 out per 1M tokens
```

### `image-to-text`

```
kimi-k2.6  # $0.95 in / $4 out per 1M tokens
```

---

## Krea

`provider = "krea"` · key `KREA_API_KEY`

### `text-to-image`

```
krea-2-large   # per_image $0.06
krea-2-medium  # per_image $0.03
krea-2-turbo   # per_image $0.015
krea-3         # per_image $0.03
```

### `image-to-image`

```
krea-3  # per_image $0.03
```

---

## Meta (Muse)

`provider = "meta"` · key `META_API_KEY`

### `text-to-text`

```
muse-spark-1.1              # $1.25 in / $4.25 out per 1M tokens
muse-spark-1.2              # $1.25 in / $4.25 out per 1M tokens
muse-spark-1.2-contributor  # $0.1 in / $0.2 out per 1M tokens
muse-spark-1.3              # $1.25 in / $4.25 out per 1M tokens
muse-spark-1.3-contributor  # $0.1 in / $0.2 out per 1M tokens
```

### `image-to-text`

```
muse-spark-1.1              # $1.25 in / $4.25 out per 1M tokens
muse-spark-1.2              # $1.25 in / $4.25 out per 1M tokens
muse-spark-1.2-contributor  # $0.1 in / $0.2 out per 1M tokens
muse-spark-1.3              # $1.25 in / $4.25 out per 1M tokens
muse-spark-1.3-contributor  # $0.1 in / $0.2 out per 1M tokens
```

---

## MiniMax

`provider = "minimax"` · key `MINIMAX_API_KEY`

### `text-to-text`

```
MiniMax-M2              # $0.3 in / $1.2 out per 1M tokens
MiniMax-M2.1            # $0.3 in / $1.2 out per 1M tokens
MiniMax-M2.1-highspeed  # $0.6 in / $2.4 out per 1M tokens
MiniMax-M2.5            # $0.3 in / $1.2 out per 1M tokens
MiniMax-M2.5-highspeed  # $0.6 in / $2.4 out per 1M tokens
MiniMax-M2.7            # $0.3 in / $1.2 out per 1M tokens
MiniMax-M2.7-highspeed  # $0.6 in / $2.4 out per 1M tokens
MiniMax-M3              # $0.3 in / $1.2 out per 1M tokens
```

### `image-to-text`

```
MiniMax-M3  # $0.3 in / $1.2 out per 1M tokens
```

---

## Mistral AI

`provider = "mistral"` · key `MISTRAL_API_KEY`

### `text-to-text`

```
codestral-2508      # $0.3 in / $0.9 out per 1M tokens
ministral-14b-2512  # $0.2 in / $0.2 out per 1M tokens
ministral-3b-2512   # $0.1 in / $0.1 out per 1M tokens
ministral-8b-2512   # $0.15 in / $0.15 out per 1M tokens
mistral-large-2512  # $0.5 in / $1.5 out per 1M tokens
mistral-large-4-0   # $1.36 in / $4.18 out per 1M tokens
mistral-medium-3-5  # $1.5 in / $7.5 out per 1M tokens
mistral-small-2603  # $0.15 in / $0.6 out per 1M tokens
```

### `image-to-text`

```
ministral-14b-2512  # $0.2 in / $0.2 out per 1M tokens
ministral-3b-2512   # $0.1 in / $0.1 out per 1M tokens
ministral-8b-2512   # $0.15 in / $0.15 out per 1M tokens
mistral-large-2512  # $0.5 in / $1.5 out per 1M tokens
mistral-large-4-0   # $1.36 in / $4.18 out per 1M tokens
mistral-medium-3-5  # $1.5 in / $7.5 out per 1M tokens
```

---

## OpenAI

`provider = "openai"` · key `OPENAI_API_KEY`

### `text-to-text`

```
gpt-4.1        # $2 in / $8 out per 1M tokens
gpt-4o         # $2.5 in / $10 out per 1M tokens
gpt-4o-mini    # $0.15 in / $0.6 out per 1M tokens
gpt-5          # $1.25 in / $10 out per 1M tokens
gpt-5-mini     # $0.25 in / $2 out per 1M tokens
gpt-5-nano     # $0.05 in / $0.4 out per 1M tokens
gpt-5.2        # $1.75 in / $14 out per 1M tokens
gpt-5.4        # $2.5 in / $15 out per 1M tokens
gpt-5.4-mini   # $0.75 in / $4.5 out per 1M tokens
gpt-5.4-nano   # $0.2 in / $1.25 out per 1M tokens
gpt-5.4-pro    # $30 in / $180 out per 1M tokens
gpt-5.5        # $5 in / $30 out per 1M tokens
gpt-5.5-pro    # $30 in / $180 out per 1M tokens
gpt-5.6-luna   # $0.2 in / $1.2 out per 1M tokens
gpt-5.6-sol    # $5 in / $30 out per 1M tokens
gpt-5.6-terra  # $2 in / $12 out per 1M tokens
gpt-6-astra    # $10 in / $50 out per 1M tokens
gpt-6-luna     # $0.1 in / $0.5 out per 1M tokens
gpt-6-sol      # $2 in / $10 out per 1M tokens
gpt-6.1-sol    # $2 in / $10 out per 1M tokens
```

### `text-to-image`

```
chatgpt-image-latest    # $5 in / $32 out per 1M tokens
gpt-image-1             # $5 in / $40 out per 1M tokens
gpt-image-1-mini        # $2 in / $8 out per 1M tokens
gpt-image-1.5           # $5 in / $32 out per 1M tokens
gpt-image-2             # $5 in / $30 out per 1M tokens
gpt-image-2.5-flare     # $5 in / $30 out per 1M tokens
gpt-image-2.5-sunburst  # $5 in / $30 out per 1M tokens
```

### `image-to-text`

```
gpt-4.1        # $2 in / $8 out per 1M tokens
gpt-4o         # $2.5 in / $10 out per 1M tokens
gpt-4o-mini    # $0.15 in / $0.6 out per 1M tokens
gpt-5          # $1.25 in / $10 out per 1M tokens
gpt-5-mini     # $0.25 in / $2 out per 1M tokens
gpt-5-nano     # $0.05 in / $0.4 out per 1M tokens
gpt-5.2        # $1.75 in / $14 out per 1M tokens
gpt-5.4        # $2.5 in / $15 out per 1M tokens
gpt-5.4-mini   # $0.75 in / $4.5 out per 1M tokens
gpt-5.4-nano   # $0.2 in / $1.25 out per 1M tokens
gpt-5.4-pro    # $30 in / $180 out per 1M tokens
gpt-5.5        # $5 in / $30 out per 1M tokens
gpt-5.5-pro    # $30 in / $180 out per 1M tokens
gpt-5.6-luna   # $0.2 in / $1.2 out per 1M tokens
gpt-5.6-sol    # $5 in / $30 out per 1M tokens
gpt-5.6-terra  # $2 in / $12 out per 1M tokens
gpt-6-astra    # $10 in / $50 out per 1M tokens
gpt-6-luna     # $0.1 in / $0.5 out per 1M tokens
gpt-6-sol      # $2 in / $10 out per 1M tokens
gpt-6.1-sol    # $2 in / $10 out per 1M tokens
```

### `image-to-image`

```
chatgpt-image-latest    # $5 in / $32 out per 1M tokens
gpt-image-1             # $5 in / $40 out per 1M tokens
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
sonar-reasoning-pro  # $2 in / $8 out per 1M tokens
```

---

## Qwen (DashScope)

`provider = "qwen"` · key `DASHSCOPE_API_KEY`

### `text-to-text`

```
qwen-flash     # $0.05 in / $0.4 out per 1M tokens
qwen-max       # $1.6 in / $6.4 out per 1M tokens
qwen-plus      # $0.4 in / $1.2 out per 1M tokens
qwen-turbo     # $0.05 in / $0.2 out per 1M tokens
qwen3-vl-plus  # $0.2 in / $1.6 out per 1M tokens
qwen3.5-plus   # $0.4 in / $2.4 out per 1M tokens
qwen3.6-flash  # $0.25 in / $1.5 out per 1M tokens
qwen3.6-plus   # $0.5 in / $3 out per 1M tokens
qwen3.7-flash  # $0.03 in / $0.13 out per 1M tokens
qwen3.7-max    # $2.5 in / $7.5 out per 1M tokens
qwen3.7-plus   # $0.4 in / $1.6 out per 1M tokens
qwen3.8-flash  # $0.15 in / $0.47 out per 1M tokens
qwen3.8-max    # $2 in / $6 out per 1M tokens
```

### `text-to-image`

```
wan2.2-t2i-flash  # per_image $0.025
wan2.2-t2i-plus   # per_image $0.05
```

### `image-to-text`

```
qwen-vl-max    # $0.8 in / $3.2 out per 1M tokens
qwen-vl-plus   # $0.21 in / $0.63 out per 1M tokens
qwen3-vl-plus  # $0.2 in / $1.6 out per 1M tokens
```

### `image-to-image`

```
qwen-image-edit       # per_image $0.045
qwen-image-edit-max   # per_image $0.075
qwen-image-edit-plus  # per_image $0.03
```

---

## Recraft

`provider = "recraft"` · key `RECRAFT_API_TOKEN`

### `text-to-image`

```
recraftv3                   # per_image $0.04
recraftv3_vector            # per_image $0.08
recraftv4_1                 # per_image $0.035
recraftv4_1_flash           # per_image $0.007
recraftv4_1_pro             # per_image $0.21
recraftv4_1_utility
recraftv4_1_utility_pro
recraftv4_1_utility_vector
recraftv4_1_vector          # per_image $0.08
```

### `image-to-image`

```
recraft-vectorize  # per_image $0.01
recraftv3          # per_image $0.04
recraftv3_vector   # per_image $0.08
```

---

## StepFun

`provider = "stepfun"` · key `STEPFUN_API_KEY`

### `text-to-text`

```
step-3.5-flash       # $0.1 in / $0.3 out per 1M tokens
step-3.5-flash-2603  # $0.1 in / $0.3 out per 1M tokens
step-3.7-flash       # $0.2 in / $1.15 out per 1M tokens
step-5-preview       # $1 in / $2.7 out per 1M tokens
```

### `image-to-text`

```
step-3.7-flash  # $0.2 in / $1.15 out per 1M tokens
step-5-preview  # $1 in / $2.7 out per 1M tokens
```

---

## xAI

`provider = "xai"` · key `XAI_API_KEY`

### `text-to-text`

```
grok-3                        # $1.25 in / $2.5 out per 1M tokens
grok-3-fast
grok-3-mini                   # $0.3 in / $0.5 out per 1M tokens
grok-4-0709                   # $1.25 in / $2.5 out per 1M tokens
grok-4-1-fast-reasoning       # $1.25 in / $2.5 out per 1M tokens
grok-4-fast-non-reasoning     # $1.25 in / $2.5 out per 1M tokens
grok-4-fast-reasoning         # $1.25 in / $2.5 out per 1M tokens
grok-4.20-0309-non-reasoning  # $1.25 in / $2.5 out per 1M tokens
grok-4.20-0309-reasoning      # $1.25 in / $2.5 out per 1M tokens
grok-4.3                      # $1.25 in / $2.5 out per 1M tokens
grok-4.5                      # $2 in / $6 out per 1M tokens
grok-4.6                      # $2 in / $6 out per 1M tokens
grok-4.7                      # $2 in / $6 out per 1M tokens
grok-build-0.1                # $1 in / $2 out per 1M tokens
```

### `text-to-image`

```
grok-imagine-image          # per_image $0.02
grok-imagine-image-2.0      # per_image $0.04
grok-imagine-image-quality  # per_image $0.05
```

### `image-to-text`

```
grok-3                        # $1.25 in / $2.5 out per 1M tokens
grok-3-fast
grok-3-mini                   # $0.3 in / $0.5 out per 1M tokens
grok-4-0709                   # $1.25 in / $2.5 out per 1M tokens
grok-4-1-fast-reasoning       # $1.25 in / $2.5 out per 1M tokens
grok-4-fast-non-reasoning     # $1.25 in / $2.5 out per 1M tokens
grok-4-fast-reasoning         # $1.25 in / $2.5 out per 1M tokens
grok-4.20-0309-non-reasoning  # $1.25 in / $2.5 out per 1M tokens
grok-4.20-0309-reasoning      # $1.25 in / $2.5 out per 1M tokens
grok-4.3                      # $1.25 in / $2.5 out per 1M tokens
grok-4.5                      # $2 in / $6 out per 1M tokens
grok-4.6                      # $2 in / $6 out per 1M tokens
grok-4.7                      # $2 in / $6 out per 1M tokens
```

### `image-to-image`

```
grok-imagine-image          # per_image $0.02
grok-imagine-image-2.0      # per_image $0.04
grok-imagine-image-quality  # per_image $0.05
```

---

## Xiaomi (MiMo)

`provider = "xiaomi"` · key `MIMO_API_KEY`

### `text-to-text`

```
mimo-v2.6-flash           # $0.14 in / $0.28 out per 1M tokens
mimo-v2.6-pro             # $0.435 in / $0.87 out per 1M tokens
mimo-v2.6-pro-ultraspeed  # $4.35 in / $8.7 out per 1M tokens
```

### `image-to-text`

```
mimo-v2.6-flash           # $0.14 in / $0.28 out per 1M tokens
mimo-v2.6-pro             # $0.435 in / $0.87 out per 1M tokens
mimo-v2.6-pro-ultraspeed  # $4.35 in / $8.7 out per 1M tokens
```

---

## Z.ai (GLM)

`provider = "zai"` · key `ZAI_API_KEY`

### `text-to-text`

```
glm-4.5         # $0.6 in / $2.2 out per 1M tokens
glm-4.5-air     # $0.2 in / $1.1 out per 1M tokens
glm-4.6         # $0.6 in / $2.2 out per 1M tokens
glm-4.7         # $0.6 in / $2.2 out per 1M tokens
glm-5           # $1 in / $3.2 out per 1M tokens
glm-5-turbo     # $1.2 in / $4 out per 1M tokens
glm-5.1         # $1.4 in / $4.4 out per 1M tokens
glm-5.2         # $1.4 in / $4.4 out per 1M tokens
glm-5.3         # $1.4 in / $4.4 out per 1M tokens
glm-5.3-flash   # $0.15 in / $0.5 out per 1M tokens
glm-5.3-flashx  # $0.37 in / $1.25 out per 1M tokens
```

### `image-to-text`

```
glm-5.3-flash   # $0.15 in / $0.5 out per 1M tokens
glm-5.3-flashx  # $0.37 in / $1.25 out per 1M tokens
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
| `PROVIDERS` | `('anthropic', 'bfl', 'deepseek', 'google', 'kimi', 'krea', 'meta', 'minimax', 'mistral', 'openai', 'perplexity', 'qwen', 'recraft', 'stepfun', 'xai', 'xiaomi', 'zai')` |
| `OPTIONS` | `('temperature', 'top_p', 'top_k', 'max_tokens', 'reasoning', 'cache_control', 'cache_ttl')` |
<!-- /g:registry-constants -->

---

## See also

- [Model factory](../models/models.md) — how model names are routed to provider clients.
- [Environment variables](environment-variables.md) — which key each provider needs.
- [Tools reference](tools.md) — all built-in tools.
