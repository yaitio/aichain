# Model registry

A complete, machine-queryable inventory of every model officially supported by the library, organised by provider and task.

The registry is **reference data only** — the `Model` factory accepts any valid model name regardless of whether it appears here. Use the registry for discovery, documentation, and light validation in application code.

```python
from models import registry

registry.models(provider="anthropic")
registry.providers(task="text-to-image")     # ["google", "openai", "xai"]
registry.models(provider="kimi")             # all Kimi models
registry.models(provider="deepseek")         # all DeepSeek models
registry.tasks("gpt-4o")                     # ["image-to-text", "text-to-text"]
registry.is_supported("gpt-image-1", "text-to-image")   # True
```

---

## Tasks

| Task | Description |
|---|---|
| `text-to-text` | Text prompt → text response. Chat, reasoning, instruction-following, code. |
| `text-to-image` | Text prompt → image. Uses dedicated image endpoints (`/v1/images/generations` for OpenAI/xAI; `generateContent` with `responseModalities: ["IMAGE"]` for Google). |
| `image-to-text` | Image (+ optional text) → text response. Models that accept image parts in the universal message format. |

## Providers

`openai`, `anthropic`, `google`, `xai`, `perplexity`, `kimi`, `deepseek`.

A provider is absent for a task it does not support (Anthropic has no text-to-image; Perplexity, Kimi, and DeepSeek have no image-generation models).

---

## OpenAI

### `text-to-text`
```
gpt-5                    # GPT-5 series — Responses API
gpt-5-mini

gpt-4.1                  # GPT-4.1 series
gpt-4.1-mini
gpt-4.1-nano

gpt-4o                   # GPT-4o series
gpt-4o-mini

o3                       # o-series reasoning
o4-mini
o1
```

### `text-to-image`
```
gpt-image-1
gpt-image-1.5            # latest, highest quality
gpt-image-1-mini
```
`dall-e-2` / `dall-e-3` are deprecated (scheduled removal 2026-05-12) and excluded from the registry.

### `image-to-text`
Every GPT-5, GPT-4.1, GPT-4o, and o-series model accepts image input.

---

## Anthropic

All Claude models accept image input — the `image-to-text` list mirrors `text-to-text`. Anthropic has no image-generation models.

### `text-to-text` and `image-to-text`
```
claude-opus-4-6
claude-sonnet-4-6
claude-haiku-4-5-20251001
```

---

## Google AI

### `text-to-text`
```
gemini-3.1-pro-preview
gemini-3.1-flash-lite-preview
gemini-3-flash-preview

gemini-2.5-pro           # GA — no -preview suffix
gemini-2.5-flash
```

### `text-to-image`
```
gemini-3.1-flash-image-preview
gemini-3-pro-image-preview
```

### `image-to-text`
Every Gemini chat model accepts image (and video/audio) input — the list mirrors `text-to-text`.

---

## xAI

### `text-to-text`
```
grok-4-0709              # Grok 4 series — always-on reasoning
grok-4-fast-reasoning
grok-4-1-fast-reasoning

grok-3                   # Grok 3 series
grok-3-fast
grok-3-mini
grok-3-mini-fast
```

### `text-to-image`
```
grok-imagine-image-pro
grok-imagine-image
```

### `image-to-text`
All Grok 3/4 text models accept image input — list mirrors `text-to-text`.

---

## Perplexity

Text-only (no image tasks).

### `text-to-text`
```
sonar-pro                # Sonar search models
sonar

sonar-reasoning-pro      # Built-in chain-of-thought

sonar-deep-research      # Multi-step deep research
```

---

## Kimi (Moonshot AI)

No image-generation models. `kimi-k2.5` accepts image and video input.

### `text-to-text`
```
kimi-k2.5                # K2.5 series — multimodal, thinking toggle
kimi-k2-0905-preview     # K2 series — text-only, 256K context
kimi-k2-turbo-preview    # K2 series — high-speed (60–100 tok/s)
kimi-k2-thinking         # Thinking series — always-on reasoning, 256K context
kimi-k2-thinking-turbo   # Thinking series — fast always-on reasoning
```

### `image-to-text`
```
kimi-k2.5                # Accepts images (PNG, JPEG, WebP, GIF) and video
```

---

## DeepSeek

Text-only (no image tasks). DeepSeek exposes reasoning via a **model switch**, not an API parameter — `reasoning="high"` transparently routes requests to `deepseek-reasoner`.

### `text-to-text`
```
deepseek-chat       # DeepSeek-V3 — standard chat, full parameter support
deepseek-reasoner   # DeepSeek-R1 — always-on chain-of-thought; reasoning_content in response
```

> **Note:** When `reasoning="high"` is set on any DeepSeek model instance, the library automatically uses `deepseek-reasoner` as the effective model name. `temperature` and `top_p` are omitted from the request body for `deepseek-reasoner` (the API ignores them).

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
registry.providers()                      # all seven
registry.providers(task="image-to-text")  # ["anthropic", "google", "kimi", "openai", "xai"]
```

Providers are returned in the canonical order declared in `PROVIDERS`.

### `tasks(model_name) → list[str]`

Tasks supported by a given model across all providers. Empty list when the model is not in the registry — **no error is raised**.

```python
registry.tasks("gpt-4o")        # ["image-to-text", "text-to-text"]
registry.tasks("gpt-image-1")   # ["text-to-image"]
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

| Name | Type | Description |
|---|---|---|
| `TASKS` | `tuple[str, ...]` | `("text-to-text", "text-to-image", "image-to-text")` |
| `PROVIDERS` | `tuple[str, ...]` | `("openai", "anthropic", "google", "xai", "perplexity", "kimi", "deepseek")` |
| `REGISTRY` | `dict[str, dict[str, list[str]]]` | Nested `provider → task → model list`. |

---

## See also

- [Model factory](../primitives/models.md) — how model names are routed to provider clients.
- [Environment variables](environment-variables.md) — which key each provider needs.
- [Tools reference](../tools-reference/index.md) — all built-in tools.
