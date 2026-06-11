# Model

`Model` is the routing layer of the gateway. You ask for a model by name; the library detects the provider, constructs the correct subclass, and returns a fully-configured instance with an HTTP client already attached.

Everything above a `Model` — Skills, Tools, Chains, Agents — is provider-agnostic. Every provider difference (URL, auth header, request shape, response shape, reasoning parameter) is absorbed here.

---

## Construction

```python
from models import Model

Model("gpt-4o")                  # → OpenAIModel
Model("claude-sonnet-4-6")       # → AnthropicModel
Model("gemini-2.5-pro")          # → GoogleAIModel
Model("grok-3")                  # → XAIModel
Model("sonar-pro")               # → PerplexityModel
Model("kimi-k2.5")               # → KimiModel
Model("deepseek-chat")           # → DeepSeekModel
```

Provider is resolved from a well-known prefix:

| Prefix / pattern | Provider |
|---|---|
| `claude-` | Anthropic |
| `gemini-` | Google AI |
| `grok-` | xAI |
| `sonar`, `r1-1776` | Perplexity |
| `kimi-` | Kimi (Moonshot AI) |
| `deepseek-` | DeepSeek |
| `gpt-`, `dall-e-`, `text-embedding-`, `whisper-`, `tts-`, `o<digit>` | OpenAI |

Unknown names raise `ValueError`. If you need a model name that doesn't match any prefix (e.g. a custom fine-tune), instantiate the provider subclass directly:

```python
from models import OpenAIModel
model = OpenAIModel("my-custom-ft-name")
```

---

## Options

### `options` — generation parameters

Every provider accepts the same keys. Unsupported keys are silently ignored by providers that don't use them.

```python
Model("gpt-4o", options={
    "temperature": 0.3,
    "max_tokens":  2048,
    "top_p":       0.9,
})
```

| Key | Type | Notes |
|---|---|---|
| `temperature` | float | Sampling temperature. |
| `max_tokens` | int | Maximum output tokens. |
| `top_p` | float | Nucleus sampling probability mass. |
| `top_k` | int | Top-K sampling (provider-dependent). |
| `cache_control` | bool | Enable provider-level prompt caching. |
| `reasoning` | `None` \| `"low"` \| `"medium"` \| `"high"` | Universal reasoning depth — see below. |

### Universal reasoning

`reasoning` is one knob that maps to each provider's native reasoning controls:

| Provider | Translated to |
|---|---|
| Anthropic | `{"type": "enabled", "budget_tokens": N}` — low=4k, medium=10k, high=20k. Temperature forced to 1.0 when active. |
| Google AI | `generationConfig.thinkingConfig.thinkingBudget` — low=2 048, medium=8 192, high=24 576. |
| OpenAI | `reasoning_effort` on o-series. GPT models ignore it. |
| xAI | `reasoning_effort` for grok-3-mini / grok-3-mini-fast. Other grok models ignore it. |
| Kimi | `{"thinking": {"type": "enabled"}}`. All three levels map to enabled (no budget knob). Temperature forced to 1.0 when active. |
| DeepSeek | Model-name switch — `"high"` routes to `deepseek-reasoner` (always-on CoT); `"low"`/`"medium"` keep `deepseek-chat`. Temperature and top_p are omitted from the request for `deepseek-reasoner`. |
| Perplexity | Not used. |

```python
Model("claude-opus-4-6",     options={"reasoning": "high"})    # 20k thinking tokens
Model("o3",                  options={"reasoning": "medium"})  # OpenAI reasoning effort
Model("grok-3-mini-fast",    options={"reasoning": "low"})     # xAI reasoning effort
Model("gemini-2.5-pro",      options={"reasoning": "high"})    # Gemini thinking budget
Model("kimi-k2.5",           options={"reasoning": "high"})    # Kimi thinking enabled
Model("deepseek-chat",       options={"reasoning": "high"})    # routes to deepseek-reasoner
```

### `client_options` — HTTP client tuning

| Key | Type | Notes |
|---|---|---|
| `url` | str | Override the base URL (useful for proxies or regional endpoints). |
| `timeout` | `urllib3.Timeout` | Custom connect / read timeout. |
| `retries` | `urllib3.Retry` | Custom retry policy for the underlying HTTP layer. |
| `proxy` | dict | `{"url": "http://proxy:3128"}`. |

```python
Model(
    "gpt-4o",
    client_options={"proxy": {"url": "http://corp-proxy:3128"}},
)
```

### `api_key`

Explicit keys override the environment variable:

```python
Model("claude-sonnet-4-6", api_key="sk-ant-...")
```

If neither `api_key` nor the provider's environment variable is set, construction raises `ValueError`.

---

## What a Model does

Exactly two things:

1. **`to_request(messages, output) → (path, body)`** — convert the universal message format plus output spec into a provider-native request.
2. **`from_response(response, output) → str | dict`** — extract a clean Python value from the raw JSON response.

A Model knows nothing about prompts, templates, tools, or pipelines. It does not touch retries, errors, or substitution — those belong to `Skill` and `Chain`.

That isolation is why swapping models is a one-line change.

---

## The Model registry

The registry is **reference data**, not validation. The factory accepts any syntactically valid name — use the registry to discover what this library has been tested with.

```python
from models import registry

registry.models(task="text-to-image")
# ['gemini-3-pro-image-preview', 'gemini-3.1-flash-image-preview',
#  'gpt-image-1', 'gpt-image-1-mini', 'gpt-image-1.5',
#  'grok-imagine-image', 'grok-imagine-image-pro']

registry.providers(task="text-to-image")
# ['openai', 'google', 'xai']

registry.tasks("gpt-4o")
# ['image-to-text', 'text-to-text']

registry.is_supported("gpt-image-1", "text-to-image")   # True
registry.is_supported("gpt-image-1", "text-to-text")    # False
```

Registered tasks: `text-to-text`, `text-to-image`, `image-to-text`.

---

## Provider reference

All providers expose the same interface. The table below is for discovery only; full model lists live in `models/_registry.py` and change as providers release new models.

| Provider | Env var | Representative models |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-5`, `gpt-4.1`, `gpt-4o`, `o3`, `o4-mini`, `gpt-image-1` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` |
| Google AI | `GOOGLE_AI_API_KEY` | `gemini-3.1-pro-preview`, `gemini-2.5-pro`, `gemini-2.5-flash` |
| xAI | `XAI_API_KEY` | `grok-4-0709`, `grok-4-fast-reasoning`, `grok-3`, `grok-imagine-image` |
| Perplexity | `PERPLEXITY_API_KEY` | `sonar-pro`, `sonar`, `sonar-reasoning-pro`, `sonar-deep-research` |
| Kimi | `MOONSHOT_API_KEY` | `kimi-k2.5`, `kimi-k2-0905-preview`, `kimi-k2-thinking`, `kimi-k2-thinking-turbo` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat`, `deepseek-reasoner` |

---

## Direct subclass construction

The factory is the recommended entry point, but every subclass is also a public import:

```python
from models import (
    OpenAIModel, AnthropicModel, GoogleAIModel, XAIModel, PerplexityModel, KimiModel, DeepSeekModel,
)

model = AnthropicModel("claude-sonnet-4-6", options={"max_tokens": 8192})
model = KimiModel("kimi-k2.5", options={"reasoning": "high"})
model = DeepSeekModel("deepseek-chat", options={"reasoning": "high"})  # routes to deepseek-reasoner
```

Use the subclass form when a model name doesn't match any built-in prefix.

---

## See also

- **Use a Model in a task** → [Skill](skills.md)
- **Pipeline of model calls** → [Chain](chain.md)
- **Model-driven planning & tool use** → [Agent](../agents/overview.md)
