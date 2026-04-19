# `deepl_translate` and `deepl_rephrase`

Two tools wrapping the DeepL API:

| Tool | Class | API |
|---|---|---|
| `deepl_translate` | `DeepLTranslateTool` | Translate text between 30+ languages. |
| `deepl_rephrase` | `DeepLRephraseTool` | Rewrite for clarity, grammar, style, and tone (DeepL Write). |

Both share the same env var and authentication, and both automatically route to the correct endpoint based on key suffix.

---

## Requirements

| | |
|---|---|
| Env var | `DEEPL_API_KEY` |
| Get a key | <https://www.deepl.com/pro-api> |
| Free endpoint | `api-free.deepl.com` (keys ending with `:fx`) |
| Pro endpoint | `api.deepl.com` |

Key suffix → endpoint routing is handled automatically. Pass `api_key=` to override, or set `DEEPL_API_KEY`.

---

## `DeepLTranslateTool`

### Parameters

| Name | Type | Required | Notes |
|---|---|---|---|
| `text` | `string` | ✓ | UTF-8 plain text, max 128 KiB. |
| `target_lang` | `string` | ✓ | `DE`, `FR`, `RU`, `JA`, `ZH`, `ES`, `IT`, `EN-US`, `EN-GB`, `PT-BR`, … |
| `source_lang` | `string` | | Auto-detected when omitted. |
| `formality` | `string` | | `default` / `more` / `less` / `prefer_more` / `prefer_less`. |
| `context` | `string` | | Contextual hint (not itself translated). Improves word-choice in ambiguous cases. |
| `preserve_formatting` | `boolean` | | Preserve original whitespace/punctuation. Default `false`. |

### Examples

```python
from tools import DeepLTranslateTool

tool = DeepLTranslateTool()

# Auto-detect source
tool.run(text="Good morning", target_lang="RU")
# → "Доброе утро"

# Formal German with a contextual hint
tool.run(
    text        = "Can you help me with this?",
    target_lang = "DE",
    formality   = "more",
    context     = "Customer support email to a corporate client",
)

# Preserve whitespace + punctuation
tool.run(
    text                = "  Hello —  world.  ",
    target_lang         = "FR",
    preserve_formatting = True,
)
```

---

## `DeepLRephraseTool`

Rewrites input for clarity, grammar, style, and tone while preserving meaning. Backed by the DeepL Write endpoint.

Supported languages: `de`, `en` (+ `en-GB`, `en-US`), `es`, `fr`, `it`, `ja`, `ko`, `pt` (+ `pt-BR`, `pt-PT`), `zh` (+ `zh-Hans`).

### Parameters

| Name | Type | Required | Notes |
|---|---|---|---|
| `text` | `string` | ✓ | UTF-8 plain text. |
| `target_lang` | `string` | | Auto-detected when omitted. |
| `writing_style` | `string` | | `academic`, `business`, `casual`, `default`, `simple` (and `prefer_*` fallbacks). |
| `tone` | `string` | | `confident`, `default`, `diplomatic`, `enthusiastic`, `friendly` (and `prefer_*` fallbacks). |

### Examples

```python
from tools import DeepLRephraseTool

tool = DeepLRephraseTool()

# Polish business prose
improved = tool.run(
    text          = "We are pleased to inform you of our decision.",
    writing_style = "business",
    tone          = "confident",
)

# Simplify for a general audience
simple = tool.run(text=technical_paragraph, writing_style="simple")

# Academic Russian
tool.run(text=russian_text, target_lang="ru", writing_style="academic")
```

### `prefer_*` fallbacks

Some styles/tones aren't available for every target language. The `prefer_*` variants ask DeepL to apply the style/tone where supported and silently fall back to default elsewhere — useful in multi-language pipelines.

---

## Chain example — translate → polish

```python
from chain import Chain
from tools import DeepLTranslateTool, DeepLRephraseTool

chain = Chain(steps=[
    (DeepLTranslateTool(),  "draft",    {"text": "english_copy"}),
    (DeepLRephraseTool(),   "final",    {"text": "draft"}),
], variables={"target_lang": "DE", "writing_style": "business"})
```

(In practice, translate/rephrase arguments are passed via each step's `input_map` or via a skill that formulates the call for an agent.)

---

## Notes

- Both endpoints raise `RuntimeError` on non-2xx. The `tool(…)` call style captures it in `ToolResult.error`.
- Request timeout: connect=10s, read=30s.
- DeepL language codes are case-insensitive — the tool uppercases for translate and lowercases for rephrase automatically.

---

## See also

- [Models overview](../primitives/models.md) — DeepL is **not** a chat model; these tools stand in for translation-focused steps in Chains and Agents.
