# Models

A Model is a name. `Model("claude-haiku-5-5")` is Anthropic, `Model("gpt-6-luna")`
is OpenAI, `Model("private/llama3.3")` is a server you run; the rest of your
code does not change when the name does. That is the promise this whole
library rests on, and this page is about what it takes to keep it honestly.

```python
Model("claude-haiku-5-5")                              # the key from ANTHROPIC_API_KEY
Model("gpt-6-luna", options={"reasoning": "high"})     # one vocabulary of options
Model("openai/my-fine-tune")                           # an explicit provider for a name it cannot guess
```

→ Every parameter: [API: Model](../api/model.md) · the catalogue: [model registry](../reference/model-registry.md)

---

## A provider is data, not a class

There is one `Model` class. What differs between providers — the URL, the auth
header, the request and response shapes, what each option is called — lives in
data, `models/providers/*.toml`, read by seven family clients — one per
wire dialect — that cover every provider. A model's capabilities and its price are in the same files. Adding a
provider that speaks an existing dialect is a data file, not code.

A Model does two things: turn the universal messages into the provider's
request, and turn the response back into a clean value. It knows nothing about
prompts, retries or pipelines — those are the [Skill](../pipelines/skills.md)'s
— and that separation is why changing the name changes nothing around it.

## One vocabulary of options

You ask for `temperature`, `max_tokens`, `reasoning` the same way everywhere:

<!-- g:model-options -->
| Key | What it asks for | Providers with a control for it |
|---|---|---|
| `temperature` | how much randomness the model is allowed | Anthropic, BytePlus, DeepSeek, Google AI, Inception (Mercury), Kimi (Moonshot AI), Meta (Muse), MiniMax, Mistral AI, OpenAI, Perplexity, Qwen (DashScope), StepFun, xAI, Xiaomi (MiMo), Z.ai (GLM) |
| `top_p` | nucleus sampling: the probability mass to sample from | Anthropic, BytePlus, DeepSeek, Google AI, Kimi (Moonshot AI), Meta (Muse), MiniMax, Mistral AI, OpenAI, Perplexity, Qwen (DashScope), StepFun, xAI, Xiaomi (MiMo), Z.ai (GLM) |
| `top_k` | sample only from the k most likely tokens | Anthropic, Google AI |
| `max_tokens` | ceiling on the answer's length | Anthropic, BytePlus, DeepSeek, Google AI, Inception (Mercury), Kimi (Moonshot AI), Meta (Muse), MiniMax, Mistral AI, OpenAI, Perplexity, Qwen (DashScope), StepFun, xAI, Xiaomi (MiMo), Z.ai (GLM) |
| `reasoning` | how much deliberation before answering: low / medium / high | Anthropic, BytePlus, DeepSeek, Google AI, Inception (Mercury), Kimi (Moonshot AI), Meta (Muse), MiniMax, Mistral AI, OpenAI, Qwen (DashScope), StepFun, xAI, Z.ai (GLM) |
| `cache_control` | cache the stable prefix so a repeated one is not re-billed: True, an int, or 'last' for an agent loop | Anthropic |
| `cache_ttl` | how long a cached prefix lives: '5m' or '1h' | Anthropic |

`output["format"]` keys, asked per call rather than per model. ✓ marks a key whose loss changes *what* you get rather than how good it is — the set `Model(on_unsupported="requirements")` raises for.

| Key | What it asks for | ✓ | Providers that read it |
|---|---|---|---|
| `size` | pixel dimensions, as 'WIDTHxHEIGHT' | ✓ | Black Forest Labs (FLUX), BytePlus, Google AI, Kling, Krea, OpenAI, Qwen (DashScope), Recraft, xAI |
| `aspect_ratio` | shape without committing to a pixel count, as '16:9' | ✓ | Black Forest Labs (FLUX), BytePlus, Google AI, Kling, Krea, OpenAI, Qwen (DashScope), Recraft, xAI |
| `quality` | how much work to spend on the render |  | OpenAI |
| `background` | transparent, opaque, or let the model decide | ✓ | OpenAI |
| `output_format` | the file format to return: png, jpeg, webp | ✓ | Black Forest Labs (FLUX), OpenAI |
| `compression` | compression level for jpeg and webp, 0-100 | ✓ | OpenAI |
| `seed` | fix the randomness so the same prompt renders the same way | ✓ | Black Forest Labs (FLUX), Krea |
| `fidelity` | how much of the original survives an edit: 0 to 1, or 'low'/'high'. One axis, and providers run it both ways — OpenAI asks how much to preserve, Recraft how much to change, so the number is inverted for the latter | ✓ | OpenAI, Recraft |
<!-- /g:model-options -->

`reasoning` is the clearest case of what one word has to do: it becomes a
thinking budget of 10 000 tokens on Claude 4.5 and 4.6, adaptive thinking at
that effort on Claude 4.7 and later, 8 192 tokens on Google, the word
`medium` on OpenAI, a flag on Qwen, and a different model on DeepSeek. You
write it once.

## Never in silence

A provider cannot always do what was asked: OpenAI has no `top_k`, a
reasoning model refuses `temperature`, an image model ignores a `seed`. Each
option meets one of four fates, and vanishing is not one of them:

| | What happens |
|---|---|
| **translated** | sent in the provider's own word — `aspect_ratio` as pixels |
| **adapted** | changed to fit, and said so — a level becomes a budget |
| **declined** | not sent, because the provider has nothing for it, and said so with what to use instead |
| **refused** | a name the library has no word for — `temperatur` — raises when the Model is made, naming the options that exist |

"Said so" is two things: a warning, once per model and option, and a record on
the result — `skill.last_adaptations`, `result.adaptations` — that a run keeps.
The second matters more: a warning is lost in a log, a line in a run record is
not, and a comparison that cannot say what its arms actually sent is not a
comparison.

Who decides whether a lost option is fatal is the caller, because the cost
depends on the use: a thumbnail does not care about its shape, a fixed layout
slot does, and a measurement is invalidated outright. `on_unsupported="warn"`,
the default, adapts and reports; `"requirements"` raises for the losses that
leave a result looking right and silently wrong — a seed, a size, a background;
`"raise"` raises for any. What each option does on each provider, built by
making the requests: [Parameters, per provider](../reference/parameters.md).

## Defaults are not a baseline

A model run with no options set is not "the model with nothing chosen".
`temperature` starts at 0.0 on DeepSeek, 0.2 on Perplexity, 0.7 on Qwen and
1.0 on the rest; `max_tokens` spans 2 048 to 32 768. They are sent on every
call whether anybody chose them or not, so **two models compared with no
options set are not one variable apart**.

They were not converged, on purpose: a library-wide default would override a
number each provider picked for its own models and change every existing
caller's output. What is owed instead is the fact — `model.effective_options`
says what goes on the wire, and each default's author is written down beside
it: the vendor's own documented default, the vendor's advice adopted, or a
choice this library made. When you compare models, set the options yourself.

## Finding a model

The registry is reference data, not a gatekeeper. It lists what the library
ships and has tested, with capabilities and prices, and answers questions:

```python
from yait_aichain.models import registry
registry.models(task="image-to-text")        # every model that reads images
registry.providers(task="text-to-image")
```

It does not block a name it does not list — a new model works the day the
provider ships it, by its prefix or `provider/` — and it is no guarantee your
account can reach every model it lists. Text and vision are models; image
generation is a tool, `image(service, model=…)`. → [Images](../tools/media.md)

A model on your own hardware is a `private/` name, needs no key, and sends
nothing outside your network. → [Private models](private-models.md)

## The message format

A Model takes one message format for every provider: `{"role", "parts"}`, the
role `system`, `user` or `assistant`, and each part typed:

```python
{"type": "text",  "text": "..."}
{"type": "image", "source": {"kind": "url", "url": "https://…"}, "meta": {"detail": "high"}}
{"type": "image", "source": {"kind": "base64", "mime": "image/png", "data": "…"}}
{"type": "audio", "source": {"kind": "file", "path": "call.wav"}}
{"type": "video", "source": {"kind": "url", "url": "https://…"},
 "meta": {"sampling": {"strategy": "fps", "fps": 1, "max_frames": 60}}}
{"type": "document", "source": {"kind": "base64", "mime": "application/pdf", "data": "…"}}
```

A plain string is a text part, and a dict with `text` and no `type` is one too,
so `{"role": "user", "parts": ["Hello!"]}` is a whole message. A `file` source
is read into base64 when the request is sent, not when the Skill is built, so
its path may carry a `{placeholder}` and one Skill can serve a different file
per call. A part a provider cannot take — video on most of them, audio on
Claude — is left out and reported, never sent in a shape that fails.

The answer's shape is `output`: `{"modalities": [...], "format": {"type": ...}}`,
the type `text`, `json`, `json_schema` (with `schema`, and `strict`) or `image`.
Left out, it is plain text; a missing key takes its default.

## Tool calls in the conversation

A model given tools answers either with text or with a request to call them,
`ToolCallRequest`. Both stay in the same `{"role", "parts"}` messages the rest
of the library speaks, through two more turn kinds:

```python
{"role": "assistant", "parts": [...],                 # the model asked to act
 "tool_calls": [{"id": "...", "name": "...", "arguments": {...}}]}
{"role": "tool", "call_id": "...", "parts": [{"type": "text", "text": "..."}]}
```

`request.as_turn()` writes the first, `tool_result_turn(call_id, result)` the
second. A result is keyed to its call by id, as every provider's native
protocol requires; a result dressed as a user turn ("TOOL RESULT: …") is read
by the model as conversation, not as ground truth.
→ [Driving the loop](../agents/driving-the-loop.md)

## A schema that travels

Schemas break between providers in opposite directions: Google forbids
`additionalProperties` and union types, while OpenAI's strict mode requires
every object closed and every property listed in `required`. A schema written
for one family is rejected by the other. `portable_schema(schema, "openai")` or
`"google"` returns the form that family takes; the one shape with no strict
equivalent, a map with arbitrary keys, raises instead of being mangled.

`check_structure(value, schema)` is the check a `json_schema` answer gets on
return. It is shallow on purpose — a full validator would be a second
dependency — and catches what happens in practice: a provider that ignored the
schema and returned fields nobody declared, or left out a required one.

## When a call fails

Every provider failure is an `APIError` (a `RuntimeError`), and the subclasses
in `yait_aichain.clients` let a caller react to one failure mode — retry on a
rate limit, re-authenticate on a 401 — without parsing messages:

| Status | Raised as |
|---|---|
| 402, or a credits or quota message on 400/403/429 | `InsufficientCreditsError` — checked first, so a billing problem is not read as auth, a rate limit or a bad request |
| 429 | `RateLimitError`, with `retry_after` in seconds |
| 401, 403 | `AuthenticationError` |
| 400, 422 | `InvalidRequestError` |
| 404 | `NotFoundError` |
| 5xx | `ServerError` |
| no response | `NetworkError` |

Tools that make HTTP calls raise the same classes, so `except RateLimitError`
around a tool works too. An error carries at most the first 2 000 characters
of the provider's body: it travels into logs, run records and a model's
context.
