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
| `temperature` | how much randomness the model is allowed | Anthropic, DeepSeek, Google AI, Kimi (Moonshot AI), OpenAI, Perplexity, Qwen (DashScope), xAI |
| `top_p` | nucleus sampling: the probability mass to sample from | Anthropic, DeepSeek, Google AI, Kimi (Moonshot AI), OpenAI, Perplexity, Qwen (DashScope), xAI |
| `top_k` | sample only from the k most likely tokens | Anthropic, Google AI |
| `max_tokens` | ceiling on the answer's length | Anthropic, DeepSeek, Google AI, Kimi (Moonshot AI), OpenAI, Perplexity, Qwen (DashScope), xAI |
| `reasoning` | how much deliberation before answering: low / medium / high | Anthropic, DeepSeek, Google AI, Kimi (Moonshot AI), OpenAI, Qwen (DashScope), xAI |
| `cache_control` | cache the stable prefix so a repeated one is not re-billed: True, an int, or 'last' for an agent loop | Anthropic |
| `cache_ttl` | how long a cached prefix lives: '5m' or '1h' | Anthropic |

`output["format"]` keys, asked per call rather than per model. ✓ marks a key whose loss changes *what* you get rather than how good it is — the set `Model(on_unsupported="requirements")` raises for.

| Key | What it asks for | ✓ | Providers that read it |
|---|---|---|---|
| `size` | pixel dimensions, as 'WIDTHxHEIGHT' | ✓ | Black Forest Labs (FLUX), Google AI, OpenAI, Qwen (DashScope), Recraft, xAI |
| `aspect_ratio` | shape without committing to a pixel count, as '16:9' | ✓ | Black Forest Labs (FLUX), Google AI, OpenAI, Qwen (DashScope), Recraft, xAI |
| `quality` | how much work to spend on the render |  | OpenAI |
| `background` | transparent, opaque, or let the model decide | ✓ | OpenAI |
| `output_format` | the file format to return: png, jpeg, webp | ✓ | Black Forest Labs (FLUX), OpenAI |
| `compression` | compression level for jpeg and webp, 0-100 | ✓ | OpenAI |
| `seed` | fix the randomness so the same prompt renders the same way | ✓ | Black Forest Labs (FLUX) |
| `fidelity` | how much of the original survives an edit: 0 to 1, or 'low'/'high'. One axis, and providers run it both ways — OpenAI asks how much to preserve, Recraft how much to change, so the number is inverted for the latter | ✓ | OpenAI, Recraft |
<!-- /g:model-options -->

`reasoning` is the clearest case of what one word has to do: it becomes a
thinking budget of 10 000 tokens on Anthropic, 8 192 on Google, the word
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
