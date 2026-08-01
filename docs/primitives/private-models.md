# Private models

Run open-weight models on hardware you control and call them through the same
`Model` interface as every cloud provider. The weights, the prompts and the
answers stay inside your perimeter — nothing is sent to a vendor, because
there is no vendor in the path.

```python
from yait_aichain.models import Model
from yait_aichain.skills import Skill

# Your own server. No API key — there is nobody to authenticate to.
model = Model("private/meta-llama/Llama-3.3-70B-Instruct")

skill = Skill(model=model, input={
    "messages": [{"role": "user", "parts": ["Summarise this contract: {text}"]}],
})
print(skill.run(variables={"text": confidential_document}))
```

That is the whole integration. Skills, Chains, Agents and Pools work
unchanged, because a server of your own is just another provider — and
switching to or from a cloud model stays a one-word change.

The `private` provider speaks to any server exposing the OpenAI-compatible
`/v1`, which is all of them: vLLM, Ollama, LM Studio, TGI, SGLang, llama.cpp.

## What `private` means, and what it does not

The prefix names **the intent of your configuration, not a property the
library verifies.** Point it at a server in your own network and the data
never leaves; point it at a hosted endpoint and it does, and the word on the
prefix changes nothing about that. The library cannot tell the two apart —
both are an HTTP request to a URL you supplied.

So the guarantee is yours to make, and it rests on one thing: **the URL.**
Everything below assumes it points at a machine you control. Aggregators are
covered in [their own section](#using-an-aggregator-and-why-that-is-not-private)
— supported, useful, and explicitly not this.

A useful tell: a server of your own rarely asks for a key. If you had to set
`PRIVATE_API_KEY` to make something work, ask yourself whose machine is on
the other end.

## Zero new dependencies, by design

Talking to the server needs nothing beyond the `urllib3` the library already
carries — it is plain HTTP, exactly like talking to OpenAI. The heavy
packages (vllm, torch, CUDA) belong to *running* the server, which stays on
your side of the line:

- **Installing this library never pulls an inference runtime.** No torch, no
  CUDA, no model weights.
- **Starting the server is your one command**, with the tool you already have
  (below). The library connects to it; it does not manage it.
- If a future feature genuinely requires a heavy package, it will live behind
  `pip install yait-aichain[private]` and fail with exactly that instruction —
  never arrive silently.

## Starting a server

One line each. The library connects to whichever you run; their docs are the
authority on installing them.

| Server | Start command | Default URL |
|---|---|---|
| [vLLM](https://docs.vllm.ai) | `vllm serve meta-llama/Llama-3.3-70B-Instruct` | `http://localhost:8000` |
| [Ollama](https://ollama.com) | `ollama run llama3.3` | `http://localhost:11434` |
| [LM Studio](https://lmstudio.ai) | *Developer → Start server* | `http://localhost:1234` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `llama-server -m model.gguf` | `http://localhost:8080` |

vLLM needs Linux + CUDA; Ollama and LM Studio run on macOS and Windows.

## Which models can I run?

The catalogue belongs to the server, not to this library — browse it where
the server does:

- **vLLM** loads models straight from the
  [Hugging Face Hub](https://huggingface.co/models); its docs keep the list
  of [supported architectures](https://docs.vllm.ai/en/latest/models/supported_models.html).
  Most open-weight text models (Llama, Qwen, Mistral, Gemma, DeepSeek) work.
- **Ollama** has its own curated [model library](https://ollama.com/library)
  with quantised builds sized for laptops.
- **LM Studio** ships a built-in model browser and a
  [catalogue](https://lmstudio.ai/models) on the web.
- **llama.cpp** runs any [GGUF file](https://huggingface.co/models?library=gguf)
  from the Hub.

Which model *fits* is a hardware question — parameter count and quantisation
against your VRAM/RAM — and the server's docs answer it, not this page.

## Naming models

The name after `private/` is passed to the server **verbatim** — there is no
registry to keep in sync, because the server decides what it serves.

- **vLLM / TGI** serve by Hugging Face id — the same string you gave
  `vllm serve`: `Model("private/meta-llama/Llama-3.3-70B-Instruct")`. Slashes
  in the id are fine; only the first one selects the provider.
- **Ollama** serves by its own tag: `Model("private/llama3.3")`.
- **LM Studio** shows the served name in its server panel.

If the names disagree, the server's error will say so — ask it what it serves:
`curl http://localhost:11434/v1/models`.

## Pointing at the server

Three ways, in order of precedence:

```python
# 1. Explicit — wins over everything
Model("private/llama3.3", client_options={"url": "http://localhost:11434"})

# 2. Environment — for a GPU box elsewhere on your network, no code change
#    export PRIVATE_BASE_URL=http://gpu.internal:8000

# 3. Default — http://localhost:8000 (vLLM's port)
Model("private/qwen3-30b")
```

`private` is about **who controls the machine, not where it sits**: a GPU
server in your own datacentre is as private as your laptop, and a rented one
is exactly as private as your agreement with whoever rents it.

Both URL spellings work: `http://localhost:11434` and
`http://localhost:11434/v1` reach the same server. Server docs quote the
`/v1`-suffixed form, so pasting it must not break — and does not.

## Authentication

Usually none — the provider sends no `Authorization` header when there is no
key, which is what an open server expects. If yours was started with
`--api-key`, provide it either way:

```python
Model("private/llama3.3", api_key="my-server-key")
# or: export PRIVATE_API_KEY=my-server-key
```

## Using an aggregator (and why that is not private)

Together, Fireworks, OpenRouter, Groq, DeepInfra and friends host the same
open weights behind the same OpenAI-compatible `/v1`. This provider talks to
them, and that is deliberately allowed:

```python
import os

Model("private/meta-llama/Llama-3.3-70B-Instruct",
      api_key=os.environ["TOGETHER_API_KEY"],
      client_options={"url": "https://api.together.xyz/v1"})
```

**Your data leaves your perimeter.** The prompt, the documents in it and the
answer all travel to a third party and fall under their retention and
processing terms. The `private/` prefix does not change that and does not
audit it — it is the same code path, aimed elsewhere. If you reached for this
provider for a compliance reason, an aggregator does not satisfy it.

What you *do* keep is the other half of the argument:

| | your own server | aggregator | cloud vendor API |
|---|---|---|---|
| Data stays in your perimeter | **yes** | no | no |
| Open weights — the model can't be retired under you | **yes** | **yes** | no |
| No GPU to own or operate | no | **yes** | **yes** |
| Same aichain code | **yes** | **yes** | **yes** |

So an aggregator is a reasonable middle: model portability without hardware,
at the cost of the privacy property. Choose it knowingly.

Two practical notes. An aggregator always needs a key — that requirement is
itself the signal that someone else's machine is answering. And `usage.cost`
stays `None`, because the library carries no price table for their catalogue;
their pricing page and your token counts give you the real figure.

## What `cost` means here: `None`

`usage.cost` on a private call is `None`, and that is an answer, not a gap.
A model you host has no price per token — its economics are **GPU-hours
divided by throughput**, and throughput depends on your card, not the model.
Token *counts* still arrive as usual (`input_tokens`, `output_tokens`), so
you can compute your own rate from your hardware cost.

## Limits, honestly

- **Chat and vision only.** Image generation, embeddings and reranking are
  not routed to private servers yet, even where the server supports them.
- **No quirk handling.** Cloud providers get per-model branches (reasoning
  flags, JSON-mode hints); a private server gets the plain OpenAI-compatible
  request. If your server needs a special flag, it won't be sent.
- **Reasoning options are not translated.** `options={"reasoning": ...}` is
  ignored here rather than mapped to server-specific flags.
- **Verified at the wire level** — request shape, routing, auth, URL forms —
  against the OpenAI-compatible contract these servers document. Run one call
  against your own server before relying on it in a pipeline.

## See also

- [Model](models.md) — the shared interface this provider plugs into.
- [Environment variables](../reference/environment-variables.md) —
  `PRIVATE_BASE_URL`, `PRIVATE_API_KEY`.
