# Local models

Run open-weight models on your own hardware and call them through the same
`Model` interface as every cloud provider. The `local` provider speaks to any
server that exposes the OpenAI-compatible `/v1` — which is all of them: vLLM,
Ollama, LM Studio, TGI, SGLang, llama.cpp. Nothing is embedded: the provider
is an HTTP client to a server *you* run, named for what you have (a server of
your own), not for any one implementation.

```python
from yait_aichain.models import Model
from yait_aichain.skills import Skill

model = Model("local/meta-llama/Llama-3.3-70B-Instruct")   # no api_key needed

skill = Skill(model=model, input={
    "messages": [{"role": "user", "parts": ["Summarise: {text}"]}],
})
print(skill.run(variables={"text": "..."}))
```

Everything else in the library — Skills, Chains, Agents, Pools — works
unchanged, because a local server is just another provider.

## Zero new dependencies, by design

Talking to a local server needs nothing beyond the `urllib3` the library
already carries — it is plain HTTP, exactly like talking to OpenAI. The heavy
packages (vllm, torch, CUDA) belong to *running* the server, which stays on
your side of the line:

- **Installing this library never pulls an inference runtime.** No torch, no
  CUDA, no model weights.
- **Starting the server is your one command**, with the tool you already have
  (see below). The library connects to it; it does not manage it.
- If a future feature genuinely requires a heavy package, it will live behind
  `pip install yait-aichain[local]` and fail with exactly that instruction —
  never arrive silently.

## Starting a server

One line each. The library connects to whichever you run.

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

Which model *fits* is a hardware question — the parameter count and
quantisation must fit your VRAM/RAM — and the server's docs are the authority
on that, not this page.

## Naming models

The name after `local/` is passed to the server **verbatim** — there is no
registry to keep in sync, because the server decides what it serves.

- **vLLM / TGI** serve models by Hugging Face id — the same string you gave
  `vllm serve`: `Model("local/meta-llama/Llama-3.3-70B-Instruct")`. Slashes in
  the id are fine; only the first one selects the provider.
- **Ollama** serves by its own tag: `Model("local/llama3.3")`.
- **LM Studio** shows the served name in its server panel.

If the names disagree, the server's error will say so — ask it what it serves:
`curl http://localhost:11434/v1/models`.

## Pointing at the server

Three ways, in order of precedence:

```python
# 1. Explicit — wins over everything
Model("local/llama3.3", client_options={"url": "http://localhost:11434"})

# 2. Environment — for a remote GPU box, no code change
#    export LOCAL_BASE_URL=http://gpu-box:8000

# 3. Default — http://localhost:8000 (vLLM's port)
Model("local/qwen3-30b")
```

Both URL spellings work: `http://localhost:11434` and
`http://localhost:11434/v1` reach the same server. Server docs quote the
`/v1`-suffixed form, so pasting it must not break — and does not.

## Authentication

None needed — the provider sends no `Authorization` header when there is no
key. If the server was started with `--api-key`, provide it either way:

```python
Model("local/llama3.3", api_key="my-server-key")
# or: export LOCAL_API_KEY=my-server-key
```

## What `cost` means here: `None`

`usage.cost` on a local call is `None`, and that is an answer, not a gap.
A local model has no price per token — its economics are **GPU-hours divided
by throughput**, and throughput depends on your card, not the model. Token
*counts* still arrive as usual (`input_tokens`, `output_tokens`), so you can
compute your own $/token from your hardware cost if you need one. A built-in
number would be invented.

## Limits, honestly

- **Chat and vision only.** Image generation, embeddings and reranking are
  not routed to local servers yet, even where the server supports them.
- **No quirk handling.** Cloud providers get per-model branches (reasoning
  flags, JSON-mode hints); local servers get the plain OpenAI-compatible
  request. If your server needs a special flag, it won't be sent.
- **Reasoning options are not translated.** `options={"reasoning": ...}` is
  ignored for local models rather than mapped to server-specific flags.
- **Verified at the wire level** — request shape, routing, auth, URL forms —
  against the OpenAI-compatible contract all five servers document. Run one
  call against your own server before relying on it in a pipeline.

## See also

- [Model](models.md) — the shared interface this provider plugs into.
- [Environment variables](../reference/environment-variables.md) —
  `LOCAL_BASE_URL`, `LOCAL_API_KEY`.
