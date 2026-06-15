# Cookbook: Image pipeline

> **Skeleton** — full content coming in a future release.

A `Chain` that composes a text model and a text-to-image model: a frontier LLM engineers detailed visual direction from a short brief, then an image model renders the engineered prompt into a structured image result. Swap either step's `Model(...)` to change providers without touching the rest of the pipeline.

## Primitives used

- `Chain`
- `Skill` (prompt engineering + image rendering)
- `Model` — text side (`claude-sonnet-4-6`, `gpt-4o`, `gemini-2.5-pro`, …)
- `Model` — image side (`chatgpt-image-latest`, `gpt-image-2`, `grok-imagine-image-pro`, `gemini-3.1-flash-image`, `wan2.2-t2i-flash`)

## Key decisions

- Single-step render vs two-step (prompt-engineer → render) — the engineered-prompt variant produces markedly stronger results because the image model receives concrete visual language rather than a raw brief.
- One image vs N variants (conservative / contemporary / distinctive) — use `json_schema` output on the art-director step to produce structured direction objects, then invoke the renderer Skill once per direction.
- Which image provider — OpenAI `chatgpt-image-latest` for fast high fidelity (and real transparent PNG via `background: "transparent"`), xAI `grok-imagine-image-pro` for artistic range, Google `gemini-3.1-flash-image` for scene coherence and the strongest in-image text.
- Base64 vs URL — every image provider is normalised to the same `{url, base64, mime_type, revised_prompt}` dict; `mime_type` is detected from image magic bytes so the file extension is always accurate.

## Worked example

See `examples/logo_creation.py` for a two-step pipeline, and `products/logo_design/` for the full three-tier productised version (essential single-call, professional prompt-engineer + render, expert brand-analysis + three variants).

## See also

- [Models](../primitives/models.md)
- [Chain](../primitives/chain.md)
- [Model registry — text-to-image](../reference/model-registry.md#openai)
