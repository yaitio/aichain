# Design — 1.4.4 / 1.5.0 / 1.5.2

Detailed feature breakdown for the next three releases. Grounded in the current
codebase (`Tool`, `Gate`, `Agent.run`/`resume`, `FileStore`, `Pool`, `Usage`,
the `send()` lifecycle seam). Each release is additive — `run()` semantics and
existing signatures stay backward compatible.

---

## 1.4.4 — Step boundary (observability & control)

**Goal:** make every step of an Agent/Chain/Skill *legible* and *governable* from
outside the model — the library emits structured signals and consults injectable
policy; the environment decides sinks and decisions. Implements the
"harness validates/authorizes/executes/records" stance.

### 1. Structured logging instead of `print()`

- One `logging.Logger` per package (`yait_aichain.agent`, `.chain`, `.pool`,
  `.clients`, `.skills`), each with a `NullHandler` — the library never
  configures handlers or formats; the app/environment owns the sink.
- Levels: `DEBUG` = wire/prompt detail, `INFO` = step/decision, `WARNING` =
  retry/fallback/budget, `ERROR` = terminal failure.
- `verbose=` is kept as a convenience that raises the package logger level (back
  compat with current `verbose=1`), but the canonical path is `logging`.
- **Secret hygiene:** api keys, `Authorization`/`x-api-key` headers, and raw
  credentials are never logged (redact at the emit site).

### 2. Lifecycle hooks (before/after)

- A hook protocol callable at each boundary:
  `before_llm_call`, `after_llm_call`, `before_tool_call`, `after_tool_call`,
  `before_step`, `after_step`, `on_suspend`, `on_resume`, `on_finish`.
- Each hook receives one immutable event object (frozen dataclass): `run_id`,
  `step`, `kind`, `name` (tool/model), `args`/`messages`, `result`, `usage`
  (delta), `duration`, `error`. No hidden chain-of-thought is exposed — only
  operational fields.
- Registration: `Agent(..., hooks=[...])` / `Chain(..., hooks=[...])` /
  `Skill(..., hooks=[...])`; multiple hooks compose in order.
- Hooks are **observe-only** by default. Mutation/approval is a separate,
  explicit mechanism (see §4–5) so observation can't silently change behavior.

### 3. Observability events / tracing

- A single typed event stream sits *on top of* the hook mechanism:
  `llm_call.{started,ended}`, `tool_call.{started,ended}`, `step.{started,ended}`,
  `run.{suspended,resumed,finished}`.
- Each event: `run_id`, `step`, `type`, `ts` (stamped by the runtime, not inside
  pure logic), `payload`, `usage_delta`, `cost_delta` (we already compute
  `Usage`/cost).
- Pluggable `Tracer` sink interface; defaults shipped: a no-op tracer and a
  logging tracer. The product plugs OpenTelemetry / file / DB. The library ships
  **no** heavy tracing dependency.

### 4. Permission matrix (beyond the binary `Gate`)

- Risk class per tool, declared as **data** on the tool (mirrors the
  data-driven provider registry): `risk ∈ {read, draft, write, external,
  financial, destructive, privileged}`. Default `write` for tools with side
  effects, `read` for pure reads.
- A `PermissionPolicy` the harness consults *before* executing any tool:
  maps risk class → `allow | approve | deny`. Shipped default: `read → allow`,
  mutating/`external` → `approve`, `destructive`/`privileged` → `deny` unless
  explicitly allowed.
- Policy is injectable: `Agent(..., permissions=PermissionPolicy(...))` — env
  decides. The model never decides its own permission (decision is harness-side).
- `Gate` becomes a thin special case of an `approve` decision, not a separate
  concept.

### 5. Tool-call approval (reuses suspend/resume)

- When the policy returns `approve`, the harness **pauses** using the existing
  1.3.x machinery: the run parks in the `Store` with a pending-approval marker;
  a separate invocation resumes via `agent.resume(run_id, signal=decision)`.
  No new persistence layer — it rides `FileStore`/`RunDocument`.
- The approval record (decision, who/when, the exact tool + args) is persisted in
  the run document for audit.
- Cleanly supports the serverless two-invocation pattern already shown in
  `examples/18_agent_external_trigger.py`.

### 6. Tool-call repair

- **Local arg validation** against `Tool.parameters` (JSON schema) *before*
  execution. On mismatch, return a model-readable remediation observation
  instead of raising — one bounded corrective retry (configurable, default 1),
  counted against the step/token budget.
- Same path for a tool that raises a structured error: convert to an observation
  and let the model correct, within the repair budget.

### 7. Invariant (formalized)

- **Every tool call returns a result** — even on denial, timeout, error, or
  abort. No silent drop. This becomes a tested invariant.

### Out of scope for 1.4.4 (tracked separately)

- Reversible PII masking / guardrails (the Presidio track) — a sibling step,
  not in this list. To be slotted (likely `1.4.5` or folded into the guardrails
  work). See the Presidio reference memory.

---

## 1.5.0 — Eval tooling

**Goal:** run a scenario across models/prompts/inputs and score it — rule-based
and LLM-judge — producing a comparison table with quality, cost, and latency.
Reuses `Pool` for parallel fan-out and `Usage`/cost for the money/latency columns.

### 1. Eval core

- `Eval(runner, cases, scorers, models=[...])` where `runner` is any
  Skill/Chain/Agent. Runs the matrix `{models × cases}` concurrently via `Pool`.
- Returns a structured `EvalReport` (per-case rows + aggregates), never prints by
  itself — caller formats/export.

### 2. Cases / dataset

- A `Case` = input `variables` + optional `expected`/`reference` + `metadata`
  (tags, difficulty). Loadable from a list, JSON, or file. No DB dependency.

### 3. Scorers

- **Rule-based:** exact match, regex, contains/not-contains, JSON-schema valid,
  numeric tolerance, latency budget, cost budget. Pure, deterministic, free.
- **LLM-judge:** a judge `Skill` scores output against a rubric (pointwise) or
  ranks candidates (pairwise / CRISP-style). Returns score + rationale.
- **Composite:** weighted combination of scorers → one aggregate per case.

### 4. Report / comparison table

- Per-case scores; aggregates per model and per prompt; a markdown comparison
  table with quality, **cost**, and **latency** columns (from `Usage`).
- Export markdown + JSON. Mirrors the hand-rolled `IMAGE-MODEL-BENCH` matrix,
  but as a library primitive.

### 5. DX riders (land with eval because eval needs typed outputs)

- `result_type=PydanticModel` on `Skill` — typed structured output: validate the
  parse, and on failure reuse the **1.4.4 tool-call-repair** path for one
  corrective retry. (Optional dependency on `pydantic`; degrade gracefully.)
- `@tool` decorator — define a `Tool` from a plain function + type hints,
  auto-deriving the JSON schema from the signature (less boilerplate than the
  current `Tool` subclass + manual `parameters`).

### 6. Regression eval

- Persist a baseline `EvalReport`; compare a new run against it; flag
  regressions (score drop beyond a threshold). Supports the best-practice
  "add a regression eval per production incident."

### 7. Showcase

- Re-express the `compare` (benchmark → leaderboard) and `llm_council`
  (ensemble → chairman) cookbooks on top of the eval primitives, so the
  cookbooks double as the feature's documentation.

---

## 1.5.2 — Local vs proprietary benchmark

**Goal:** prove the no-lock-in axis end to end — test the *same* eval scenario
against a frontier API model and a local OSS model, swapping only the
environment injection. Produces a concrete quality/cost/latency comparison.

### 0. Prerequisite — local / OpenAI-compatible provider (target 1.5.1)

- A generic "OpenAI-compatible local endpoint" provider (vLLM is the reference;
  Ollama / LM Studio / TGI / SGLang / llama.cpp serve the same `/v1`).
  `client="openai"`, default `base_url=http://localhost:8000`, `chat_path`,
  `models_path`.
- **Keyless auth** (`auth="none"` / `key_optional`): skip the api-key
  requirement and send no `Authorization` header, but allow a key when the
  server was started with `--api-key`.
- `base_url` override via `client_options={"url": ...}` (already supported) plus
  an env convenience (e.g. `VLLM_BASE_URL`). Arbitrary model id already works
  (`Model.__init__` doesn't gate on the registry).

### 1. Stand up vLLM

- Doc + a small script to serve the models:
  `vllm serve CohereLabs/aya-vision-8b` and a comparable Qwen (~9B; pick the
  nearest real `Qwen3` id at build time). Note GPU/VRAM requirements and the
  served-model-name → wire-name mapping.

### 2. Model set

- `CohereLabs/aya-vision-8b` is **multimodal (vision / image-to-text)**, so the
  benchmark includes vision cases — verify our universal message parts map to
  vLLM's OpenAI-compatible `image_url` content blocks.
- The Qwen ~9B covers text (and optionally a VL variant for vision parity).

### 3. Benchmark run

- Drive both local models and the proprietary baselines (e.g. a GPT / Gemini /
  Claude tier) through the **1.5.0 eval tooling** on the same cases.
- Compare: quality (LLM-judge), **cost** (local ≈ free vs per-token $), and
  **latency** (local infra vs hosted).

### 4. Deliverable

- A benchmark report (in the `IMAGE-MODEL-BENCH` spirit) plus a cookbook:
  "run the same eval against a local model and a frontier model, side by side"
  — the canonical no-lock-in demonstration for the funnel.

### Risks / checks

- vLLM vision via the OpenAI-compatible API: confirm our content-part format
  (text + image) serializes to what vLLM expects.
- Tool-calling on small OSS models is uneven — keep the eval scenarios for
  1.5.2 mostly generation/vision, not heavy agentic tool use.
