# Changelog

All notable changes to **aichain 2.0** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [2.0.0] — Unreleased

### Added

#### Core primitives
- **`Model` factory** — universal `Model("provider/model-or-name")` constructor dispatches to the matching provider client via prefix detection. Supports `client_options` pass-through and per-call `options` merging.
- **`Skill`** — model-bound task unit with provider-agnostic input templates, three output formats (`text` / `json` / `json_schema`), variable substitution, transient-error retries (HTTP 429/500/502/503/504), and YAML serialisation.
- **`Tool` + `ToolResult`** — JSON-Schema-first action contract with two call styles (call-style wraps errors; run-style raises). `ToolResult.__bool__` enables `if result:` idiom.
- **`Chain`** — sequential pipeline accepting Skills, Tools, and Agents as steps in a normalised 4-tuple `(runner, output_key, input_map, options)`. Accumulated variable dict flows through every step. `on_step_error` modes: `raise`, `stop`, `skip`. YAML serialisation.
- **`Agent`** — plan / act / reflect loop with five reflection decisions (`continue`, `retry`, `replan`, `stop`, `final_answer`), two modes (`waterfall`, `agile`), three budgets (`max_steps`, `max_attempts`, `max_tokens`), and configurable verbosity.
- **`AgentMemory`** — pluggable memory system with `InMemoryBackend` (default) and `FileBackend` (atomic JSON). `MemoryBackend` abstract base for custom backends.
- **`AgentResult`** — typed dataclass capturing `success`, `output`, `mode`, `steps_taken`, `tokens_used`, `plan`, `history`, `memory`, and `error`.

#### Universal reasoning
- Single `reasoning` option (`"low"` / `"medium"` / `"high"`) mapped per-provider:
  - Anthropic → `budget_tokens`
  - Google → `thinkingBudget`
  - OpenAI / xAI → `reasoning_effort`
  - Kimi → `thinking: {type: enabled}` (all levels)
  - DeepSeek → model-name switch (`"high"` → `deepseek-reasoner`)

#### Provider support (7 providers, 3 tasks)
- OpenAI — GPT-5, GPT-4.1, GPT-4o, o-series reasoning, GPT Image models
- Anthropic — Claude 4 series (Opus, Sonnet, Haiku)
- Google AI — Gemini 3.1, 3, 2.5 series; Gemini image models
- xAI — Grok 4, Grok 3 series; Grok image models
- Perplexity — Sonar Pro, Sonar, Sonar Reasoning Pro, Sonar Deep Research
- Kimi — K2.5, K2, K2 Turbo, K2 Thinking series
- DeepSeek — DeepSeek-V3 (chat), DeepSeek-R1 (reasoner)

#### Built-in tools (12)
- **Search**: `PerplexitySearchTool`, `BraveSearchTool`, `SerpApiTool`, `OpenAIWebSearchTool`
- **Conversion**: `MarkItDownTool`, `MistletoeTool`, `WeasyprintTool`
- **Language**: `DeepLTranslateTool`, `DeepLRephraseTool`
- **Long-doc**: `SectionContextTool`
- **Social**: `LateAccountsTool`, `LatePublishTool`

#### Model registry
- `models.registry` module with query helpers: `models()`, `providers()`, `tasks()`, `is_supported()`.
- Constants: `TASKS`, `PROVIDERS`, `REGISTRY`.

---

## Notes on versioning

- The library uses **semantic versioning** (`MAJOR.MINOR.PATCH`).
- `2.0.0` is a full rewrite; there is no migration path from `1.x`.
- The `Model` factory is the single stable public API surface — model name strings and provider clients are expected to be updated frequently as providers release new models.
