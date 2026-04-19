# Examples

Ten runnable scripts that demonstrate the core value of aichain — one interface, every provider.

Each example is self-contained. Export the API keys for the providers you want to test; any subset works. Models whose key is not set are skipped automatically.

---

## The examples

### 1. Universal provider interface
**`text_to_text.py`**

Sends one prompt to every registered text-to-text model and prints each response. The simplest demonstration of the gateway: change `Model("claude-sonnet-4-6")` to `Model("gpt-4o")` to `Model("gemini-2.5-flash")` — nothing else in the code changes. Provider differences (auth, request shape, response parsing) are absorbed by the `Model` layer.

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
# ... any subset of provider keys
python examples/text_to_text.py
```

---

### 2. Structured JSON output across all providers
**`json_schema_cross_provider.py`**

Extracts a validated product spec (`name`, `price`, `rating`, `pros`, `cons`, `verdict`) from a review text. Uses `json_schema` output format — one schema definition, every provider. Without aichain, structured output requires different wiring per provider: `response_format` on OpenAI, a `tool_use` trick on Anthropic, `responseSchema` on Google, and so on.

```bash
python examples/json_schema_cross_provider.py
```

---

### 3. Skill as a reusable function
**`template_variables.py`**

Builds one `Skill` object once and calls it across a batch of inputs with different `variables=` dicts. Shows default variables, per-call overrides, and automatic retry (`max_retries=3`). Contrast: with raw APIs you reconstruct the full `messages` list, serialize it, and implement retry logic from scratch on every call and every provider.

```bash
export OPENAI_API_KEY="sk-..."
python examples/template_variables.py
```

---

### 4. Reasoning depth as a single dial
**`text_to_text_reasoning.py`**

Sets `reasoning="high"` on every reasoning-capable model. The library translates this to the correct native mechanism per provider: `budget_tokens` on Anthropic, `thinkingBudget` on Google, `reasoning_effort` on OpenAI/xAI, a model switch to `deepseek-reasoner` on DeepSeek. One parameter, six provider-native implementations.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
python examples/text_to_text_reasoning.py
```

---

### 5. Chain with automatic variable flow
**`chain_summarize_translate.py`**

A two-step `Chain`: step 1 summarises an article (output stored as `{result}`), step 2 translates the summary using `{result}` and the original `{language}` variable. Variable accumulation is automatic — no manual wiring, no glue code between steps.

```bash
export OPENAI_API_KEY="sk-..."
python examples/chain_summarize_translate.py
```

---

### 6. Text-to-image normalisation across 3 providers
**`text_to_image.py`**

Sends the same image prompt to every registered text-to-image model (`gpt-image-1`, `grok-imagine-image-pro`, `gemini-3.1-flash-image-preview`) and saves the result to `examples/output/`. Every provider returns the same `{url, base64, mime_type, revised_prompt}` dict. Without aichain: OpenAI uses `/v1/images/generations`, xAI mirrors that endpoint, Google uses `generateContent` with `responseModalities: ["IMAGE"]` — three completely different request and response shapes.

```bash
export OPENAI_API_KEY="sk-..."
export XAI_API_KEY="xai-..."
export GOOGLE_AI_API_KEY="AIza..."
python examples/text_to_image.py
```

---

### 7. Text + image models in one Chain
**`logo_creation.py`**

A two-step `Chain` where step 1 (Claude Sonnet) engineers a detailed visual direction from a short brand brief, and step 2 (gpt-image-1) renders it. Two providers, two modalities, zero glue code. The Chain output key `prompt_text` flows between steps automatically. Swap either `Model(...)` to switch providers without touching anything else.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
python examples/logo_creation.py
```

---

### 8. Autonomous agent with tools and budget control
**`agent_research.py`**

A two-phase pipeline: phase 1 runs an `Agent` (plan / act / reflect loop) with Brave search and MarkItDown fetch tools, up to 12 steps and 80k tokens. Phase 2 loads the agent's memory and writes a structured report. `AgentResult` reports `steps_taken`, `tokens_used`, and a full `history`. Comparable raw-API orchestration is several hundred lines of custom code.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export BRAVE_API_KEY="..."
python examples/agent_research.py
```

---

### 9. YAML persistence — define once, load anywhere
**`save_and_load.py`**

Builds a JSON-schema keyword extractor Skill, serialises it to YAML with `.save()`, then loads it back with `Skill.load()`. The loaded Skill reconstructs its `Model` from the saved model name — attach any compatible key at load time. The same `.save()` / `.load()` API works for `Chain`.

```bash
export OPENAI_API_KEY="sk-..."
python examples/save_and_load.py
```

---

### 10. Sectional document generation — no token-limit ceiling
**`sectional_document.py`**

Generates a 4-section business brief where each section is an independent model call. A `SectionContextTool` advances a queue and exposes `{current_section_title}`, `{current_section_plan}`, and a `{recent_summaries}` rolling-context window to the write Skill. After each section a cheap summarise Skill compresses it to bullets. `assemble_document()` joins everything in order. The result length is bounded only by API budget — not by any single `max_output_tokens` ceiling.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python examples/sectional_document.py
```

---

---

### 11. Universal REST API tool — full booking lifecycle
**`booking_api.py`**

Demonstrates `RestApiTool`: one class, any REST endpoint, zero boilerplate.
Defines five endpoints for the public [Restful Booker](https://restful-booker.herokuapp.com/)
test API and runs them in a single Chain: authenticate → create booking →
read it back → partial update → delete.

**Key patterns:**
- Each endpoint is declared once in 4–8 lines (method, URL, which fields go into path / query / body, auth strategy).
- The Chain accumulates response fields automatically — `token` from `POST /auth` and `bookingid` from `POST /booking` flow into later steps by name without any `input_map` wiring.
- `input_map` is used for one rename: `updated_needs → additionalneeds` in the PATCH step.
- Dict responses are auto-merged; the plain-text DELETE response (`"Created"`) is stored as a string.

No API keys required — Restful Booker is a free public test API.

```bash
# Full Chain demo
python examples/booking_api.py

# Direct tool calls (no Chain)
python examples/booking_api.py --standalone

# Both
python examples/booking_api.py --both
```

---

---

### 12. Connect any MCP server — tools, agents, chains
**`mcp_tool.py`**

`MCPTools(server)` discovers all tools on any MCP server and returns a `list[MCPTool]` — ready to drop straight into an Agent or Chain with no async code. `MCPTool` wraps one server operation behind the standard `run()` interface. Five scenarios: (1) HTTP server — connect, list tools, call; (2) STDIO subprocess — launch a local `npx` MCP server; (3) Agent — register all MCP tools in three lines; (4) Chain — MCP fetch step piped into a Skill; (5) direct `MCPTool` without discovery when the tool name is known. Supports Streamable HTTP, SSE, and STDIO transports. Requires `pip install fastmcp`.

```bash
pip install fastmcp

python examples/mcp_tool.py                   # all scenarios
python examples/mcp_tool.py --scenario http
python examples/mcp_tool.py --scenario stdio
python examples/mcp_tool.py --scenario agent
python examples/mcp_tool.py --scenario chain
python examples/mcp_tool.py --scenario direct
```

---

### 13. Qwen / DashScope provider — text, vision, reasoning
**`qwen_skills.py`**

Three self-contained scenarios using Alibaba DashScope: (1) chat completion with `qwen-max` and `qwen-turbo`; (2) image understanding with `qwen-vl-max` — pass any public image URL; (3) deep reasoning with `QwQ-32B` (always-on chain-of-thought). The library routes `qwen-*`, `QwQ-*`, and `wanx-*` model names to `QwenModel` automatically. The base URL is region-aware — set `DASHSCOPE_REGION` to `ap` (default, international), `us`, `cn`, or `hk`.

```bash
export DASHSCOPE_API_KEY="sk-..."
export DASHSCOPE_REGION="ap"          # optional; default is ap (international)

python examples/qwen_skills.py                      # all scenarios
python examples/qwen_skills.py --scenario text
python examples/qwen_skills.py --scenario vision
python examples/qwen_skills.py --scenario reasoning
```

---

## Quick start

```bash
# Clone and set up
cd aichain_2.0

# Run any example with at least one key set
export OPENAI_API_KEY="sk-..."
python examples/text_to_text.py
```

Output files (images) are written to `examples/output/`. YAML files from `save_and_load.py` go to `examples/skills/`.
