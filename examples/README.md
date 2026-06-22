# Simple Examples

One concept per file. Read top to bottom — each example builds on the previous one.

Every file is self-contained: copy it, set your API keys, run it.

```bash
source .env   # or export ANTHROPIC_API_KEY="sk-ant-..." etc.
python examples/simple/01_skill.py
```

---

## Skills

A **Skill** is the atomic unit — one prompt template, one model, one result.

| File | What it shows | Keys needed |
|---|---|---|
| [`01_skill.py`](01_skill.py) | Run a prompt against Claude. The minimum viable aichain program. | `ANTHROPIC_API_KEY` |
| [`02_skill_models.py`](02_skill_models.py) | Same prompt, three providers — Claude, GPT, Gemini. Swap `Model()` to change provider; nothing else changes. | `ANTHROPIC_API_KEY` `OPENAI_API_KEY` `GOOGLE_AI_API_KEY` |
| [`03_skill_multimodal.py`](03_skill_multimodal.py) | Three modalities in one file: text → text (Claude) → image (Grok) → vision analysis (Qwen). | `ANTHROPIC_API_KEY` `XAI_API_KEY` `DASHSCOPE_API_KEY` |
| [`04_skill_save_load.py`](04_skill_save_load.py) | Save a skill to YAML, reload it anywhere. API keys never touch disk — resolved from env at load time. | `ANTHROPIC_API_KEY` |

---

## Tools

A **Tool** wraps any callable — a REST API, an MCP server, your own function.

| File | What it shows | Keys needed |
|---|---|---|
| [`05_tool_convert.py`](05_tool_convert.py) | Fetch any URL or file and convert it to clean Markdown in one call. | — |
| [`06_tool_mcp.py`](06_tool_mcp.py) | Connect to an MCP server, discover its tools at runtime, call them. Uses a live NewsAPI MCP server. | MCP server running |
| [`07_tool_custom.py`](07_tool_custom.py) | Define your own Tool subclass from scratch, use it standalone, then plug it directly into a Chain. | `ANTHROPIC_API_KEY` |

---

## Chains

A **Chain** wires steps in sequence — each step's output automatically flows into the next.

| File | What it shows | Keys needed |
|---|---|---|
| [`08_chain.py`](08_chain.py) | Two skills in sequence, two different providers — GPT writes, Claude reviews. Shows variable flow. | `OPENAI_API_KEY` `ANTHROPIC_API_KEY` |
| [`09_chain_tool_skill.py`](09_chain_tool_skill.py) | Tool + Skill in one pipeline — fetch a webpage, then Claude summarises it in 3 bullets. | `ANTHROPIC_API_KEY` |
| [`10_chain_save_load.py`](10_chain_save_load.py) | Save an entire chain to YAML and reload it. The full pipeline definition persists; keys do not. | `OPENAI_API_KEY` `ANTHROPIC_API_KEY` |

---

## Pool

A **Pool** runs the same runner across many items in parallel. Total time = slowest item, not the sum.

| File | What it shows | Keys needed |
|---|---|---|
| [`11_pool.py`](11_pool.py) | Same skill, 5 topics, all fired simultaneously. Shows `status` polling and per-item `history`. | `ANTHROPIC_API_KEY` |
| [`12_pool_chain.py`](12_pool_chain.py) | A full Chain (fetch → summarise) used as the Pool runner — 3 URLs processed in parallel, each running its own pipeline. | `ANTHROPIC_API_KEY` |

---

## Agents

An **Agent** plans, picks tools, acts, reflects, and decides when to stop — autonomously.

| File | What it shows | Keys needed |
|---|---|---|
| [`13_agent.py`](13_agent.py) | One tool, one task — the agent plans and executes without being told how. Shows steps taken and tokens used. | `ANTHROPIC_API_KEY` `PERPLEXITY_API_KEY` |
| [`14_agent_tools.py`](14_agent_tools.py) | Two tools available — the agent decides which to use for each step without being told. | `ANTHROPIC_API_KEY` `PERPLEXITY_API_KEY` |
| [`15_agent_orchestrator.py`](15_agent_orchestrator.py) | Orchestrator spawns sub-agents — one per research topic. Each runs independently; orchestrator compiles final report. | `ANTHROPIC_API_KEY` `PERPLEXITY_API_KEY` |

---

## State — suspend & resume

A run can **pause** until an external signal arrives (a human, a webhook, a
cron tick) and **resume** later — even in a different process. The run is
parked in a store; `resume(run_id, signal)` continues from exactly where it
stopped, without re-running completed steps. This is the serverless core.

| File | What it shows | Keys needed |
|---|---|---|
| [`17_chain_human_input.py`](17_chain_human_input.py) | Human-in-the-loop: a chain drafts a reply, **pauses** for a person to approve/edit it at the console, then resumes and sends it. | `OPENAI_API_KEY` |
| [`18_agent_external_trigger.py`](18_agent_external_trigger.py) | An agent pauses at an approval `Gate`; a **separate** trigger (webhook/cron) resumes it later, sharing only a `FileStore` — the cross-process serverless pattern. | `OPENAI_API_KEY` |
| [`20_observability.py`](20_observability.py) | The step boundary (1.4.4): a `Tracer` hook records an event timeline, a `PermissionPolicy` gates a `FINANCIAL` tool for approval (then resumes), and the agent's `logging` is routed to a handler. | `OPENAI_API_KEY` |

---

## Debug

| File | What it shows | Keys needed |
|---|---|---|
| [`16_debug.py`](16_debug.py) | How to inspect what's happening inside Chain (`history`), Pool (`status` live + `history`), and Agent (`verbose=1`). Essential for production troubleshooting. | `ANTHROPIC_API_KEY` `PERPLEXITY_API_KEY` |

---

## Quick reference

```python
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.chain  import Chain
from yait_aichain.pool   import Pool, DONE, FAILED
from yait_aichain.agent  import Agent
from yait_aichain.tools  import convertToMD, MCPTools, Wait, Gate
from yait_aichain.state  import FileStore, SuspendedResult

# Skill — one prompt, one model
skill = Skill(model=Model("claude-sonnet-4-6"), input={...})
result = skill.run(variables={"key": "value"})

# Chain — sequential steps
chain = Chain(steps=[skill_a, skill_b])
result = chain.run(variables={...})
print(chain.history)          # full audit trail

# Pool — parallel execution
pool = Pool(skill, items=[{"topic": t} for t in topics], max_flows=10)
results = pool.run()
print(pool.status)            # {PENDING: 0, RUNNING: 0, DONE: 5, FAILED: 0}

# Agent — autonomous
agent = Agent(orchestrator=Model("claude-sonnet-4-6"), tools=[...], max_steps=10)
result = agent.run("Research the top 3 vector databases.")
print(result.output, result.steps_taken, result.tokens_used)

# Suspend / resume — pause until an external signal (human, webhook, cron)
chain = Chain(steps=[draft, Wait(reason="approve?"), send], store=FileStore("runs/"))
result = chain.run(variables={...})
if isinstance(result, SuspendedResult):
    # ...later, possibly in another process sharing the same store...
    chain.resume(result.run_id, signal={"reply": "approved text"})
```
