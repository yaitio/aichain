---
name: yait-aichain
description: Build LLM pipelines, tool-using agents and parallel fan-outs in Python with yait-aichain (Skill, Chain, Pool, Agent, Tool) — provider-independent. Use when writing, fixing or reviewing code that imports yait_aichain.
---

# Building with yait-aichain

Read `llms.txt` in this repository before writing code. It holds the real
constructor signatures, one working example per primitive, and the rules that
agents most often get wrong. The library changed its agent API in 2.0; code
from older tutorials (`orchestrator=`, `max_steps=`, `agent.resume()`) does not
run.

## Pick the primitive

| The task | Use |
|---|---|
| One model call — summarise, extract, classify, translate | `Skill` |
| Known steps in a fixed order, each feeding the next | `Chain` |
| The same work over many inputs | `Pool` |
| The next step depends on what the last one returned | `Agent` with tools |
| Something with side effects the model may call | a `Tool` subclass |
| A run that must stop and continue later, in another process | `Chain` with `Wait`/`Gate` and a `FileStore` |

Prefer the smallest that fits. A Chain whose steps you can list is cheaper,
faster and more predictable than an Agent that rediscovers them.

## Workflow

1. **Choose models by name.** `Model("claude-sonnet-4-6")`,
   `Model("gpt-5.4-mini")`. Keys come from the environment; never put a key in
   code or in a saved file.
2. **Write the Skills.** `Skill(model, prompt="… {var}")`, or `input=` for
   multi-message prompts. Ask for `output={"format": {"type": "json_schema",
   "schema": …}}` when code will read the result.
3. **Wire them.** `Chain(steps=[(skill, "key"), …])`, `Pool(runner,
   items=[…])`, or `Agent(model, tools=[…], stop_when=[…])`.
4. **Bound it.** Every Agent gets `stop_when` with a ceiling
   (`step_count(n)`), and a `check(fn)` when success can be verified in code.
   Add `max_cost=` when money matters.
5. **Gate side effects.** Give risky tools a `risk` class and the agent
   `permissions=PermissionPolicy({...})` plus an `approve=` callable.
6. **Observe.** Print `result.stopped_by`, `result.cost`, `chain.history` or
   `pool.status`; attach `hooks=[Tracer()]` when debugging.

## Checklist before you finish

- [ ] No `orchestrator=`, `max_steps=`, `max_attempts=`, `memory=`, `executors=`.
- [ ] Each `Skill` has exactly one of `prompt=` / `input=`, and messages use
      `parts`, never `content`.
- [ ] `chain.run()` is read as the **last** step's output; other steps come
      from `chain.accumulated["key"]`. `agent.run()` is an `AgentResult` —
      the answer is `.output`.
- [ ] Every import is one `llms.txt` shows (`Tracer` and `Budget` come from
      `yait_aichain`).
- [ ] Every `Agent` has a ceiling in `stop_when`.
- [ ] Tools are called as `tool(input=…, options={…})`.
- [ ] A Chain Tool step's inputs match the tool's `parameters` names, or are
      renamed with the third tuple element.
- [ ] If you drive `agent.step()` yourself, every tool call gets
      `tool_result_turn(call.id, result)` before the next `step()`.
- [ ] A pause that must survive the process uses `Chain` + `store=FileStore(...)`.
- [ ] No API key appears in the code.
