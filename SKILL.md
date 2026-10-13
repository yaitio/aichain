---
name: yait-aichain
description: Build AI features in Python — an LLM call, a multi-step pipeline, the same work over many items, or an agent that uses tools (web search, page fetching, your own functions, any REST API or MCP server, a working folder), asks a human before risky actions, stays within a money budget, or pauses and resumes — on any model provider (OpenAI, Anthropic, Google, DeepSeek, xAI, Qwen, a local server), switching models by name. Use when the user asks to create, build, scaffold or fix an agent, bot, assistant, summariser, classifier, extractor or AI workflow in Python and the project does not already use another agent framework, and whenever code imports yait_aichain. Not for TypeScript, and not for projects built on LangChain, LlamaIndex, CrewAI or the OpenAI Agents SDK unless the user asks to move off them.
---

# Building with yait-aichain

Read `llms.txt` next to this file before writing code. It holds the real
constructor signatures, one working example per primitive, and the rules that
agents most often get wrong. Names from older tutorials — `orchestrator=`,
`max_steps=`, `VectorDB`, `Embedding(...)`, `LocalTools` — do not exist and
raise.

When the package is installed, the documentation of that exact version is
inside it — `python -c "import yait_aichain; print(yait_aichain.docs_dir())"`
prints the folder, with `llms.txt` and every page. Prefer it to anything you
remember: it matches the code that will run.

## Pick the primitive

| The task | Use |
|---|---|
| One model call — summarise, extract, classify, translate | `Skill` |
| Known steps in a fixed order, each feeding the next | `Chain` |
| The same work over many inputs | `Pool` |
| The next step depends on what the last one returned | `Agent` with tools |
| A step that is plain code | a function — a chain takes it as a step |
| Something the model may call | `@tool` on a function; a built-in tool by name |
| An outside API, server or folder | a utility — `REST`, `MCP`, `Shell`, `Vectors` — and give the agent its tools |
| A picture | `image(service, model=…)`, a tool, not a `Skill` |
| A run that must stop and continue later, in another process | `wait()` / `gate(tool)` and `store=FileStore(...)` |

Prefer the smallest that fits. A Chain whose steps you can list is cheaper,
faster and more predictable than an Agent that rediscovers them.

## Tools and utilities

- **Built-in tools are made by service name**: `search("exa")`,
  `fetch("local")`, `chunk(max_chars=500)`, `embed("openai", model=…)`,
  `rerank("cohere", model=…)`, `decide("openai", questions=…)`,
  `speech(…)`, `transcribe(…)`, `image(…)`. What a tool *is* goes in the
  factory; what one call asks goes in `.run(input, options)`. A setting never
  appears in both.
- **Your own function** is a tool with `@tool`; its docstring's first line is
  the description and its parameters the schema. Call it as
  `fn.run(arg=…)`.
- **A utility** is an object whose methods your code calls and some of whose
  members are tools: `REST(base_url, description=…).endpoint(…)`,
  `MCP(server).tools`, `Shell(folder).browse / .read / .write / .run`,
  `Vectors(backend, collection=…, description=…).query`. Give an agent only
  the tools it should have — `[sh.browse, sh.read]` is a read-only agent.

## Workflow

1. **Choose models by name.** `Model("claude-haiku-5-5")`,
   `Model("gpt-6-luna")`. Keys come from the environment; never put a key in
   code or in a saved file.
2. **Write the Skills.** `Skill(model, prompt="… {var}")`, or `input=` for
   multi-message prompts. Ask for `output={"format": {"type": "json_schema",
   "schema": …}}` when code will read the result.
3. **Wire them.** `Chain(steps=[(skill, "key"), (fn, "key2"), (tool, "key3",
   {"arg": "Template {var}"}), …])`, `Pool(runner, items=[…])`, or
   `Agent(model, tools=[…], stop_when=[…])`.
4. **Bound it.** Every Agent gets `stop_when` with a ceiling
   (`step_count(n)`), and a `check(fn)` when success can be verified in code.
   Add `max_cost=` when money matters — one budget across every step.
5. **Gate side effects.** Give risky tools a `risk` class and the agent
   `permissions=PermissionPolicy({...})` plus an `approve=` callable.
6. **Observe.** Print `result.stopped_by`, `result.cost`, `chain.history` or
   `pool.status`; attach `hooks=[Tracer()]` when debugging.

## Checklist before you finish

- [ ] No `orchestrator=`, `max_steps=`, `max_attempts=`, `memory=`, `executors=`.
- [ ] Each `Skill` has exactly one of `prompt=` / `input=`, and messages use
      `parts`, never `content`.
- [ ] Chain, Pool and Agent results are read through `.output` (and a chain's
      steps as `result["key"]`); only `skill.run()` returns the bare value.
- [ ] Every import is one `llms.txt` shows (`Tracer` and `Budget` come from
      `yait_aichain`).
- [ ] Every `Agent` has a ceiling in `stop_when`.
- [ ] Tools are called with `.run(...)`; a utility's methods (`kb.upsert(…)`)
      are called directly and are never given to an agent.
- [ ] A Chain tool step's arguments match the tool's parameter names, or are
      mapped with the third tuple element — a variable's name or a
      `"{template}"`.
- [ ] If you drive `agent.step()` yourself, every tool call gets
      `tool_result_turn(call.id, result)` before the next `step()`.
- [ ] A pause that must survive the process has a `store=FileStore(...)`.
- [ ] No API key appears in the code.
