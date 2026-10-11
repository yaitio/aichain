# Migrating to 4.0

4.0 has no compatibility layer. Every name and path that 3.x kept alive for
older code is gone, so the library has one way to do each thing and the
documentation describes only that one. An old name does not quietly work: it
raises, and the error names what to write instead — so the fastest migration
is to run your code and follow the errors.

```text
ImportError: cannot import name 'VectorDB' from 'yait_aichain.tools' —
a vector store is `from yait_aichain.utilities import Vectors` — Vectors(backend, collection=…, description=…, embedder=…)
```

Below is every change, so you can do it by search and replace before running
anything.

---

## Images are a tool, never a Skill

A `Skill` asked for an image output raises. A Skill makes text; an image model
is a tool.

| 3.x | 4.0 |
|---|---|
| `Skill(Model("gpt-image-1.5"), prompt=p, output={"modalities": ["image"], "format": {"type": "image", **opts}}).run()` | `image("openai", model="gpt-image-1.5").run(p, options=opts)` |
| an edit: the input image as a part of the Skill's message | `image(...).run(prompt, images=["photo.png"])` |
| `out["base64"]` | `out["images"][0]["base64"]`; `output_dir=` on the tool saves it, and `["path"]` says where |

The tool takes the same options — `size`, `aspect_ratio`, `quality`, `seed`, … —
and reports `cost` per picture where the provider reports no usage.
→ [Images](../tools/media.md)

## Utilities replace the vector, folder, MCP and REST tools

| 3.x | 4.0 |
|---|---|
| `store = VectorDB("qdrant", "docs", embedder=…)` | `kb = Vectors("qdrant", collection="docs", description="what it holds", embedder=…)` |
| `vectorUpsert(store).run(records)` | `kb.upsert(records)` |
| `vectorFetch(store).run(ids)`, `vectorDelete(store).run(ids)` | `kb.fetch(ids)`, `kb.delete(ids)` |
| `vectorQuery(store).run(q, {"n": 5})` | `kb.query.run(q, {"n": 5})` — and `Agent(model, tools=[kb.query])` |
| `vectorChunk(max_chars=800)` | `chunk(max_chars=800)` |
| `LocalTools(folder)` | `Shell(folder).tools` |
| `localBrowse(folder)`, `localRead(folder)`, … | `Shell(folder).browse`, `.read`, `.write`, `.run` |
| `MCPTools(server)` | `MCP(server).tools` |
| `RestApiTool(url=…, …)` | `REST(base_url, description=…).endpoint(name, method, path, description=…)` |

`Vectors` and `REST` take a required `description`: it is what an agent's
search or endpoint tool tells the model it reaches. A record given to `upsert`
needs an `id`; `chunk()` pieces have none, so number them:
`[{"id": f"doc-{i}", **p} for i, p in enumerate(pieces)]`.
→ [Utilities](../utilities/index.md)

## Tools by service name

| 3.x | 4.0 |
|---|---|
| `Embedding("openai/text-embedding-3-small")` | `embed("openai", model="text-embedding-3-small")` |
| `OpenAIEmbedder(…)`, `CohereEmbedder(…)`, … | `embed(service, model=…)` |
| `Reranker("rerank-v3.5")` | `rerank("cohere", model="rerank-v3.5")` |
| `TTS("openai")`, `STT("openai")` | `speech("openai")`, `transcribe("openai")` |
| `PerplexitySearchTool()`, `BraveSearchTool()` | `search("perplexity")`, `search("brave")` |
| `tool.search(query)` on a search tool | `tool.run(query)` |

## Import paths

| 3.x | 4.0 |
|---|---|
| `yait_aichain.tools.vectordb` | `yait_aichain.utilities` (`Vectors`) |
| `yait_aichain.tools.local` | `yait_aichain.utilities` (`Shell`) |
| `yait_aichain.tools.mcp` | `yait_aichain.utilities` (`MCP`) |
| `yait_aichain.tools.rest_api` | `yait_aichain.utilities` (`REST`) |
| `yait_aichain.tools.embedding` | `yait_aichain.tools` (`embed`) |
| `yait_aichain.tools.reranking` | `yait_aichain.tools` (`rerank`) |
| `yait_aichain.tools.convert` | `yait_aichain.tools` (`speech`, `transcribe`); a page by URL is `fetch("local")`; file conversion is not part of the library |

## Renamed

| 3.x | 4.0 |
|---|---|
| `convertToSpeech` (the speech tools' base) | `Speech` |
| `convertToText` (the transcription tools' base) | `Transcription` |
| `VectorChunkTool` | `Chunk` |
| image option `output_compression` | `compression` |
| image options `input_fidelity`, `reference_fidelity` | `fidelity` — how much of the original survives, 0 to 1 |
| a `Hook`'s `step_started` / `step_ended` for an agent's tool calls | `tool_call_started` / `tool_call_ended` (`step.*` is a Chain's step) |

An old image option is now an unknown key: it is reported as one, and the
request goes without it.

## Shell, MCP and REST tools ask before they act

Every `Shell`, MCP and REST tool used to inherit the default risk class,
`write`, which a policy allows — so with `permissions=PermissionPolicy()` on,
an agent could still run any shell command, call any tool a server declared,
or `DELETE` through an API, unasked.

| 3.x | 4.0 |
|---|---|
| `Shell(...).run` is `write`: allowed | `privileged`: needs approval |
| `Shell(...).browse`, `.read` are `write` | `read` |
| every MCP tool is `write` | from the server's annotations; `privileged` when there are none |
| every REST endpoint is `write` | by method: `GET` `read`, `POST`/`PUT`/`PATCH` `external` (approval), `DELETE` `destructive` (denied) |

An agent with no `permissions=` is unchanged — it runs every tool, as before.
With a policy, a `run` or an unannotated MCP tool is now refused unless
`approve=` is attached. To keep the old behaviour on purpose, say so:
`PermissionPolicy({"privileged": "allow"})`, or
`MCP(server, risk={"tool_name": "write"})` or `api.endpoint(..., risk="write")`
for one tool.
→ [Permissions](../control/permissions.md)

## Saved chains

A chain saved by 3.x names each tool by its class path. A path that moved —
`yait_aichain.tools.vectordb…`, `yait_aichain.tools.local…` — no longer loads,
and the error names the new place. Load the file with 3.10, or rebuild the
chain in code, and save it again with 4.0.

## What did not change

Models, Skills for text, Chains, Pools and Agents, `@tool`, the factories that
already existed in 3.x — `search`, `fetch`, `decide`, `chunk`, `embed`,
`rerank`, `speech`, `transcribe`, `image`, `wait`, `gate` — results,
budgets, permissions, events, pausing and eval. Code written against the 3.x
documentation since 3.9 needs none of this page.
