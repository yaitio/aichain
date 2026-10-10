# Chunk, embed, rerank

Answering from your own documents — a manual, a policy, last year's tickets —
is four steps, and the library has a tool for each part that needs one:

```text
documents ──chunk──▶ pieces ──embed + upsert──▶ a vector store
question ──query──▶ thirty candidates ──rerank──▶ the five that answer it ──▶ a Skill writes the answer
```

```python
from yait_aichain.tools import chunk, embed, rerank
from yait_aichain.utilities import Vectors

kb = Vectors("qdrant", collection="manual", description="the product manual",
             embedder=embed("openai", model="text-embedding-3-small"))
pieces = chunk(max_chars=800).run(manual_text)
kb.create(dimension=1536)
kb.upsert([{"id": f"p{i}", **p} for i, p in enumerate(pieces)])

candidates = kb.query.run("how do I reset the device?", {"n": 30})
top = rerank("cohere", model="rerank-v3.5").run(candidates, {"query": "how do I reset the device?",
                                                             "top_n": 5})
```

→ Every parameter: [API: embed](../api/tools/embed.md) · [API: rerank](../api/tools/rerank.md) · [API: Vectors and chunk](../api/utilities/vectors.md)

---

## Chunk: pieces a search can find

A whole manual is too long to embed as one vector and too long to put in a
prompt. `chunk(max_chars=800)` cuts a text into pieces of at most that size,
with Markdown's headings, code blocks and tables handled on their own terms,
and records for each piece the headings it sits under. Consecutive pieces
overlap — a tenth of the size by default — so a sentence cut at a boundary is
whole in one of them, and neighbours under the same heading are merged when
they fit together. The sizes are settings of the tool, not of a call: a store
is filled with pieces cut one way. It runs locally and costs nothing.

Each piece comes back as `{"text", "chars", "metadata"}` — with no `id`. A
vector store needs one per record, and only you know what makes an id stable
across re-indexing: a file name and a position, a section's slug. Give it one
before `upsert`.

## Embed: meaning as a vector

`embed(service, model=…)` turns texts into vectors — `openai`, `cohere`,
`voyage`, `google` or `qwen`. **The model is fixed when the tool is made, not
chosen per call:** vectors from two models cannot be compared, so a store is
filled by one model and searched with the same one. `input_type` per call says
whether the texts are documents or queries, which some providers encode
differently.

What a model reads of the result is one line — how many vectors, how many
dimensions, how many tokens — never the numbers, which it can do nothing with.
An agent rarely needs this tool directly; it reaches embeddings through a
store's `query`.

## Store and search: Vectors

`Vectors` is the database — Chroma, Pinecone or Qdrant — as a utility: your
code creates the collection and writes to it, and `kb.query` is the one tool
an agent gets. Its `description` is required, because it is what the search
tool tells a model it searches: an agent with two knowledge bases cannot tell
them apart without it. → [Utilities](../utilities/index.md)

A score means one thing on every backend: a similarity from 0 to 1, higher
being closer, with the backend's own number kept beside it as `raw_score`. The
one exception is said rather than papered over: a dot-product score is
unbounded and passes through as it is, since mapping it onto 0–1 would need
the vectors' magnitudes.

## Rerank: the five that answer it

A vector search finds what is *near* the question, which is not the same as
what *answers* it. A reranker reads each candidate against the question and
puts them in order. Thirty candidates in, five out: the cheap search casts the
net, the reranker chooses. `rerank(service, model=…)` takes `cohere`, `voyage`
or `qwen`, documents as texts or as the records a store's search returns, and
keeps a record's own fields on what it returns.

## As a pipeline

The question-to-answer half is a Chain, with templates filling each tool's
arguments:

```python
reranker = rerank("cohere", model="rerank-v3.5")
answer = Chain(steps=[
    (kb.query,  "candidates", {"input": "{question}", "options": {"n": 30}}),
    (reranker,  "context",    {"input": "{candidates}", "options": {"query": "{question}", "top_n": 5}}),
    (Skill(Model("claude-haiku-5-5"),
           prompt="Answer from these passages only:\n{context}\n\nQuestion: {question}"), "answer"),
])
answer.run(variables={"question": "How do I reset the device?"})
```

Or give an agent `kb.query` and the reranker as tools, and let it search again
when the first results were not enough.
