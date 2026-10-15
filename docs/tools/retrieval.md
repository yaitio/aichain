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

How a text is cut. It is read as Markdown when a line starts with `#`, or it
holds a code fence, a table separator row or emphasis around a word; otherwise
as plain text. Headings are never pieces of their own: they become the
`metadata.headings` of what follows, and a heading closes the ones at its level
and below. A code block or a table that fits is kept whole; one that does not
is split by lines or rows, with the opening fence, or the header and separator
rows, repeated in every piece. Text is split on the first of a blank line, a
line break, `. `, `。！？` and a space that cuts it, and by characters only for
one word longer than the limit. Overlap joins text to text under the same
headings — never a code block or a table — and starts at a word boundary.
An empty text, or one of headings only, gives `[]`; an unclosed code fence is
read as text; `overlap_chars >= max_chars` is refused when the tool is made.

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

A list of any length is split into requests the provider accepts — by rows,
by estimated tokens and by bytes, whichever binds first — and the vectors come
back in the order of the texts. `batch_size` lowers the rows per request, for
rate limits. An empty text is refused before anything is sent.

What a model reads of the result is one line — how many vectors, how many
dimensions, how many tokens — never the numbers, which it can do nothing with.
An agent rarely needs this tool directly; it reaches embeddings through a
store's `query`.

What each provider does with the two settings and the one option:

| Service | Models | `dimensions` | `input_type` |
|---|---|---|---|
| `openai` | `text-embedding-3-small` (1536), `-3-large` (3072), `ada-002` (1536) | the `3-*` models; `ada-002` is fixed | ignored |
| `cohere` | `embed-v4.0`, `embed-english-v3.0`, `embed-multilingual-v3.0` (1024) | every model | `search_query` / `search_document`; none sends `search_document`, as the API requires one |
| `voyage` | `voyage-3-large`, `voyage-3`, `voyage-3-lite` (512), `voyage-code-3`, `voyage-finance-2`, `voyage-law-2` | every model | passed through; none is omitted |
| `google` | `gemini-embedding-001` (768), `gemini-embedding-exp-03-07` (3072), `text-embedding-004` (768) | `gemini-embedding-001` and `-exp-03-07` | `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT`, sent to `gemini-embedding-001` and `text-embedding-004` only |
| `qwen` | `text-embedding-v4` (64–2048), `-v3` (512–8192), `-v2` (fixed 1536) | `v4` and `v3` | passed through; none is omitted |

`dimensions` on a fixed-length model raises when the tool is made, naming the
models that take it. Google reports no token counts, so its `total_tokens` is
`None`. Qwen's host follows `EmbeddingQwen(region=…)` or `DASHSCOPE_REGION` — `ap` (the
default), `us`, `cn` or `hk`.

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

The models each takes: `cohere` — `rerank-v3.5`, `rerank-english-v3.0`,
`rerank-multilingual-v3.0`; `voyage` — `rerank-2`, `rerank-2-lite`; `qwen` —
`gte-rerank`, `gte-rerank-v2`, with its host chosen by `RerankQwen(region=…)`
or `DASHSCOPE_REGION` as for embeddings. Cohere bills in search units, and
that is the number its result carries as `total_tokens`.

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
