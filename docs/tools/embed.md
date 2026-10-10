## One tool, five providers

```python
from yait_aichain.tools import embed

vectors = embed("openai", model="text-embedding-3-small").run(["a text", "another"],
                                                              {"input_type": "document"})
vectors.embeddings[0][:3], vectors.dimensions, vectors.total_tokens
str(vectors)        # what a model reads: "2 vector(s) of 1536 dimensions from openai …"
```

`embed(service, model=…)` makes the tool — `openai`, `cohere`, `voyage`,
`google` or `qwen`. **The model is a setting, not an option:** vectors from two
models cannot be compared, so a store is filled by one, and the tool is fixed
to it. `run()` returns an `EmbeddingResult` — the vectors, the model, the
dimensions, the tokens — whose text is a one-line summary: an agent, or a
Skill prompt that substitutes it, reads that and never the numbers.

An agent rarely needs this tool: it reaches embeddings through a vector store's
search. It is a tool so that code, a chain and an agent call it alike.
`Embedding("openai/text-embedding-3-small")` is the factory it replaced and
still works.
