## One tool, three providers

```python
from yait_aichain.tools import rerank

top = rerank("cohere", model="rerank-v3.5").run(docs, {"query": "how does caching work?", "top_n": 5})
top[0]      # {"index", "text", "score", …} — the best first; a record's own fields are kept
```

`rerank(service, model=…)` makes the tool — `cohere`, `voyage` or `qwen`. It
takes documents as texts or as records with a `text` field — what a vector
store's search returns — and puts them in order of how well they answer
`query`. Thirty candidates in, the five that answer the question out: an agent
can use it after a search of its own. `Reranker("rerank-v3.5")` is the factory
it replaced and still works.
