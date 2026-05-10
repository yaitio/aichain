# Cookbook: RAG (retrieval-augmented generation)

> **Skeleton** — full content coming in a future release.

A `Chain` or `Agent` pattern that grounds generation in retrieved source material: search → fetch → chunk/extract → synthesise with citations.

## Primitives used

- `Chain` or `Agent` (agile mode for adaptive retrieval)
- `BraveSearchTool` or `PerplexitySearchTool` or `SerpApiTool`
- `MarkItDownTool`
- `Skill` (extraction, synthesis, citation formatting)
- `Model`

## Key decisions

- Single-hop (search → synthesise) vs multi-hop (iterative search driven by reflection)
- Source freshness (`search_recency_filter`, `freshness`, `tbs`)
- Domain restriction for authoritative sources
- Citation format in the final output (inline vs footnotes)
- `AgentMemory` to accumulate source material across steps

## See also

- [Research agent pattern](research-agent.md)
- [perplexity_search](../tools-reference/perplexity-search.md)
- [markitdown](../tools-reference/markitdown.md)
- [Agent memory](../agents/memory.md)
