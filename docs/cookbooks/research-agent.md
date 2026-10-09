# Cookbook: Research agent

> **Skeleton** — full content coming in a future release.

An `Agent` equipped with search and fetch tools that autonomously investigates a topic, reads source material, and produces a cited report.

## Primitives used

- `Agent` (agile mode)
- `PerplexitySearchTool` or `BraveSearchTool`
- `fetch("local")` or `fetch("firecrawl")`
- `Model` (orchestrator)

## Key decisions

- Search tool choice (snippets-first vs links-first)
- Rolling vs full memory
- Reflection budget (`max_steps`, `max_tokens`)
- Persona for citation style

## See also

- [Agent overview](../agents/overview.md)
- [Agent configuration](../agents/configuration.md)
- [perplexity_search](../tools-reference/search.md#searchperplexity)
- [brave_search](../tools-reference/search.md#searchbrave)
- [fetch](../tools-reference/fetch.md)
