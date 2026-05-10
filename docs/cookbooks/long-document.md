# Cookbook: Long-document generation

> **Skeleton** — full content coming in a future release.

A dynamically-built `Chain` that generates a multi-section document one section at a time, using `SectionContextTool` to maintain a rolling context window and keep each section coherent with what came before.

## Primitives used

- `Chain` (dynamic step construction)
- `SectionContextTool`
- `Skill` (write, summarise — per section)
- `Agent` (optional — for complex sections requiring research)
- `Model`

## Key decisions

- Section list shape (`id`, `title`, `plan`, `runner`)
- Skill vs Agent runner per section
- Summary prompt design (controls rolling context quality)
- Post-run assembly of `{sid}_content` variables into final document

## See also

- [section_context](../tools-reference/section-context.md)
- [Chain primitives](../primitives/chain.md)
- [Agent as a chain step](../agents/agent-as-chain-step.md)
