# Cookbook: Multi-provider routing

> **Skeleton** — full content coming in a future release.

Using multiple models within a single `Chain` or `Agent` — a fast/cheap model for early steps (classification, extraction, outline) and a capable model for the final synthesis or reasoning-heavy step.

## Primitives used

- `Chain` with mixed-model `Skill` steps
- `Model` (multiple instances, different providers)
- Optional: `Agent` as a final step using a premium orchestrator

## Key decisions

- Cost/quality split point (which steps warrant the expensive model)
- Reasoning option (`"low"` / `"medium"` / `"high"`) as a dial instead of model switching
- Consistent output format across models (JSON schema for structured handoff)
- Fallback strategy when the cheap model produces low-confidence output

## See also

- [Model factory](../primitives/models.md)
- [Universal reasoning](../primitives/models.md#universal-reasoning)
- [Chain primitives](../primitives/chain.md)
