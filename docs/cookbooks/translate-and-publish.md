# Cookbook: Translate and publish

> **Skeleton** — full content coming in a future release.

A `Chain` that takes source copy, translates it with DeepL, refines tone and style, then publishes to one or more social platforms via Late — with per-platform content overrides.

## Primitives used

- `Chain`
- `Skill` (copywriting, platform adaptation)
- `DeepLTranslateTool`
- `DeepLRephraseTool`
- `LateAccountsTool`
- `LatePublishTool`
- `Model`

## Key decisions

- Translate-then-rephrase vs single-pass (language support matrix)
- Per-platform `custom_content` generation (Twitter character limit, LinkedIn tone)
- Scheduling vs publish-now
- Draft-first workflow for human review

## See also

- [deepl](../tools-reference/deepl.md)
- [late](../tools-reference/late.md)
- [Chain primitives](../primitives/chain.md)
