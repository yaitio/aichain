# Build with your coding agent

You do not have to write the agents yourself. Claude Code, Codex or Cursor can
write them on aichain, and you review what they wrote. The library is built
for that: small enough to read in one go, with one skill that teaches it, and
a measurement of how well the teaching works.

---

## Install the skill

One plugin carries it for all three agents.

**Claude Code**

```text
/plugin marketplace add yaitio/aichain
/plugin install yait-aichain@yait
```

**Codex** — add the repository as a marketplace, then install `yait-aichain`
from it:

```bash
codex plugin marketplace add yaitio/aichain
```

**Cursor** — Customize → *From GitHub Repository* → `yaitio/aichain`, then
install `yait-aichain`.

**Without a plugin**, copy `plugin/skills/yait-aichain/` into
`~/.agents/skills/` — Codex and Cursor both read it — or into
`~/.claude/skills/` for Claude Code.

## What the agent gets

| File | Size | What it is |
|---|---|---|
| `SKILL.md` | 4 KB | when to use the library, which primitive for which task, a checklist before finishing |
| `llms.txt` | 26 KB | the whole contract: every constructor's signature, one working example per primitive, the rules agents most often get wrong |
| `recipes.md` | — | twenty worked programs, each passing its task's check in this repository |

The skill fires on the language of the task — "build an agent that…", "a
pipeline that summarises…" — in a Python project that does not already use
another agent framework, and whenever code imports `yait_aichain`.

Nothing in the plugin is written by hand. `.github/scripts/skill.py` generates it from
the files this repository tests, and CI fails when the copy drifts: a skill
that teaches a removed API is worse than no skill, because an agent trusts its
context over its weights.

When an agent writes a name the library does not have — `max_steps=` on an
`Agent`, `result.text` on its result, `from yait_aichain.hooks import Tracer` —
the error names the call that works. The wrong name is never accepted as an
alias: two spellings of one API leave the docs, the examples and the programs
agents write disagreeing about which one is real.

## How well it works

Measured on tasks the documents were not written against, each program run
offline with every model call scripted, and judged by what it built and ran —
not by what it printed about itself.

Tasks passed. The small model is `gpt-5.6-luna`, with one reply per attempt
and a task counted only when all three attempts pass; Claude Code could run
its program and fix it, one trial.

| Set | Small model, docs | Claude Code | Small model, no docs |
|---|---|---|---|
| 1 — 20 tasks across the primitives (2026-09-15) | 11 of 20 | 18 of 20 | 0 of 20 |
| 2 — 20 more, after `llms.txt` was extended (2026-09-15) | 15 of 20 | 19 of 20 | — |
| 3 — 11 on what 3.x added (2026-10-10) | 11 of 11 | 11 of 11 | — |

The third set is at its ceiling: it shows the skill teaches 3.x, and cannot
show how much better a better document would be. Codex and Cursor read the same
skill folder and have not been measured.

## Asking for an agent

Say what it must do and what it must never do; the agent picks the primitive.
What the skill makes it reach for, without being told:

- a `Chain` when the steps can be listed, an `Agent` only when the next step
  depends on the last;
- a ceiling in `stop_when` on every agent, and `max_cost=` when money matters;
- a `risk` class on a tool that writes, spends or deletes, and an approver in
  front of it;
- tools made by name — `search("exa")` — and only the tools a task needs:
  `[sh.browse, sh.read]`, not every tool of a folder.

What to check in review is the same list. Every primitive's result carries
`cost` and `usage`, so the first run says what the agent's program spends.
