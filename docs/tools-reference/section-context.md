# `section_context` — `SectionContextTool`

A **zero-API-call** state-management tool used by the sectional document-generation pattern. It pops the next section off a queue, writes its metadata into well-known variable names, and rotates a **rolling window of the last two section summaries** — so every section-writing step has fresh context without bloating the prompt.

```python
(SectionContextTool(), "section_ctx", {}),
```

Because the tool returns a `dict`, the Chain merges every returned key into `accumulated` — the `output_key` is effectively unused.

---

## Role in the pipeline

When a Chain is built dynamically from a list of sections:

1. The section list is serialised once into an initial variable `section_queue` (JSON).
2. Before each section's write steps, a `SectionContextTool` step pops the front item.
3. Its output is merged into `accumulated`, exposing `{current_section_*}` to the downstream write skill/agent.
4. After the section is written and summarised, the next `SectionContextTool` call rotates the rolling window.

No API calls. No external state. No glue code.

---

## Input parameters

| Name | Type | Required | Notes |
|---|---|---|---|
| `section_queue` | `string` | ✓ | JSON-encoded list of section objects (see below). |
| `current_section_actual_summary` | `string` | | Summary of the section just written. Empty on first call. |
| `prev_section_summary` | `string` | | From the previous call; rotated into `two_sections_ago_summary`. |
| `two_sections_ago_summary` | `string` | | Accepted for symmetry; dropped this call. |

### Section object shape

Each object in the queue must have:

```json
{
  "id":       "situation",
  "title":    "Situation Analysis",
  "plan":     "Current market and competitive context.",
  "position": 1,
  "runner":   "skill"
}
```

Missing keys → `ValueError`. Empty queue → `RuntimeError` (fail-fast: an accidental extra call is caught immediately, not silently).

---

## Output — variables merged into `accumulated`

| Key | Meaning |
|---|---|
| `current_section_id` | Machine identifier for the current section. |
| `current_section_title` | Human-readable title. |
| `current_section_plan` | Planned summary / brief. |
| `current_section_position` | Integer position in the final document. |
| `current_section_runner` | `"skill"` or `"agent"`. |
| `current_section_sources` | Reset to `""`. |
| `current_section_actual_summary` | Reset to `""`. |
| `prev_section_summary` | Rotated in from the just-completed summary. |
| `two_sections_ago_summary` | Rotated in from the prior `prev_section_summary`. |
| `recent_summaries` | Formatted rolling-context block (see below). |
| `section_queue` | Updated queue (current item removed). |

### `recent_summaries` format

```
ROLLING CONTEXT (recently written sections)
===========================================
Two sections ago:
<two_ago summary>

Previous section:
<prev summary>
```

Empty string when both summaries are empty (first call, typically).

---

## Rotation logic

On each call:

```
two_sections_ago_summary   ← old prev_section_summary
prev_section_summary       ← old current_section_actual_summary
recent_summaries           ← formatted block derived from the above
current_section_actual_summary ← "" (reset for new section)
```

Only the last two summaries are ever kept — the window is bounded regardless of document length.

---

## Usage

### Standalone

```python
import json
from tools import SectionContextTool

sections = [
    {"id": "situation",   "title": "Situation Analysis",
     "plan": "Current market and competitive context.",
     "position": 1, "runner": "skill"},
    {"id": "competitive", "title": "Competitive Analysis",
     "plan": "Key players, positioning, white spots.",
     "position": 2, "runner": "agent"},
]

tool   = SectionContextTool()
result = tool(section_queue=json.dumps(sections))

print(result.output["current_section_title"])
# "Situation Analysis"
```

### In a sectional Chain

```python
from chain import Chain
from tools import SectionContextTool

def build_chain_for_sections(sections, write_skill, summarise_skill):
    steps = []
    for _ in sections:
        steps.append((SectionContextTool(), "_"))                    # merges section ctx
        steps.append((write_skill,          "section_content"))      # uses {current_section_*}
        steps.append((summarise_skill,      "current_section_actual_summary"))
    return Chain(steps=steps, variables={"section_queue": json.dumps(sections)})
```

Each iteration:

1. `SectionContextTool` pops + rotates.
2. `write_skill` references `{current_section_title}`, `{current_section_plan}`, `{recent_summaries}`, etc.
3. `summarise_skill` writes the new summary into `current_section_actual_summary`, which the next `SectionContextTool` call rotates into `prev_section_summary`.

### Post-run assembly

After `chain.run()`, every section's output lives under its own key (e.g. `situation_content`, `competitive_content`). Iterate `chain.accumulated` to stitch the final document together.

---

## Notes

- Pure Python — zero dependencies, zero API calls.
- Deliberate fail-fast on empty queue: prevents silent overwrites with stale state.
- Pairs naturally with [Chain](../primitives/chain.md)'s dict-merge behaviour (tool output → `accumulated.update(output)`).

---

## See also

- [`primitives/chain.md`](../primitives/chain.md) — sectional generation pattern and dict-merge semantics.
- [`primitives/tools.md`](../primitives/tools.md) — custom tools that return dicts.
