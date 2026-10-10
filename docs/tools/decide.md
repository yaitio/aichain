## What a decision tool is

A decision model writes no prose. It is given evidence — a text, a record, an
image — and questions fixed in advance, and answers each with a number. Two
providers sell exactly this and differ in every wire detail, so the library
holds one format of questions and answers and each provider tool translates
it, the way the search tools work.

| Tool | Provider | Takes | Input price | Key |
|---|---|---|---|---|
| `decisionOpenAI` | OpenAI Decisions API, `gpt-6-luna` (beta) | text, images | $0.10 / 1M tokens | `OPENAI_API_KEY` |
| `decisionTypeSafe` | TypeSafe Jev | text, structured state | $0.042 / 1M tokens | `TYPESAFE_API_KEY` |

Output is free on both. Both take up to 255 options and 2 to 10 levels.

## Questions

A map of name → question; the answer comes back under the same name.

| Type | Asks | Fields |
|---|---|---|
| `predicate` | is this statement true? | `instructions` |
| `choice` | which of these options? | `instructions`, `choices`: option → description (or `None`) |
| `score` | where on these ordered levels? | `instructions`, `levels`: lowest first, each a label or `{"label", "description"}` |

```python
from yait_aichain.agent import Agent
from yait_aichain.models import Model
from yait_aichain.tools import decide

triage = decide("openai", name="triage", questions={      # or "typesafe" — same arguments
    "urgent": {"type": "predicate", "instructions": "Does this convey urgency?"},
    "team":   {"type": "choice", "instructions": "Which team should handle it?",
               "choices": {"billing": "Payments, refunds", "technical": "Bugs, outages", "other": None}},
    "anger":  {"type": "score", "instructions": "How frustrated is the customer?",
               "levels": ["Calm", "Frustrated", "Very angry"]},
})

agent = Agent(Model("gpt-6-luna"), tools=[triage],
              instructions="Call triage first; escalate only if urgent is above 0.8.")
```

A malformed question is refused when the tool is made, by name — before any
network.

## Answers

The same shape from every provider:

```python
out = triage.run("Help! My payouts have been failing for 3 days.")
# {"answers": {
#    "urgent": {"type": "predicate", "probability": 0.96},
#    "team":   {"type": "choice", "choice": "billing", "confidence": 0.91,
#               "probabilities": {"billing": 0.94, "technical": 0.05, "other": 0.01}},
#    "anger":  {"type": "score", "score": 1.0, "confidence": 1.0,
#               "probabilities": {"Calm": 0.0, "Frustrated": 1.0, "Very angry": 0.0}}},
#  "model": "gpt-6-luna", "usage": {"input_tokens": 395, "output_tokens": 0}, "cost": 3.9e-05,
#  "cost_basis": "input tokens reported by OpenAI, at the published rate (…)"}
str(out)
# urgent: yes with probability 0.96
# team: billing (confidence 0.91; billing 0.94, technical 0.05, other 0.01)
# anger: score 1.00 (confidence 1.00; Calm 0.00, Frustrated 1.00, Very angry 0.00)
```

`decide(service, questions=…)` makes the tool — `openai` or `typesafe` — with
`model`, `name`, `description` and `api_key`; the questions are fixed there, so
an agent passes only the text. `run()` returns a `Record`: the dict for code,
and one line per answer for a model — which is what an agent sees, and what a
Skill prompt gets when it substitutes the result.

- **A score** is the probability-weighted average of level indices from 0, so
  it can fall between levels; `probabilities` are keyed by level label.
- **The numbers are untouched.** `0.96` is not turned into `True`: the
  threshold is the caller's, who knows what a wrong answer costs. Set it on
  labelled examples from your own data.
- **`model`** is the version that answered. Pin Jev (`model="jev-1.13.0"`) when
  thresholds were tuned against it — its aliases move when a release ships.
- **Money is reported, not budgeted.** `cost` counts input tokens at each
  provider's price; no tool is held to `max_cost` yet.

## The agent sees one argument

The model calls the tool with `state`, the text to judge; the questions are
fixed when the tool is made, so a model cannot write one badly. Its
description is written from the questions, so it knows what it will learn.
Name each tool by what it decides — `triage`, `is_duplicate`.

## What differs between providers

- **Images** — `decisionOpenAI` only, passed from code:
  `run(state=..., images=["photo.png", b"...", "data:image/png;base64,..."])`.
  The endpoint takes base64 data URLs, not hosted URLs; a hosted URL is
  refused before the call. `decisionTypeSafe` refuses images by name.
- **Structured state** — a dict or a list from code reaches Jev as structure
  and OpenAI as JSON text.
- **Dependent questions** — answered in one call, questions cannot see each
  other's answers. For a question that depends on another, make two calls.

## A new provider

Subclass `Decision`, set `name`, `_ENV_KEY`, `_PROVIDER`, `_DEFAULT_MODEL`,
`_PRICE_PER_INPUT_TOKEN` and `images`, and implement
`_ask(state, images) -> (answers, model, usage)` returning answers in the
common format. The checks, the description, images, cost and errors are the
base's.
