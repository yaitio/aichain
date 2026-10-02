<!-- g:tool -->
# `convertToDecision`

Typed questions about a text, answered with probabilities, by TypeSafe's Jev.

| | |
|---|---|
| Import | `from yait_aichain.tools import convertToDecision` |
| Risk class | `read` |
| Key | `TYPESAFE_API_KEY` |

```python
convertToDecision(
    questions: dict,
    *,
    name: str | None = None,
    description: str | None = None,
    model: str = 'jev-latest',
    api_key: str | None = None,
)
```

The call schema is built from the constructor arguments, so it is whatever the caller declares.
<!-- /g:tool -->

## What it is, and why it is a tool

Jev, TypeSafe's System One model, is not a chat model. It writes no text and
holds no conversation: it answers typed questions about a **state** and returns
one answer per question, with a probability for every option.

| Question | Asks | Answer |
|---|---|---|
| `noul` | is this statement true? | `noul` — the probability of yes, 0 to 1 |
| `choice` | which of these options? (up to 255) | `choice`, `probabilities` per option, `confidence` |
| `score` | where on these ordered levels? (2 to 10) | `score` (can land between levels), `legend`, `probabilities`, `confidence` |

All questions are answered in one call, in parallel, against the same state.

Every specialised model here that does not converse — speech, transcription,
embeddings, reranking, search — is a tool, and so is this one. As a tool an
agent can call it: the questions are fixed when the tool is made and the
agent passes only the text, so the model cannot write a question badly.

```python
from yait_aichain.agent import Agent
from yait_aichain.models import Model
from yait_aichain.tools import convertToDecision

triage = convertToDecision(name="triage", questions={
    "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "team":   {"type": "choice", "instructions": "Which team should handle it?",
               "criteria": {"billing": "Payments, invoicing, refunds",
                            "technical": "Bugs, outages, integrations"}},
})

triage.run(state="Help! My payouts have been failing for 3 days.")
# {"answers": {"urgent": {"type": "noul", "noul": 0.95},
#              "team": {"type": "choice", "choice": "billing", "confidence": 0.83,
#                       "probabilities": {"billing": 0.89, "technical": 0.11}}},
#  "model": "jev-1.13.0", "usage": {"input_tokens": 358, "output_tokens": 54},
#  "cost": 1.5e-05}

agent = Agent(Model("gpt-6-luna"), tools=[triage],
              instructions="Call triage first; escalate only if urgent is above 0.8.")
```

- **The answers are untouched.** `0.95` is not turned into `True`; the
  threshold is the caller's, because only the caller knows what a wrong answer
  costs.
- **The model told the tool's description** is written from the questions, so
  it knows what it will learn by calling it.
- **A structured state** — a record, a chat log — can be passed from code as a
  dict or a list, and travels as structure.
- **Pin a version** (`model="jev-1.13.0"`) when confidence thresholds were tuned
  against it: the aliases `jev-latest` and `jev-preview` move when a release
  ships. The answer names the version that answered.
- **Money is reported, not budgeted.** `cost` counts input tokens at $0.042 per
  million; output is free. `max_cost` is counted inside `Skill` and no tool is
  held to it yet.
- Text only: describe an image in words before asking about it.
