# API: decide

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../../tools/decide.md).

<!-- g:tool -->
## `decide()`

```python
decide(
    service: str,
    *,
    questions: dict,
    model: str | None = None,
    name: str | None = None,
    description: str | None = None,
    api_key: str | None = None,
) -> yait_aichain.tools.decision._base.Decision
```

A decision tool for *service* — one of `SERVICES` — answering *questions*.

The questions are fixed here, for the life of the tool: an agent passes
only the text. `model` defaults to the provider's; `name` to the
provider's tool name and `description` to one written from the
questions; `api_key` to the provider's environment variable.

```python
triage = decide("openai", name="triage", questions={
    "urgent": {"type": "predicate", "instructions": "Does this convey urgency?"},
})
triage.run("Help! My payouts have been failing for 3 days.")["answers"]["urgent"]
```

---

## The classes behind it

What the factory returns, with the schema a model is shown.

### `decisionOpenAI`

Typed questions about a text — and images — answered with probabilities, by OpenAI's Decisions API.

| | |
|---|---|
| Import | `from yait_aichain.tools import decisionOpenAI` |
| Risk class | `read` |
| Key | `OPENAI_API_KEY` |

```python
decisionOpenAI(
    questions: dict,
    *,
    name: str | None = None,
    description: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
)
```

The call schema is built from the constructor arguments, so it is whatever the caller declares.

### `decisionTypeSafe`

Typed questions about a text, answered with probabilities, by TypeSafe's Jev.

| | |
|---|---|
| Import | `from yait_aichain.tools import decisionTypeSafe` |
| Risk class | `read` |
| Key | `TYPESAFE_API_KEY` |

```python
decisionTypeSafe(
    questions: dict,
    *,
    name: str | None = None,
    description: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
)
```

The call schema is built from the constructor arguments, so it is whatever the caller declares.
<!-- /g:tool -->
