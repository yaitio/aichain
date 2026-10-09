"""
tools.decision
==============

Typed questions about a text, answered with probabilities — one format of
questions and answers, a provider per tool, the way ``tools.search`` works.

Tools
-----
decisionOpenAI     OpenAI's Decisions API — text and images, gpt-6-luna.
decisionTypeSafe   TypeSafe's Jev — text and structured state.

``decide(service, questions=...)`` makes one by the provider's name and is the
way the documentation shows. ``Decision`` is the base; it holds the format, the
checks and the translation of answers, and a new provider is a subclass that
implements ``_ask``, and an entry in ``SERVICES``.
"""

from ._base    import Decision
from .openai   import decisionOpenAI
from .typesafe import decisionTypeSafe

#: Provider name → tool class, for :func:`decide`.
SERVICES = {
    "openai":   decisionOpenAI,
    "typesafe": decisionTypeSafe,
}


def decide(service: str, *, questions: dict, model: "str | None" = None, name: "str | None" = None,
           description: "str | None" = None, api_key: "str | None" = None) -> Decision:
    """
    A decision tool for *service* — one of ``SERVICES`` — answering *questions*.

    The questions are fixed here, for the life of the tool: an agent passes
    only the text. ``model`` defaults to the provider's; ``name`` to the
    provider's tool name and ``description`` to one written from the
    questions; ``api_key`` to the provider's environment variable.

    ::

        triage = decide("openai", name="triage", questions={
            "urgent": {"type": "predicate", "instructions": "Does this convey urgency?"},
        })
        triage.run("Help! My payouts have been failing for 3 days.")["answers"]["urgent"]
    """
    cls = SERVICES.get(service.lower()) if isinstance(service, str) else None
    if cls is None:
        raise ValueError(f"unknown decision service {service!r}; one of: {', '.join(SERVICES)}")
    return cls(questions, name=name, description=description, model=model, api_key=api_key)


__all__ = ["decide", "SERVICES", "Decision", "decisionOpenAI", "decisionTypeSafe"]



# The package is callable — `decide("…")` — and stays the package; see tools._callable.
from .._callable import callable_package as _callable_package
_callable_package(__name__, decide)
