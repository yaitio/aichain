"""Typed questions about a text, answered with probabilities: ``decisionOpenAI``,
``decisionTypeSafe`` and ``decisionInception`` behind one format, made with ``decide(service,
questions=...)``. A new provider subclasses ``Decision``, implements ``_ask``
and joins ``SERVICES``. See docs/tools/decide.md.
"""

from ._base    import Decision
from .openai   import decisionOpenAI
from .typesafe import decisionTypeSafe
from .inception import decisionInception

#: Provider name → tool class, for :func:`decide`.
SERVICES = {
    "openai":   decisionOpenAI,
    "typesafe": decisionTypeSafe,
    "inception": decisionInception,
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


__all__ = ["decide", "SERVICES", "Decision", "decisionOpenAI", "decisionTypeSafe", "decisionInception"]



# The package is callable — `decide("…")` — and stays the package; see tools._callable.
from .._callable import callable_package as _callable_package
_callable_package(__name__, decide)
