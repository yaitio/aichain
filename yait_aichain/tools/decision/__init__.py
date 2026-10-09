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


# ── The package is callable ──────────────────────────────────────────────────
# `decide("…")` is called on this package, not on a function that takes its name
# in `tools`: a function there hides the package, and `import
# yait_aichain.tools.decision.x as m`, attribute access and — on Python 3.10 and
# 3.11 — `mock.patch("yait_aichain.tools.decision.x.…")` resolve through that name
# and fail. Measured 2026-10-09; see docs/design/tool-convention.md.
import inspect as _inspect
import sys as _sys
import types as _types


class _CallableModule(_types.ModuleType):
    def __call__(self, *args, **kwargs):
        return decide(*args, **kwargs)


_sys.modules[__name__].__class__ = _CallableModule
_sys.modules[__name__].__signature__ = _inspect.signature(decide)
