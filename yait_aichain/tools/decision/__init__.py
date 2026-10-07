"""
tools.decision
==============

Typed questions about a text, answered with probabilities — one format of
questions and answers, a provider per tool, the way ``tools.search`` works.

Tools
-----
decisionOpenAI     OpenAI's Decisions API — text and images, gpt-6-luna.
decisionTypeSafe   TypeSafe's Jev — text and structured state.

``Decision`` is the base; it holds the format, the checks and the translation
of answers, and a new provider is a subclass that implements ``_ask``.
"""

from ._base    import Decision
from .openai   import decisionOpenAI
from .typesafe import decisionTypeSafe

__all__ = ["Decision", "decisionOpenAI", "decisionTypeSafe"]
