"""
tools.search._base
==================

``Search`` — abstract base class for all web search tool implementations.

Every concrete search tool:
  1. Inherits from ``Search``.
  2. Sets ``name``, ``description``, and optionally extends ``parameters``
     (the base already declares ``input`` + ``options``).
  3. Implements ``run(input, options=None) -> str`` — executes the search
     and returns a human-readable numbered plain-text result string.

The ``parameters`` schema defined here is the *minimum* contract shared by
all providers.  Individual implementations may extend the ``options``
sub-schema with provider-specific keys.

Standard ``options`` keys (all optional)
-----------------------------------------
  ``max_results``   int   — maximum number of results to return
  ``recency``       str   — freshness filter: "hour", "day", "week", "month", "year"
  ``country``       str   — ISO 3166-1 alpha-2 country code
  ``language``      str   — ISO 639-1 language code
  ``domains``       list  — restrict results to these domains
"""

from __future__ import annotations
from .._base import Tool


class Search(Tool):
    """
    Abstract base class for web search tools.

    All providers share the same ``run(input, options)`` interface so they
    are interchangeable inside Agent / Chain pipelines.

    Parameters
    ----------
    None at the base level — concrete subclasses accept ``api_key``.
    """

    name        = "search"
    description = "Search the web and return a plain-text numbered list of results."
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "string",
                "description": "The search query.",
            },
            "options": {
                "type":        "object",
                "description": "Optional search filters and configuration.",
                "properties": {
                    "max_results": {
                        "type":        "integer",
                        "description": "Maximum number of results to return (default 10).",
                    },
                    "recency": {
                        "type":        "string",
                        "enum":        ["hour", "day", "week", "month", "year"],
                        "description": "Filter results by recency.",
                    },
                    "country": {
                        "type":        "string",
                        "description": "ISO 3166-1 alpha-2 country code to localise results.",
                    },
                    "language": {
                        "type":        "string",
                        "description": "ISO 639-1 language code for results.",
                    },
                    "domains": {
                        "type":        "array",
                        "items":       {"type": "string"},
                        "description": "Restrict results to these domains only.",
                    },
                },
            },
        },
        "required": ["input"],
    }

    def run(self, input: str, options: "dict | None" = None) -> str:  # type: ignore[override]
        raise NotImplementedError(f"{type(self).__name__} must implement run()")
