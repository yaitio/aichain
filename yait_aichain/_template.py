"""
yait_aichain._template
======================

Safe ``{placeholder}`` substitution shared by skills and the agent.

``str.format_map`` is unusable for LLM prompt templates: any literal brace
in the text — a JSON example (``{"a": 1}``), an empty ``{}``, an unbalanced
``}`` in model-generated content — raises ``ValueError``, and ``{var.attr}``
/ ``{var[0]}`` can raise ``AttributeError``/``IndexError`` at format time.

``substitute_placeholders`` replaces **only** ``{name}`` occurrences where
*name* is a valid Python identifier present in *variables*.  Everything
else — unknown placeholders, format specs (``{x:>10}``), conversions
(``{x!r}``), attribute/index access, literal braces of any kind — is left
intact.  No escaping syntax is needed or honoured: ``{{`` stays ``{{``.
"""

from __future__ import annotations

import re

# Match a single ``{name}`` placeholder, but NOT one wrapped in doubled
# braces: the lookbehind/lookahead skip ``{{name}}`` so a doubled brace stays
# literal (``{{`` stays ``{{``), as documented.
_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")


def substitute_placeholders(text: str, variables: dict) -> str:
    """
    Return *text* with every ``{name}`` replaced by ``str(variables[name])``.

    Only bare identifiers are substituted, and only when the key exists in
    *variables*; all other brace constructs pass through unchanged, so
    templates may freely contain JSON examples and model-generated braces.
    """
    if not variables or "{" not in text:
        return text

    def _replace(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key in variables:
            return str(variables[key])
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, text)
