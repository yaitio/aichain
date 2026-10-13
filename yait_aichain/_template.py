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

import json
import re

# A doubled brace, or a single ``{name}``. Scanned left to right, so ``{{``
# and ``}}`` are consumed as literals and a placeholder beside a closing brace
# of JSON — ``{"summary": {text}}`` — is still one; a lookaround read that as
# an escape and left the document out. *name* is any identifier, ``{θέμα}``
# included. A doubled brace matches with no group.
_PLACEHOLDER_RE = re.compile(r"\{\{|\}\}|\{([^\W\d]\w*)\}")

_warned: set = set()


def placeholder_names(text: str) -> list:
    """The ``{name}`` placeholders in *text*, in order."""
    return [n for n in _PLACEHOLDER_RE.findall(text) if n]


def _text(value) -> str:
    # A dict or a list as JSON: its repr — single quotes, True, None — is
    # Python, and a model was handed that as the data. Only the plain types:
    # a subclass with its own text (a tool's Record) says what it means.
    if type(value) in (dict, list, tuple):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def substitute_placeholders(text: str, variables: dict, *, warn: bool = False) -> str:
    """
    Return *text* with every ``{name}`` replaced by the value of
    ``variables[name]`` — a dict or a list as JSON, anything else as ``str``.

    Only bare identifiers are substituted, and only when the key exists in
    *variables*; all other brace constructs pass through unchanged, so
    templates may freely contain JSON examples and model-generated braces.
    With *warn*, a placeholder left unfilled while variables were given is
    said once, with the nearest name — a typo sent ``{topic}`` to a model
    a thousand times, unsaid.
    """
    if not variables or "{" not in text:
        return text

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key is None:
            return match.group(0)
        if key in variables:
            return _text(variables[key])
        if warn and key not in _warned:
            _warned.add(key)
            import difflib
            import warnings
            near = difflib.get_close_matches(key, [str(k) for k in variables], n=1)
            warnings.warn(f"{{{key}}} is not among the variables given and was sent as written"
                          + (f"; did you mean {{{near[0]}}}?" if near else
                             f"; they are {sorted(map(str, variables))}"), stacklevel=4)
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, text)
