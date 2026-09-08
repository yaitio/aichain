"""
One place where a structured reply becomes a value, or an error that says
which of two different things went wrong.

A truncated response and an ignored schema both arrive as HTTP 200, and the
fixes are opposite: raise ``max_tokens`` for one, change the schema or the
prompt for the other. A caller handed a bare ``JSONDecodeError`` cannot tell
them apart and retries the wrong one — three identical failures at the same
column are a ceiling, not a fluctuation, but only if somebody says so.
"""

from __future__ import annotations

import json

from .._errors import TruncatedResponseError, InvalidStructuredOutputError
from ...models._schema import check_structure


#: Every provider's word for "I ran out of output budget".
_CEILING = {"length", "max_tokens", "MAX_TOKENS"}


def parse_structured(text: str, schema: "dict | None", *,
                     finish_reason: str = "",
                     output_tokens: "int | None" = None,
                     strip_fences: bool = False) -> object:
    """
    Parse *text* as the JSON *schema* describes, or raise a typed error.

    ``finish_reason`` and ``output_tokens`` come from the provider's own
    response. The token count is what makes the message actionable: it is the
    ceiling that was hit, so it is the number ``max_tokens`` has to exceed.
    """
    body = text.strip()
    if strip_fences and body.startswith("```"):
        body = body.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        if finish_reason in _CEILING:
            at = (f" at {output_tokens} output tokens"
                  if output_tokens else "")
            raise TruncatedResponseError(
                f"Response was truncated{at} (finish_reason="
                f"{finish_reason!r}) before the JSON completed — raise "
                f"max_tokens above {output_tokens or 'the current limit'}.",
                finish_reason=finish_reason, output_tokens=output_tokens,
            ) from exc
        raise InvalidStructuredOutputError(
            f"Model returned invalid JSON: {exc}", value=body,
        ) from exc

    # Whole, but not necessarily what was asked for. Google strips
    # `additionalProperties` from the schema it is sent, so a closed object
    # comes back with fields nobody declared and nothing notices until two
    # providers' outputs are diffed by hand.
    if isinstance(schema, dict) and (problems := check_structure(value, schema)):
        raise InvalidStructuredOutputError(
            "Response does not match the requested schema: "
            + "; ".join(problems[:5])
            + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""),
            problems=problems, value=value,
        )
    return value
