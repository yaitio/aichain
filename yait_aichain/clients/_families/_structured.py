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


def hit_ceiling(obj) -> bool:
    """Whether *obj* — a whole response or one stream event, any family —
    says the output budget ran out.

    The cases `refuse_cut_off` lets through are prose, and prose cut short was
    returned as if it were whole: `Skill.run` gave half a sentence and an
    agent reported it as its answer, `success=True`. This is what lets the
    caller be told. Read from the response itself, so a buffered call and a
    stream answer the same question the same way.
    """
    if not isinstance(obj, dict):
        return False
    for choice in obj.get("choices") or ():                 # OpenAI chat
        if isinstance(choice, dict) and choice.get("finish_reason") in _CEILING:
            return True
    for cand in obj.get("candidates") or ():                # Google
        if isinstance(cand, dict) and cand.get("finishReason") in _CEILING:
            return True
    if obj.get("stop_reason") in _CEILING:                  # Anthropic, whole
        return True
    delta = obj.get("delta")
    if isinstance(delta, dict) and delta.get("stop_reason") in _CEILING:   # Anthropic, event
        return True
    whole = obj.get("response") if isinstance(obj.get("response"), dict) else obj
    details = whole.get("incomplete_details")                # Responses API
    return (whole.get("status") == "incomplete" and isinstance(details, dict)
            and details.get("reason") == "max_output_tokens")


def refuse_cut_off(finish_reason: str, output_tokens: int | None, *,
                   calls: bool, text: str) -> None:
    """Raise `TruncatedResponseError` when the ceiling cut a reply that cannot
    stand: a tool call — its arguments may be half-written, and a tool run on
    defaults writes nothing and reports success — or an answer with no text,
    which ended a run as a success with an empty output. Prose cut short is
    still prose, and is returned."""
    if finish_reason not in _CEILING or not (calls or not (text or "").strip()):
        return
    at = f" at {output_tokens} output tokens" if output_tokens else ""
    what = "a tool call" if calls else "the answer, with no text yet"
    raise TruncatedResponseError(
        f"Response was cut off{at} (finish_reason={finish_reason!r}) in {what} — raise "
        f"max_tokens above {output_tokens or 'the current limit'}; reasoning models spend "
        "much of it thinking.",
        finish_reason=finish_reason, output_tokens=output_tokens)


def parse_structured(text: str, schema: dict | None, *,
                     finish_reason: str = "",
                     output_tokens: int | None = None,
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
    # can come back with fields nobody declared, which is checked here.
    if isinstance(schema, dict) and (problems := check_structure(value, schema)):
        raise InvalidStructuredOutputError(
            "Response does not match the requested schema: "
            + "; ".join(problems[:5])
            + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""),
            problems=problems, value=value,
        )
    return value
