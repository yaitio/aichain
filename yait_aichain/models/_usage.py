"""
models._usage
=============

Normalised token-usage accounting, uniform across every provider.

Each provider reports token counts under a different shape:

    OpenAI-compatible : usage.prompt_tokens / completion_tokens / total_tokens
    Anthropic         : usage.input_tokens / output_tokens
    Google            : usageMetadata.promptTokenCount / candidatesTokenCount

``extract_usage(response)`` flattens all of them into a single ``Usage``
object, so ``result.usage.input_tokens`` means the same thing no matter
which model produced it.

``Usage`` is additive (``a + b``), so a Chain/Pool can sum the usage of its
steps into one total.  Cost is attached separately in block 1.2-C.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Usage:
    """
    Token counts for one or more LLM calls.

    Attributes
    ----------
    input_tokens  : prompt / input tokens billed.
    output_tokens : completion / output tokens billed.
    total_tokens  : provider-reported total, or input+output when absent.
    cost          : estimated cost in USD (filled in 1.2-C; ``None`` until then).
    """

    input_tokens:  int = 0
    output_tokens: int = 0
    total_tokens:  int = 0
    cost:          "float | None" = None

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        a = self.cost
        b = other.cost
        merged_cost = None if (a is None and b is None) else (a or 0.0) + (b or 0.0)
        return Usage(
            input_tokens  = self.input_tokens  + other.input_tokens,
            output_tokens = self.output_tokens + other.output_tokens,
            total_tokens  = self.total_tokens  + other.total_tokens,
            cost          = merged_cost,
        )

    def __radd__(self, other):
        # Enables sum([...]) which starts from int 0.
        if other == 0:
            return self
        return self.__add__(other)

    def __bool__(self) -> bool:
        return bool(self.input_tokens or self.output_tokens or self.total_tokens)


def extract_usage(response: dict) -> Usage:
    """
    Build a ``Usage`` from a raw provider response dict.

    Recognises the OpenAI-compatible, Anthropic, and Google shapes; returns
    a zero ``Usage`` when no usage block is present (never raises).
    """
    if not isinstance(response, dict):
        return Usage()

    # OpenAI / Anthropic / OpenAI-compatible providers: top-level "usage".
    u = response.get("usage")
    if isinstance(u, dict):
        inp = u.get("input_tokens")
        out = u.get("output_tokens")
        if inp is None and out is None:
            inp = u.get("prompt_tokens", 0)
            out = u.get("completion_tokens", 0)
        inp = inp or 0
        out = out or 0
        total = u.get("total_tokens") or (inp + out)
        return Usage(input_tokens=inp, output_tokens=out, total_tokens=total)

    # Google: "usageMetadata".
    g = response.get("usageMetadata")
    if isinstance(g, dict):
        inp = g.get("promptTokenCount", 0) or 0
        out = g.get("candidatesTokenCount", 0) or 0
        total = g.get("totalTokenCount") or (inp + out)
        return Usage(input_tokens=inp, output_tokens=out, total_tokens=total)

    return Usage()
