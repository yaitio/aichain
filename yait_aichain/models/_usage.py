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

from dataclasses import dataclass, replace


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
    # Prompt-cache accounting (1.6.1). A provider bills a reused prefix at a
    # fraction of the fresh rate, so a report that folds cached tokens into
    # ``input_tokens`` overstates what a long conversation actually costs —
    # measured at roughly tenfold on Anthropic.
    cache_write_tokens: int = 0    # prefix stored this call (billed above 1x)
    cache_read_tokens:  int = 0    # prefix reused from an earlier call

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
            cache_write_tokens = self.cache_write_tokens + other.cache_write_tokens,
            cache_read_tokens  = self.cache_read_tokens  + other.cache_read_tokens,
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
        # The two providers disagree about what ``input_tokens`` contains.
        # Anthropic reports the cached prefix in fields of its own, *beside*
        # the input count. OpenAI does the opposite: ``cached_tokens`` is a
        # subset of ``prompt_tokens``. Taken at face value that difference
        # prices a reused OpenAI prefix at 1.1x instead of 0.1x — the same
        # class of quiet arithmetic error this accounting exists to remove —
        # so subtract it there, leaving three disjoint counts either way.
        write = u.get("cache_creation_input_tokens") or 0
        read  = u.get("cache_read_input_tokens") or 0
        details = u.get("prompt_tokens_details")
        if isinstance(details, dict) and not read:
            read = details.get("cached_tokens") or 0
            inp  = max(0, inp - read)
        total = u.get("total_tokens") or (inp + out + write + read)
        return Usage(input_tokens=inp, output_tokens=out, total_tokens=total,
                     cache_write_tokens=write, cache_read_tokens=read)

    # Google: "usageMetadata".
    g = response.get("usageMetadata")
    if isinstance(g, dict):
        inp = g.get("promptTokenCount", 0) or 0
        out = g.get("candidatesTokenCount", 0) or 0
        total = g.get("totalTokenCount") or (inp + out)
        return Usage(input_tokens=inp, output_tokens=out, total_tokens=total)

    return Usage()


# ---------------------------------------------------------------------------
# Cost — prices live in the provider data (providers/*.toml)
# ---------------------------------------------------------------------------
#
# The price lookup is imported lazily so that merely importing ``Usage`` (which
# every primitive does for ``.last_usage``) stays light and never triggers the
# provider-data load.

def _price_of(model_name: str) -> "dict | None":
    from ._data import PROVIDERS
    for data in PROVIDERS.values():
        mdl = data.get("models", {}).get(model_name)
        if mdl is not None and "price" in mdl:
            return mdl["price"]
    return None


#: Cache multipliers relative to the model's base input rate. Storing a prefix
#: costs more than sending it once; reading it back costs a fraction. The write
#: premium is what the longer lifetime buys — 1.25x for five minutes, 2x for an
#: hour — against 0.1x per read either way, which puts break-even at the second
#: read at 5m and the third at 1h.
CACHE_WRITE_MULTIPLIERS = {"5m": 1.25, "1h": 2.0}
CACHE_READ_MULTIPLIER   = 0.10


def estimate_cost(usage: Usage, model_name: str,
                  cache_ttl: str = "5m") -> "float | None":
    """
    Estimate USD cost of *usage* for *model_name*, or ``None`` if the model
    has no price entry in the provider data.

    Cached tokens are priced at their own rates. Folding them into the input
    line would overstate a long conversation's cost by close to an order of
    magnitude, since almost all of a cached turn is reused prefix.

    *cache_ttl* selects the write premium. The response says how many tokens
    were stored but not for how long, so the lifetime has to come from the
    caller that asked for it; an unknown value is priced as the cheaper 5m
    rather than raising, since a cost estimate should never break a call.
    """
    price = _price_of(model_name)
    if price is None:
        return None
    rate_in    = price["input"] / 1_000_000
    write_mult = CACHE_WRITE_MULTIPLIERS.get(cache_ttl,
                                             CACHE_WRITE_MULTIPLIERS["5m"])
    return (
        usage.input_tokens  * rate_in
        + usage.cache_write_tokens * rate_in * write_mult
        + usage.cache_read_tokens  * rate_in * CACHE_READ_MULTIPLIER
        + usage.output_tokens / 1_000_000 * price["output"]
    )


def attach_cost(usage: Usage, model_name: str, cache_ttl: str = "5m") -> Usage:
    """Return *usage* with its ``cost`` field filled in (``None`` if unknown)."""
    return replace(usage, cost=estimate_cost(usage, model_name, cache_ttl))
