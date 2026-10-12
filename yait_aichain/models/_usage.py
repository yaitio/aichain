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
    cache_write_tokens : int
        Prompt prefix stored for reuse this call, billed above fresh input.
    cache_read_tokens : int
        Prompt prefix reused from an earlier call, billed well below it.
    image_input_tokens : int
        Images sent in, kept apart because they are billed at their own rate.
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
    # Image tokens sent *in*, kept out of ``input_tokens`` because they are
    # not billed at the same rate: an edit carrying reference images costs
    # $8/M against text's $5/M on the GPT Image 2.5 family. Measured on a real
    # edit call — 1024 image tokens beside 19 of text — folding them together
    # under-states the input line by 37% and the whole call by 22%.
    image_input_tokens: int = 0

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
            image_input_tokens = self.image_input_tokens + other.image_input_tokens,
        )

    def __radd__(self, other):
        # Enables sum([...]) which starts from int 0.
        if other == 0:
            return self
        return self.__add__(other)

    def __bool__(self) -> bool:
        return bool(self.input_tokens or self.output_tokens or self.total_tokens)


class Spend:
    """`tokens_used` and `cost` for a result that carries a `usage`.

    Read through `is not None`, not truthiness: a `Usage` with no tokens is
    falsy, and a cost the provider reported — or a price per picture — would
    otherwise read as unknown.
    """

    @property
    def tokens_used(self) -> int:
        return (self.usage.total_tokens or 0) if self.usage is not None else 0

    @property
    def cost(self) -> "float | None":
        return self.usage.cost if self.usage is not None else None


def extract_usage(response: dict) -> Usage:
    """
    Build a ``Usage`` from a raw provider response dict.

    Recognises the OpenAI-compatible, Anthropic, and Google shapes; returns
    a zero ``Usage`` when no usage block is present (never raises).

    Cached-prefix tokens are reported separately by all three, and are always
    left disjoint from ``input_tokens`` here whatever the provider's own
    convention — Anthropic reports the cache beside the input count, OpenAI
    and Google inside it.
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
        idet = u.get("input_tokens_details")
        # The Responses API reports the same thing under its own names —
        # ``input_tokens_details.cached_tokens``, a subset of ``input_tokens``.
        # Only the Chat Completions spelling was read until 2026-10-01, so
        # every gpt-5.x and gpt-6 call reported its reused prefix as full-price
        # input: the cache worked and the bill said it had not.
        if isinstance(idet, dict) and not read:
            read = idet.get("cached_tokens") or 0
            inp  = max(0, inp - read)
        # The images API reports the modality split of its input. Image tokens
        # are dearer than text, so they are taken out here and priced apart.
        img_in = 0
        if isinstance(idet, dict):
            img_in = idet.get("image_tokens") or 0
            inp    = max(0, inp - img_in)
        total = u.get("total_tokens") or (inp + out + write + read + img_in)
        # Some providers bill in the response itself — OpenRouter puts the
        # call's price in `usage.cost`. That is the bill, not an estimate, so
        # it is kept and wins over the registry's arithmetic.
        reported = u.get("cost")
        return Usage(input_tokens=inp, output_tokens=out, total_tokens=total,
                     cost=float(reported) if isinstance(reported, (int, float)) else None,
                     cache_write_tokens=write, cache_read_tokens=read,
                     image_input_tokens=img_in)

    # Google: "usageMetadata".
    g = response.get("usageMetadata")
    if isinstance(g, dict):
        inp = g.get("promptTokenCount", 0) or 0
        out = g.get("candidatesTokenCount", 0) or 0
        # Google follows OpenAI's convention, not Anthropic's: the cached
        # prefix is counted *inside* promptTokenCount, so it is subtracted
        # here to leave three disjoint counts, exactly as the OpenAI branch
        # above does. Left folded in, a reused prefix reads as full-price
        # input and the report says the cache never happened.
        read = g.get("cachedContentTokenCount") or 0
        inp  = max(0, inp - read)
        # Thinking tokens are billed, and are not in candidatesTokenCount:
        # "When thinking is turned on, response pricing is the sum of output
        # tokens and thinking tokens." Dropped, they under-state a reasoning
        # call enormously — measured at 1 output token beside 235 of thought.
        out += g.get("thoughtsTokenCount") or 0
        # Google reports the input split by modality, same reasoning as above.
        img_in = 0
        for entry in g.get("promptTokensDetails") or []:
            if isinstance(entry, dict) and entry.get("modality") == "IMAGE":
                img_in += entry.get("tokenCount") or 0
        inp = max(0, inp - img_in)
        total = g.get("totalTokenCount") or (inp + out + read + img_in)
        return Usage(input_tokens=inp, output_tokens=out, total_tokens=total,
                     cache_read_tokens=read, image_input_tokens=img_in)

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
    has no price per token in the provider data.

    Cached tokens are priced at their own rates. Folding them into the input
    line would overstate a long conversation's cost by close to an order of
    magnitude, since almost all of a cached turn is reused prefix.

    *cache_ttl* selects the write premium. The response says how many tokens
    were stored but not for how long, so the lifetime has to come from the
    caller that asked for it; an unknown value is priced as the cheaper 5m
    rather than raising, since a cost estimate should never break a call.

    ``price`` may carry, beside ``input`` and ``output`` (USD per 1M):

    * ``image_input`` — an image sent *in*, where it is dearer than text.
      Without it an edit sending reference images is billed at the text rate:
      measured on a real call, 22% under.
    * ``cache_read`` — a reused prefix, where it is not the usual tenth of the
      input rate. Claude 5.5 reads at a twentieth and Fable 5.1 at a fortieth;
      priced at a tenth, a cached Fable 5.1 loop reads up to four times dear.
    * ``long_context = {above, input, output}`` — the rates once the prompt
      passes *above* tokens. Claude Haiku 5.5 charges five times as much past
      100K, which is the difference between a cheap call and one that is not.

    A model priced only ``per_image`` has no token rate and returns ``None``
    here; the image tool prices it by the picture.
    """
    price = _price_of(model_name)
    if price is None or "input" not in price or "output" not in price:
        return None
    rate_in, rate_out = price["input"], price["output"]
    tier = price.get("long_context")
    prompt = (usage.input_tokens + usage.image_input_tokens
              + usage.cache_write_tokens + usage.cache_read_tokens)
    if isinstance(tier, dict) and prompt > tier.get("above", float("inf")):
        rate_in, rate_out = tier.get("input", rate_in), tier.get("output", rate_out)
    rate_in /= 1_000_000
    # Falls back to the text rate: most models bill an image input at the
    # same price, and only those that do not carry the field.
    rate_img   = price.get("image_input", price["input"]) / 1_000_000
    rate_read  = (price["cache_read"] / 1_000_000 if "cache_read" in price
                  else rate_in * CACHE_READ_MULTIPLIER)
    write_mult = CACHE_WRITE_MULTIPLIERS.get(cache_ttl,
                                             CACHE_WRITE_MULTIPLIERS["5m"])
    return (
        usage.input_tokens  * rate_in
        + usage.image_input_tokens * rate_img
        + usage.cache_write_tokens * rate_in * write_mult
        + usage.cache_read_tokens  * rate_read
        + usage.output_tokens / 1_000_000 * rate_out
    )


def price_per_image(model_name: str) -> "float | None":
    """USD per picture for a model the provider prices by the image, else ``None``."""
    price = _price_of(model_name)
    value = price.get("per_image") if price else None
    return float(value) if isinstance(value, (int, float)) else None


def attach_cost(usage: Usage, model_name: str, cache_ttl: str = "5m") -> Usage:
    """Return *usage* with its ``cost`` filled in: what the provider reported
    when it reported one, else the registry's estimate, else ``None``."""
    if usage.cost is not None:
        return usage
    return replace(usage, cost=estimate_cost(usage, model_name, cache_ttl))
