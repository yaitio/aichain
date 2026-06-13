"""
models._pricing
===============

Per-model token prices as **data**, and cost estimation on top of ``Usage``.

Prices are USD per 1 000 000 tokens, split into input (prompt) and output
(completion).  This is a *reference snapshot* — prices change, and the whole
point of keeping them as a plain dict is that they are trivial to update (or,
later, refresh from an external source such as models.dev; see block 1.2-E).

A model absent from ``PRICING`` yields ``cost=None`` — honest "unknown"
rather than a wrong number.  Cost never affects request inputs or outputs;
it only annotates ``Usage``.

Snapshot date: 2026-06. Public list prices, standard (non-cached) tier.
"""

from __future__ import annotations

from dataclasses import replace

from ._usage import Usage

# model name (without provider prefix) → {"input": $/Mtok, "output": $/Mtok}
PRICING: dict[str, dict[str, float]] = {
    # ── OpenAI ──────────────────────────────────────────────────────────
    "gpt-5":            {"input": 1.25,  "output": 10.0},
    "gpt-5-mini":       {"input": 0.25,  "output": 2.0},
    "gpt-4.1":          {"input": 2.0,   "output": 8.0},
    "gpt-4.1-mini":     {"input": 0.4,   "output": 1.6},
    "gpt-4.1-nano":     {"input": 0.1,   "output": 0.4},
    "gpt-4o":           {"input": 2.5,   "output": 10.0},
    "gpt-4o-mini":      {"input": 0.15,  "output": 0.6},
    "o3":               {"input": 2.0,   "output": 8.0},
    "o4-mini":          {"input": 1.1,   "output": 4.4},
    "o1":               {"input": 15.0,  "output": 60.0},

    # ── Anthropic ───────────────────────────────────────────────────────
    "claude-opus-4-6":            {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6":          {"input": 3.0,  "output": 15.0},
    "claude-haiku-4-5-20251001":  {"input": 1.0,  "output": 5.0},

    # ── Google ──────────────────────────────────────────────────────────
    "gemini-2.5-pro":    {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash":  {"input": 0.3,  "output": 2.5},
    "gemini-2.0-flash":  {"input": 0.1,  "output": 0.4},

    # ── DeepSeek ────────────────────────────────────────────────────────
    "deepseek-chat":      {"input": 0.27, "output": 1.1},
    "deepseek-reasoner":  {"input": 0.55, "output": 2.19},

    # ── xAI ─────────────────────────────────────────────────────────────
    "grok-3":        {"input": 3.0, "output": 15.0},
    "grok-3-mini":   {"input": 0.3, "output": 0.5},

    # ── Perplexity ──────────────────────────────────────────────────────
    "sonar":            {"input": 1.0, "output": 1.0},
    "sonar-pro":        {"input": 3.0, "output": 15.0},

    # ── Kimi ────────────────────────────────────────────────────────────
    "kimi-k2-0905-preview":  {"input": 0.6, "output": 2.5},

    # ── Qwen ────────────────────────────────────────────────────────────
    "qwen-max":   {"input": 1.6, "output": 6.4},
    "qwen-plus":  {"input": 0.4, "output": 1.2},
}


def estimate_cost(usage: Usage, model_name: str) -> "float | None":
    """
    Estimate USD cost of *usage* for *model_name*, or ``None`` if the model
    has no price entry.
    """
    price = PRICING.get(model_name)
    if price is None:
        return None
    return (
        usage.input_tokens  / 1_000_000 * price["input"]
        + usage.output_tokens / 1_000_000 * price["output"]
    )


def attach_cost(usage: Usage, model_name: str) -> Usage:
    """Return *usage* with its ``cost`` field filled in (``None`` if unknown)."""
    return replace(usage, cost=estimate_cost(usage, model_name))
