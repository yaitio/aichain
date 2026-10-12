"""
A picture's shape, in the two spellings providers use: pixels and a ratio.

Five providers convert one into the other, and each had its own copy — they had
drifted in rounding and in what they accepted. The differences that are real
(a provider that needs edges in multiples of 16, one that writes `*` for `x`)
are arguments here, not copies.
"""

from __future__ import annotations

from math import gcd


def pixels(size) -> "tuple[int, int] | None":
    """'1024x1536' → (1024, 1536), or None for anything else."""
    w, _, h = str(size).partition("x")
    if not (w.isdigit() and h.isdigit() and int(w) and int(h)):
        return None
    return int(w), int(h)


def ratio_of(size) -> "str | None":
    """'1024x1536' → '2:3', reduced: a provider that honours any ratio exactly
    has no list to snap to."""
    wh = pixels(size)
    if wh is None:
        return None
    g = gcd(*wh) or 1
    return f"{wh[0] // g}:{wh[1] // g}"


def size_from_ratio(ratio, short: int = 1024, multiple: int = 1, sep: str = "x") -> "str | None":
    """'16:9' → '1820x1024': the short edge pinned at *short*, the long one
    following, each rounded to *multiple* (16 where a provider requires it)."""
    a, _, b = str(ratio).partition(":")
    if not (a.isdigit() and b.isdigit() and int(a) and int(b)):
        return None
    w, h = int(a), int(b)
    def edge(n): return max(multiple, int(round(n / multiple)) * multiple)
    long_ = edge(short * max(w, h) / min(w, h))
    return f"{long_}{sep}{edge(short)}" if w >= h else f"{edge(short)}{sep}{long_}"


def nearest_ratio(size, ratios) -> "str | None":
    """The ratio in *ratios* closest to a 'WIDTHxHEIGHT' size."""
    wh = pixels(size)
    if wh is None:
        return None
    def value(r):
        a, _, b = r.partition(":")
        return int(a) / int(b)
    return min(ratios, key=lambda r: abs(value(r) - wh[0] / wh[1]))
