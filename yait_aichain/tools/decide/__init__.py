"""
tools.decide
============

Decisions, not prose: a text and fixed typed questions in, calibrated answers
out. One provider so far — TypeSafe's Jev, :class:`decideTypeSafe`.
"""

from .typesafe import decideTypeSafe

__all__ = ["decideTypeSafe"]
