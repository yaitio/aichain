"""
chain — ``Chain``, steps in a fixed order sharing one set of variables.

A step is a Skill, a tool, an Agent, a plain function or a utility's method,
mixed in any order; each step's output is stored under its key for the steps
after it. A pause is a step made with ``wait(...)`` or ``gate(tool, ...)``.
See docs/pipelines/chain.md.
"""

from ._chain import Chain
from ._result import ChainResult
from ._step import Step

__all__ = ["Chain", "ChainResult", "Step"]
