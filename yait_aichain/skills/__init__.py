"""The skill layer: ``Skill`` binds a ``Model`` to a provider-agnostic input
template and a declared output format; ``run()`` fills the placeholders,
sends the request and returns a clean result. See docs/pipelines/skills.md.
"""

from ._skill import Skill

__all__ = ["Skill"]
