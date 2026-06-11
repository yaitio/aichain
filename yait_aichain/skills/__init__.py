"""
skills
======

Public API for the aichain 2.0 skill layer.

A ``Skill`` binds a :class:`~models.Model` to a provider-agnostic input
template and a declared output format.  Calling ``skill.run()`` substitutes
any placeholder variables, translates the input to the model's native wire
format, executes the HTTP request, and returns a clean result.

Typical usage
-------------
::

    from models import Model
    from skills import Skill

    model = Model("gpt-4o")

    skill = Skill(
        model=model,
        input={
            "messages": [
                {"role": "system", "parts": [{"type": "text", "text": "Be concise."}]},
                {"role": "user",   "parts": [{"type": "text", "text": "Explain {topic}."}]},
            ]
        },
        output={"modalities": ["text"], "format": {"type": "text"}},
        variables={"topic": "quantum entanglement"},
        name="explainer",
    )

    # Use default variable
    answer = skill.run()

    # Override variable at call time
    answer = skill.run(variables={"topic": "black holes"})
"""

from ._skill import Skill

__all__ = ["Skill"]
