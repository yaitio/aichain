"""
chain
=====

Public API for the aichain 2.0 chain layer.

A ``Chain`` wires a sequence of :class:`~skills.Skill` and
:class:`~tools.Tool` instances into a pipeline where each step's output
automatically flows into the next step's input as a named variable.
Skills and Tools can be freely mixed in any order.

Typical usage — Skills only
-----------------------------
::

    from models import Model
    from skills import Skill
    from chain  import Chain

    analyst = Skill(
        model=Model("gpt-4o"),
        input={"messages": [
            {"role": "user", "parts": [
                {"type": "text", "text": "Briefly analyse: {topic}"},
            ]},
        ]},
        output={"modalities": ["text"], "format": {"type": "text"}},
        name="analyst",
    )

    translator = Skill(
        model=Model("gpt-4o"),
        input={"messages": [
            {"role": "user", "parts": [
                {"type": "text",
                 "text": "Translate the following text to {language}:\\n\\n{result}"},
            ]},
        ]},
        output={"modalities": ["text"], "format": {"type": "text"}},
        name="translator",
    )

    chain = Chain([analyst, translator], name="analyse_and_translate")
    output = chain.run(variables={"topic": "renewable energy", "language": "Spanish"})

Typical usage — Mixed Tool + Skill pipeline
--------------------------------------------
::

    from tools import MarkItDownTool
    from chain import Chain

    fetch      = MarkItDownTool()          # Tool: converts URL → Markdown
    cleaner    = Skill(...)                # Skill: strips boilerplate
    translator = Skill(...)                # Skill: translates

    chain = Chain(
        steps=[
            (fetch,      "raw_markdown", {"source": "url"}),  # input_map
            (cleaner,    "article"),
            (translator, "translation"),
        ],
    )

    result = chain.run(variables={
        "url":      "https://example.com/article",
        "language": "French",
    })

Step specification
------------------
Each element of *steps* may be:

  * A bare ``Skill`` or ``Tool`` — output stored as ``"result"``.

  * ``(runner, output_key: str)`` — output stored under *output_key*.

  * ``(runner, output_key: str, input_map: dict)`` — same, plus an
    *input_map* that renames accumulated variables before they are passed
    to a Tool: ``{"tool_param": "accumulated_var"}``.

How variables are forwarded
---------------------------
  * **Skill** receives the full accumulated dict for template substitution.
  * **Tool** receives only the kwargs matching its declared
    ``parameters["properties"]`` keys (remapped via *input_map* if given).

When a step returns a **dict**, all keys are merged directly into the
accumulated variable dict (``output_key`` is ignored for dict outputs).

Persistence
-----------
Chains can be saved to YAML and reloaded without re-specifying every
detail in code::

    # Save
    chain.save("chains/my_chain.yaml")

    # Load — API keys resolved from environment variables automatically
    from chain import Chain
    chain = Chain.load("chains/my_chain.yaml")
    result = chain.run(variables={"topic": "AI", "language": "French"})

The YAML file stores all step metadata.  For Skill steps the model name,
input template, and output spec are persisted.  For Tool steps the
fully-qualified class path and any serialisable ``init_args`` are stored;
API keys are **never** written and are resolved from environment variables
at load time.
"""

from ._chain import Chain

__all__ = ["Chain"]
