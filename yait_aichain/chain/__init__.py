"""
chain
=====

``Chain`` — steps in a fixed order, sharing one set of variables.

A step is a :class:`~skills.Skill`, a tool, an :class:`~agent.Agent`, or a
plain function or a utility's method; each step's output flows into the next
step's input as a named variable.  The kinds can be freely mixed in any order.

Typical usage — Skills only
-----------------------------
::

    from yait_aichain.models import Model
    from yait_aichain.skills import Skill
    from yait_aichain.chain  import Chain

    analyst    = Skill(Model("claude-haiku-5-5"), prompt="Briefly analyse: {topic}",
                       name="analyst")
    translator = Skill(Model("gpt-6-luna"),
                       prompt="Translate the following text to {language}:\\n\\n{analysis}",
                       name="translator")

    chain = Chain([(analyst, "analysis"), (translator, "translation")],
                  name="analyse_and_translate")
    result = chain.run(variables={"topic": "renewable energy", "language": "Spanish"})
    print(result.output)            # the last step
    print(result["analysis"])       # any step, by its key

Typical usage — tools, Skills and functions
--------------------------------------------
::

    from yait_aichain.tools import fetch
    from yait_aichain.chain import Chain

    def word_count(text: str) -> int:
        return len(text.split())

    reader     = fetch("local")            # tool: URL → Markdown
    cleaner    = Skill(...)                # Skill: strips boilerplate
    translator = Skill(...)                # Skill: translates

    chain = Chain(
        steps=[
            (reader,     "raw_markdown"),                  # takes `url`
            (cleaner,    "article"),
            (translator, "translation"),
            (word_count, "words", {"text": "translation"}),  # input_map
        ],
    )

    result = chain.run(variables={
        "url":      "https://example.com/article",
        "language": "French",
    })

Step specification
------------------
Each element of *steps* may be:

  * A bare runner — output stored as ``"result"``.

  * ``(runner, output_key: str)`` — output stored under *output_key*.

  * ``(runner, output_key: str, input_map: dict)`` — same, plus an
    *input_map* that fills the step's arguments:
    ``{"param": "variable"}`` passes a variable, ``{"param": "{template}"}``
    the template filled in.

A pause for a person or a signal is a step made with ``wait(reason=…)`` or
``gate(tool, reason=…)``; the run parks there and ``Chain.resume()`` carries
it on.

How variables are forwarded
---------------------------
  * **Skill** receives the full accumulated dict for template substitution.
  * **Tool** receives only the kwargs matching its declared
    ``parameters["properties"]`` keys (remapped via *input_map* if given).
  * **Function / method** receives the arguments its signature names.

When a step returns a **dict**, all keys are merged directly into the
accumulated variable dict (``output_key`` is ignored for dict outputs).

Persistence
-----------
Chains can be saved to YAML and reloaded without re-specifying every
detail in code::

    # Save
    chain.save("chains/my_chain.yaml")

    # Load — API keys resolved from environment variables automatically
    from yait_aichain.chain import Chain
    chain = Chain.load("chains/my_chain.yaml")
    result = chain.run(variables={"topic": "AI", "language": "French"})

The YAML file stores all step metadata.  For Skill steps the model name,
input template, and output spec are persisted.  For Tool steps the
fully-qualified class path and any serialisable ``init_args`` are stored;
API keys are **never** written and are resolved from environment variables
at load time.  A function step is code, not configuration, and cannot be
saved.
"""

from ._chain import Chain
from ._result import ChainResult
from ._step import Step

__all__ = ["Chain", "ChainResult", "Step"]
