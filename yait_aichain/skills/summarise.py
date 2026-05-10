"""
skills.summarise — make_summarise_skill()
==========================================

Factory that returns a lightweight Haiku-powered Skill for compressing a
completed document section into a compact rolling-context summary.

Role in the sectional document pipeline
----------------------------------------
After each section is written, a ``summarise`` Skill step reads the section
content and produces 3–5 bullet points capturing its key conclusions and
facts.  These bullets are stored in ``recent_summaries`` and injected into
the *next* section's prompt as context — keeping the overall context window
small regardless of how many sections have been written.

Only the two most-recent actual summaries are kept in ``recent_summaries``
at any time.  This is enforced by the prompt — the model is instructed to
return only the current section's bullets, and the accumulation of the last
two summaries is managed by the pipeline builder in ``run.py``, not here.

Usage
-----
    from skills.summarise import make_summarise_skill

    summarise = make_summarise_skill()
    # Use as a Chain step:
    (summarise, "current_section_actual_summary")

    # The step reads {current_section_title} and {current_section_content}
    # from the accumulated dict and writes the summary to the output_key.

Variables consumed (from accumulated dict)
------------------------------------------
``current_section_title``    — title of the section just written
``current_section_content``  — full markdown content just generated

Variable produced
-----------------
``current_section_actual_summary`` — 3–5 bullet summary of what was written
    (store with this output_key; the pipeline builder assembles
    ``recent_summaries`` from the last two values)
"""

from __future__ import annotations


_SUMMARISE_PROMPT = """\
You have just written a section of a long-form business document.
Produce a compact summary of what was actually written — 3 to 5 bullet
points covering the key conclusions, facts, and figures.

This summary will be injected into the next section's prompt as context
so the author can maintain consistency without re-reading the full text.
Be specific: names, numbers, and conclusions are more useful than vague
topic labels.

Section title   : {current_section_title}
Section content :
{current_section_content}

Output only the bullet points (starting each with "- "), nothing else.
"""


def make_summarise_skill(model_name: str = "claude-haiku-4-5-20251001"):
    """
    Return a Skill that summarises the most recently written section.

    Parameters
    ----------
    model_name : str, optional
        Model to use for summarisation.  Defaults to ``claude-haiku-4-5-20251001``
        (fast and inexpensive — appropriate for this mechanical task).

    Returns
    -------
    Skill
        A configured Skill instance ready to use as a Chain step.
        Reads ``{current_section_title}`` and ``{current_section_content}``
        from the accumulated dict.

    Examples
    --------
    ::

        from skills.summarise import make_summarise_skill

        summarise = make_summarise_skill()

        chain = Chain(steps=[
            ...
            (write_skill,  "current_section_content"),
            (summarise,    "current_section_actual_summary"),
            ...
        ])
    """
    # Local import to avoid circular dependency at module load time
    from models import Model
    from skills import Skill

    return Skill(
        model = Model(model_name),
        input = {
            "messages": [{
                "role":  "user",
                "parts": [{"type": "text", "text": _SUMMARISE_PROMPT}],
            }]
        },
        output = {
            "modalities": ["text"],
            "format":     {"type": "text"},
        },
        name        = "summarise_section",
        description = (
            "Compress the most recently written section into 3–5 bullet "
            "points for use as rolling context in the next section."
        ),
    )
