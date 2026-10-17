"""The brief, one section at a time: each sees the outline, the summaries of its
papers and everything already written, and no call is asked for the whole."""

import os

from yait_aichain import Model, Skill

WRITER_MODEL = os.getenv("WRITER_MODEL", "claude-sonnet-5-5")
SECTION_TOKENS = 3000          # room for one section and no more
#: No thinking: its tokens count against max_tokens and would cut a section short,
#: and they would make the output tokens counted below more than the brief's text.
WRITER_OPTIONS = {"max_tokens": SECTION_TOKENS, "reasoning": "none"}
SECTION = """You are writing one section of a literature brief: {topic}.

The outline of the whole brief:
{outline}

Already written (do not repeat it; continue from it):
{written}

Summaries of the papers this section covers:
{summaries}

Write the section "{title}" in about {words} words of plain prose, as a Markdown
section starting with "## {title}". Compare the papers rather than listing them, and
cite each by its arXiv id in square brackets, as [2406.12045], every time you use it."""


def write(topic: str, outline: list[dict], summaries: dict, hooks=None) -> tuple[list[str], int]:
    """(sections, output tokens). A section cut at max_tokens is asked again once for
    half the length; cut twice, the run fails rather than keep half a sentence."""
    plan = "\n".join(f"{i}. {s['title']} — {', '.join(s['papers'])}" for i, s in enumerate(outline, 1))
    sections, spent = [], 0
    for section in outline:
        words = 1100
        for _ in range(2):
            skill = Skill(Model(WRITER_MODEL, options=WRITER_OPTIONS),
                          prompt=SECTION, hooks=hooks)
            text = skill.run(variables={
                "topic": topic, "outline": plan, "title": section["title"], "words": words,
                "written": "\n\n".join(sections) or "(nothing yet: this is the first section)",
                "summaries": "\n\n".join(f"[{p}]\n{summaries[p]}" for p in section["papers"]
                                         if p in summaries)})
            spent += skill.last_usage.output_tokens
            if not skill.last_truncated:
                break
            words //= 2
        else:
            raise SystemExit(f"section {section['title']!r} was cut at {SECTION_TOKENS} tokens twice")
        sections.append(text.strip())
    return sections, spent
