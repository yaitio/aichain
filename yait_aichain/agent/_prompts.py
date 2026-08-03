"""
agent._prompts
==============

What is left of the prompt layer once tool calling is native: the system
message and the rendering of results. Nothing else.

The previous versions of this module are a history of the text protocol —
first five prompt builders composing fresh two-message prompts each step, then
one growing conversation carrying an action vocabulary and every tool schema
as JSON text. Measured on a 16-tool task, that put 13,473 characters of
machinery into the system prompt against 357 for a native-calling agent doing
the same job, and the policy that actually mattered drowned in it. Schemas now
travel as the provider's ``tools`` field and the reply arrives as typed calls,
so the prompt carries what only the prompt can: the caller's instructions.
"""

from __future__ import annotations

import json

#: The one rule that is prompt-shaped by nature: answering is the absence of a
#: call. Everything else — which tools exist, what they take — the provider
#: presents natively.
_GROUND_RULES = """\
When the work is done, reply in plain text — that reply is your final answer.
Call tools only while you still need something done or looked up."""

_PLAN_RULE = """\
Before anything else, call write_plan with a short ordered list of what this
task requires. You may not rewrite it — decide once and hold to it."""


def system_message(instructions: str, mode: str,
                   ground_rules: bool = True) -> str:
    """
    The stable prefix. Written once per run and never rewritten — anything
    that changes as the run proceeds belongs in the conversation tail, because
    editing the prefix invalidates the cache for every turn that follows.

    *ground_rules* carries the "plain text ends the run" rule, and it belongs
    to **whoever owns the loop**. ``run()`` owns it, so it sets the rule. An
    external driver — a dialogue harness, a serverless step — owns its own
    loop, where a plain reply is an ordinary conversational turn; imposing the
    run() semantics there miscasts asking the user a question as declaring the
    work finished. Measured on a dialogue benchmark as a bias toward polite
    refusal over action.
    """
    parts = [instructions.strip()] if instructions.strip() else []
    if ground_rules:
        parts.append(_GROUND_RULES)
    if mode == "waterfall":
        parts.append(_PLAN_RULE)
    return "\n\n".join(parts)


def result_message(result, error: "str | None" = None) -> str:
    """One call's result, rendered for its ``tool`` turn. Untruncated."""
    if error:
        return f"ERROR: {error}"
    if isinstance(result, (dict, list)):
        return json.dumps(result, indent=1, ensure_ascii=False, default=str)
    return str(result)
