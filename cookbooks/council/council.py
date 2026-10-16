"""
council.py — An ensemble answers, ranks each other blind, and a chairman decides.

Reimplements karpathy/llm-council (~250 lines of backend, OpenRouter, FastAPI
and React). Here the members are any mix of vendors talked to directly, and the whole council is
the fifty lines below — library calls and plain Python, nothing else.

Required env vars — one per provider you list in COUNCIL:
    ANTHROPIC_API_KEY  OPENAI_API_KEY  GOOGLE_AI_API_KEY  XAI_API_KEY
    DEEPSEEK_API_KEY   MOONSHOT_API_KEY  DASHSCOPE_API_KEY
"""

import re
from collections import defaultdict

from yait_aichain import Model, Skill

from receipt import Receipt

QUESTION = "What is the most underrated idea in computer science, and why?"
COUNCIL  = ["claude-sonnet-5-5", "gpt-6.1-sol", "gemini-3.1-pro-preview", "grok-4.7",
            "deepseek-v4-pro", "kimi-k3", "qwen3.8-max"]
CHAIRMAN = "gemini-3.1-pro-preview"

#: Room to think and then answer: a reasoning model spends the provider's
#: default ceiling (4096 on DeepSeek) on thinking and is cut off before a word.
ROOM = {"max_tokens": 16000}

RUBRIC = (
    "Question: {question}\n\n{answers}\n\n"
    "Critique each response, then end with a line 'FINAL RANKING:' followed by "
    "the labels best-to-worst, one 'N. Response X' per line and nothing else."
)
VERDICT = (
    "You are the chairman of an LLM council. Question: {question}\n\n"
    "Member answers (anonymised):\n{answers}\n\n"
    "Peer standings (average rank, lower is better): {standings}\n\n"
    "Weigh the answers and the peer verdict, then write the single best final answer."
)


receipt = Receipt()


def ask(model, prompt, **variables):
    """One prompt, one model, the text back — or None, and the member drops out.
    Skill fills the prompt's {slots}."""
    skill = Skill(Model(model, options=ROOM), input={"messages": [{"role": "user", "parts": [prompt]}]}) \
        if _has_key(model) else None
    if skill is None:
        return None
    try:
        return skill.run(variables)
    except Exception as exc:          # no credit, a retired model, an outage
        print(f"  {model} drops out: {type(exc).__name__}: {str(exc)[:120]}")
        return None
    finally:
        with receipt.stage(model):
            if skill.last_usage is not None:
                receipt.add(skill.last_usage)


def _has_key(model):
    try:
        Model(model, options=ROOM)
        return True
    except ValueError:
        print(f"  {model} sits out: no key")
        return False


# ── 1. Every member answers ───────────────────────────────────────────────────
answered = [(m, a) for m in COUNCIL if (a := ask(m, QUESTION))]
members  = [m for m, _ in answered]

# Anonymised, so a judge cannot favour a name it recognises — its own included.
labels = [chr(65 + i) for i in range(len(members))]
block  = "\n\n".join(f"Response {l}:\n{a}" for l, (_, a) in zip(labels, answered))

# ── 2. Every member ranks the answers, not knowing whose they are ─────────────
reviews = [ask(m, RUBRIC, question=QUESTION, answers=block) for m in members]

places = defaultdict(list)
for review in filter(None, reviews):
    ranking = review.split("FINAL RANKING:")[-1]
    for position, label in enumerate(re.findall(r"Response ([A-Z])", ranking), 1):
        if label in labels:           # a judge may name a response that is not there
            places[members[labels.index(label)]].append(position)

standings = sorted((sum(p) / len(p), m) for m, p in places.items())

# ── 3. The chairman weighs both and writes the answer ─────────────────────────
print(ask(CHAIRMAN, VERDICT,
          question=QUESTION, answers=block, standings=standings))

# ── What the council thought of itself ────────────────────────────────────────
print("\nPeer standings")
for place, (average, model) in enumerate(standings, 1):
    print(f"  {place}. {model:<28} {average:.2f}")
receipt.print()
