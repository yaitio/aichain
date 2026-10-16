"""
parallel.py — The same council, with the two fan-out stages run at once.

Members answer independently, so nothing forces them to wait for each other.
Three lines of `concurrent.futures` turn the two list comprehensions of
council.py into parallel ones; measured, that is 2.11x on one round, and the
gap widens with the size of the council.

Not `Pool`: Pool runs one runner over many inputs, and this is one input over
many models. Compare with consensus.py, where the same trick is impossible —
each turn there needs the previous one.

Required env vars — one per provider you list in COUNCIL:
    ANTHROPIC_API_KEY  OPENAI_API_KEY  GOOGLE_AI_API_KEY  XAI_API_KEY
    DEEPSEEK_API_KEY   MOONSHOT_API_KEY  DASHSCOPE_API_KEY
"""

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

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


def everyone(prompt, **variables):
    """The whole council on one prompt, at once; None where a member could not answer.
    The receipt is written here, after the threads: a stage is the receipt's own state."""
    with ThreadPoolExecutor(len(COUNCIL)) as threads:
        done = list(threads.map(lambda m: ask(m, prompt, **variables), COUNCIL))
    for model, (_, usage) in zip(COUNCIL, done):
        with receipt.stage(model):
            if usage is not None:
                receipt.add(usage)
    return [text for text, _ in done]


def ask(model, prompt, **variables):
    """(the text or None, its usage). Skill fills the prompt's {slots}."""
    try:
        skill = Skill(Model(model, options=ROOM), input={"messages": [{"role": "user", "parts": [prompt]}]})
    except ValueError:                    # no key: this member sits out
        return None, None
    try:
        return skill.run(variables), skill.last_usage
    except Exception as exc:              # no credit, a retired model, an outage
        print(f"  {model} drops out: {type(exc).__name__}: {str(exc)[:120]}")
        return None, skill.last_usage


# ── 1. Every member answers ───────────────────────────────────────────────────
answers = everyone(QUESTION)

# Anonymised, so a judge cannot favour a name it recognises — its own included.
labels = [chr(65 + i) for i in range(len(COUNCIL))]
block  = "\n\n".join(f"Response {l}:\n{a}" for l, a in zip(labels, answers) if a)

# ── 2. Every member ranks the answers, not knowing whose they are ─────────────
reviews = everyone(RUBRIC, question=QUESTION, answers=block)

places = defaultdict(list)
for review in filter(None, reviews):
    ranking = review.split("FINAL RANKING:")[-1]
    for position, label in enumerate(re.findall(r"Response ([A-Z])", ranking), 1):
        if label in labels and answers[labels.index(label)]:
            places[COUNCIL[labels.index(label)]].append(position)

standings = sorted((sum(p) / len(p), m) for m, p in places.items())

# ── 3. The chairman weighs both and writes the answer ─────────────────────────
verdict, usage = ask(CHAIRMAN, VERDICT, question=QUESTION, answers=block, standings=standings)
with receipt.stage(CHAIRMAN):
    if usage is not None:
        receipt.add(usage)
print(verdict)

# ── What the council thought of itself ────────────────────────────────────────
print("\nPeer standings")
for place, (average, model) in enumerate(standings, 1):
    print(f"  {place}. {model:<28} {average:.2f}")
receipt.print()
