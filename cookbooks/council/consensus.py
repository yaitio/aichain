"""
consensus.py — The models argue in a circle until nobody wants to change anything.

A different shape of agreement from council.py: no chairman and no vote. Each
model takes a turn seeing the discussion so far and either says AGREE or
rewrites the answer. Consensus is a full lap in which nobody rewrote anything.

Two things make it a conversation rather than a pile of text:

  * every model is told it is one of a panel whose **shared goal is to converge**,
    not to improve the answer — otherwise each turn simply adds to the last;
  * the transcript is built **per speaker**: its own earlier replies are
    `assistant` turns, everyone else's are `user` turns naming who spoke. Each
    model sees the discussion the way it would have lived through it — and an
    AGREE is a turn like any other, so a late speaker can see that the others
    have already settled.

This cannot be parallelised — every turn needs the one before it, which is the
honest counterpoint to parallel.py.

Required env vars — one per provider you list in COUNCIL:
    ANTHROPIC_API_KEY  OPENAI_API_KEY  GOOGLE_AI_API_KEY  XAI_API_KEY
    DEEPSEEK_API_KEY   MOONSHOT_API_KEY  DASHSCOPE_API_KEY
"""

from yait_aichain import Model, Skill

from receipt import Receipt

QUESTION = "What is the most underrated idea in computer science, and why?"
COUNCIL  = ["claude-sonnet-5-5", "gpt-6.1-sol", "gemini-3.1-pro-preview", "grok-4.7",
            "deepseek-v4-pro", "kimi-k3", "qwen3.8-max"]
MAX_LAPS = 3          # a cap, not a plan: without it a stubborn pair never stops

#: Room to think and then answer: a reasoning model spends the provider's
#: default ceiling (4096 on DeepSeek) on thinking and is cut off before a word.
ROOM = {"max_tokens": 16000}

BRIEF = (
    "You are one of {n} models on a panel: {members}. Your shared goal is to "
    "converge on a single answer all of you accept — not to write the best "
    "answer of your own. Converging matters more than being right in detail: "
    "accept a wording you can live with rather than restating your own.\n\n"
    "You speak in turn. When it is your turn, reply with exactly AGREE and "
    "nothing else if you would not change the answer as it stands. Otherwise "
    "reply with the full answer you propose instead — it replaces the current "
    "one, so keep it self-contained and no longer than it needs to be. Never "
    "write anything else: no preamble, no commentary on the discussion."
)


receipt = Receipt()


def ask(model, messages):
    """One conversation, one model, the text back."""
    try:
        skill = Skill(model=Model(model, options=ROOM), input={"messages": messages})
    except ValueError:                    # no key
        return None
    try:
        return skill.run()
    except Exception as exc:              # no credit, a retired model, an outage
        print(f"  {model} drops out: {type(exc).__name__}: {str(exc)[:120]}")
        return None
    finally:
        with receipt.stage(model):
            if skill.last_usage is not None:
                receipt.add(skill.last_usage)


def agrees(reply):
    return reply.strip().upper().startswith("AGREE")


def seen_by(speaker, dialogue):
    """The discussion as this speaker lived it: own turns assistant, others' user."""
    brief = BRIEF.format(n=len(COUNCIL), members=", ".join(COUNCIL))
    turns = [{"role": "system", "parts": [brief]},
             {"role": "user",   "parts": [f"Question: {QUESTION}"]}]
    for who, what in dialogue:
        turns.append({"role": "assistant", "parts": [what]} if who == speaker else
                     {"role": "user",      "parts": [f"{who} says:\n{what}"]})
    return turns


# ── The circle ────────────────────────────────────────────────────────────────
# Every turn is recorded, AGREE included: the next speaker should be able to see
# that the others have settled. Consensus is one full lap of nothing but AGREE.
dialogue = []

turn = 0
while turn < MAX_LAPS * len(COUNCIL) and len(COUNCIL) > 1:
    speaker = COUNCIL[turn % len(COUNCIL)]
    reply   = ask(speaker, seen_by(speaker, dialogue))
    if reply is None:                     # a member that cannot speak leaves the circle
        COUNCIL.remove(speaker)
        continue
    turn += 1
    dialogue.append((speaker, reply))

    print(f"  {speaker:<28} "
          + ("agrees" if agrees(reply) else f"proposes ({len(reply)} chars)"))

    lap = dialogue[-len(COUNCIL):]
    if len(lap) == len(COUNCIL) and all(agrees(r) for _, r in lap):
        break

# ── What they settled on, if they settled ─────────────────────────────────────
settled   = all(agrees(r) for _, r in dialogue[-len(COUNCIL):])
proposals = [(who, what) for who, what in dialogue if not agrees(what)]

print(f"\n{'Consensus' if settled else 'No consensus'} after "
      f"{len(dialogue) / len(COUNCIL):.2f} laps · {len(proposals)} proposals\n")
print(proposals[-1][1] if proposals else "nobody proposed anything")
receipt.print()
