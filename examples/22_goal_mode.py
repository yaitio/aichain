"""
22 · Goal mode — a loop with no plan
====================================

A plan is a bet that you know the steps up front. Sometimes you don't: the next
action depends on what the last one returned. That is what ``mode="goal"`` is
for — you give an objective and a **done condition**, and the agent decides one
action at a time.

The task here makes the difference obvious. A hidden number is locked in a safe;
the only way in is a ``probe`` tool that answers "higher", "lower" or "correct".
No plan can be written in advance — step 4 is unknowable until step 3 answers.

Run it::

    export ANTHROPIC_API_KEY=...     # or set MODEL below to a model you have
    python examples/22_goal_mode.py

What to check in the output
---------------------------
* The agent never plans — it goes straight to iteration 1.
* Each guess narrows the range: a binary search should land in ~10 probes,
  a linear scan would not finish inside the budget.
* The run ends on a **check**, not on the model's word: ``done_when`` is a
  Python callable the harness evaluates itself.
* The journal at the end is the record of what was actually attempted.

The loop is only as good as the orchestrator driving it. Measured on this task
(secret in 1–1000, 15 iterations allowed): ``claude-sonnet-4-6`` ran a textbook
binary search and opened the safe in 9 probes; ``gpt-4o-mini`` bisected for a
while, then degenerated into +1 scanning and hit the iteration cap. The harness
behaved identically in both runs — the stop rules are there precisely because
the orchestrator cannot be assumed competent.
"""

import random

MODEL = "claude-sonnet-4-6"      # any orchestrator model you have a key for

from yait_aichain import Agent, Model
from yait_aichain.agent import Journal
from yait_aichain.tools import Tool

# ── The safe ───────────────────────────────────────────────────────────────────
# Chosen at import time and never revealed to the model — the only channel is
# the probe tool below.
SECRET   = random.randint(1, 1000)
attempts: list[int] = []


class Probe(Tool):
    name        = "probe"
    description = ("Try a combination between 1 and 1000. Answers 'higher', "
                   "'lower', or 'correct'. This is the only way to learn the "
                   "number.")
    risk        = "read"
    parameters  = {
        "type": "object",
        "properties": {"guess": {"type": "integer",
                                 "description": "the number to try, 1–1000"}},
        "required": ["guess"],
    }

    def run(self, guess, options=None):
        guess = int(guess)
        attempts.append(guess)
        if guess == SECRET:
            return f"{guess} → CORRECT. The safe is open."
        direction = "higher" if guess < SECRET else "lower"
        return f"{guess} → wrong, the number is {direction} than {guess}."


class RecordAnswer(Tool):
    name        = "record_answer"
    description = "Record the combination once probe has confirmed it."
    risk        = "write"
    parameters  = {
        "type": "object",
        "properties": {"combination": {"type": "integer"}},
        "required": ["combination"],
    }

    def run(self, combination, options=None):
        return f"recorded: {combination}"


# ── The done condition ─────────────────────────────────────────────────────────
# A callable, not a sentence. The harness runs it against memory, so the run can
# only finish on a fact — the model cannot talk its way to success.

def safe_is_open(memory: dict) -> bool:
    """the recorded combination actually opens the safe"""
    for value in memory.values():
        digits = "".join(c for c in str(value) if c.isdigit())
        if digits and int(digits) == SECRET:
            return True
    return False


# ── The agent ──────────────────────────────────────────────────────────────────

agent = Agent(
    orchestrator = Model(MODEL),
    tools        = [Probe(), RecordAnswer()],
    mode         = "goal",
    done_when    = safe_is_open,
    max_steps    = 15,          # a binary search needs ~10; a linear scan can't
    max_tokens   = 60_000,
    verbose      = 1,
)

result = agent.run(
    "Open the safe. The combination is a whole number between 1 and 1000. "
    "Use probe to narrow it down, then record the answer with record_answer. "
    "Each probe costs money — use as few as you can."
)

# ── What happened ──────────────────────────────────────────────────────────────

print("\n" + "═" * 72)
print(f"secret was     : {SECRET}")
print(f"probes used    : {len(attempts)}  →  {attempts}")
print(f"success        : {result.success}")
print(f"output         : {result.output}")
print(f"tokens         : {result.tokens_used:,}")

print("\nJournal — what the agent actually did:")
for e in result.journal:
    kind = e["evidence"]["kind"] if e.get("evidence") else "—"
    mark = "✓" if kind == "check" else "·"
    print(f"  {e['seq']:>2}. {mark} [{e['outcome']:<8} {kind:<11}] {e['intent']}")

j = Journal.from_list(result.journal)
print(f"\nmade progress  : {j.has_progress(5)}")
print(f"ruled out      : {j.do_not_redo() or '(nothing)'}")

# A binary search over 1–1000 needs at most 10 probes; anything much worse means
# the agent was not using the feedback it got.
if result.success:
    print(f"\nefficiency     : {len(attempts)} probes "
          f"({'optimal-ish' if len(attempts) <= 12 else 'wasteful'}; "
          f"binary search needs ≤10)")
