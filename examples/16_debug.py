"""
16_debug.py — Inspect intermediate steps in Chain, Pool, and Agent.

Shows how to debug complex pipelines:

  Chain  → chain.history  (step, name, input keys, output preview, duration)
  Pool   → pool.status    (live counts while running)
           pool.history   (per-item status, output, error, duration)
  Agent  → verbose=1      (plan + one line per step)
           verbose=2      (full action payloads)

Required env vars:
    ANTHROPIC_API_KEY
    PERPLEXITY_API_KEY
"""

import os, sys, threading, time

from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.chain  import Chain
from yait_aichain.pool import Pool, PENDING, RUNNING, DONE, FAILED
from yait_aichain.agent  import Agent
from yait_aichain.tools import searchPerplexity

ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY")
PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY")


def _hr(label): print(f"\n{'─' * 60}\n  {label}\n{'─' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHAIN — inspect history after run
# ─────────────────────────────────────────────────────────────────────────────
_hr("CHAIN · chain.history")

writer = Skill(
    model = Model("claude-sonnet-4-6", api_key=ANTHROPIC_KEY),
    input = {"messages": [{"role": "user", "parts": ["Write one sentence about {topic}."]}]},
    name  = "writer",
)
reviewer = Skill(
    model = Model("claude-sonnet-4-6", api_key=ANTHROPIC_KEY),
    input = {"messages": [{"role": "user", "parts": ["Improve this: {result}"]}]},
    name  = "reviewer",
)

chain = Chain(steps=[writer, reviewer])
chain.run(variables={"topic": "black holes"})

for record in chain.history:
    output_preview = str(record["output"])[:80].replace("\n", " ")
    print(f"  step {record['step']}  [{record['kind']}]  {record['name']:<12}"
          f"  keys_in={sorted(record['input'].keys())}")
    print(f"          output → {output_preview!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. POOL — poll status live while running, inspect history after
# ─────────────────────────────────────────────────────────────────────────────
_hr("POOL · pool.status (live) + pool.history")

skill = Skill(
    model = Model("claude-sonnet-4-6", api_key=ANTHROPIC_KEY),
    input = {"messages": [{"role": "user", "parts": ["Explain {topic} in one sentence."]}]},
)

items = [
    {"topic": "neural networks"},
    {"topic": "transformers"},
    {"topic": "diffusion models"},
    {"topic": "reinforcement learning"},
]

pool = Pool(skill, items=items, max_flows=4)

# Poll status in a background thread while pool runs
def _poll():
    while True:
        s = pool.status
        done    = s[DONE]
        running = s[RUNNING]
        pending = s[PENDING]
        failed  = s[FAILED]
        print(f"  status → pending={pending}  running={running}  done={done}  failed={failed}")
        if done + failed == len(items):
            break
        time.sleep(0.3)

thread = threading.Thread(target=_poll)
thread.start()
pool.run()
thread.join()

print()
for r in pool.history:
    status_name = {DONE: "DONE", FAILED: "FAILED"}[r["status"]]
    preview     = str(r["output"])[:70].replace("\n", " ") if r["output"] else r["error"]
    print(f"  [{r['index']}] {r['variables']['topic']:<25} {status_name:<6}  {r['duration']}s")
    print(f"       {preview!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. AGENT — verbose=1 shows plan + step-by-step progress
# ─────────────────────────────────────────────────────────────────────────────
_hr("AGENT · verbose=1 (plan + steps)")

agent = Agent(
    orchestrator = Model("claude-sonnet-4-6", api_key=ANTHROPIC_KEY),
    tools        = [searchPerplexity(api_key=PERPLEXITY_KEY)],
    max_steps    = 5,
    verbose      = 1,    # ← shows plan + one line per step
)

result = agent.run("What is the latest version of Python and when was it released?")
print(f"\n  ▸ final answer: {result.output[:120]}")
print(f"  ▸ steps={result.steps_taken}  tokens={result.tokens_used:,}")
