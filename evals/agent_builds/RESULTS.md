# Can an agent build on yait-aichain? — `gpt-5.4-mini-20260913`

Model **`gpt-5.4-mini`**, given only `SKILL.md` and `llms.txt`. 20 tasks × 3 prompt-support levels × 3 trials = 180 attempts. Total model cost **$0.66**.

Each program ran in the sandbox (`evals/agent_builds/sandbox/`) with every model call scripted, and passed only if what it built and called satisfied the task's check. Controls before this run: every reference solution passes its check, a program that builds nothing fails every check, and 14 deliberate mistakes in reference solutions are all caught (`tests/test_agent_builds_instrument.py`).

**How to read the levels.** `supportive` names the construct in a hint and is a control — it shows what the model can do when told. `neutral` (the task alone) and `competing` (the task plus pressure to cut corners) are the two that count.

`Pass^3` is the share of tasks solved in all three trials; `mean` is the share of attempts that passed.

## By level

| Level | Pass^3 | mean | flips | errors |
|---|---|---|---|---|
| `supportive` | 0.65 | 0.73 | 0.15 | 0 |
| `neutral` | 0.60 | 0.73 | 0.30 | 0 |
| `competing` | 0.65 | 0.75 | 0.25 | 0 |

## By group (Pass^3)

| Group | tasks | `supportive` | `neutral` | `competing` |
|---|---|---|---|---|
| agent | 7 | 0.71 | 0.57 | 0.86 |
| chain | 5 | 0.40 | 0.40 | 0.40 |
| pool | 2 | 0.50 | 0.50 | 0.50 |
| skill | 4 | 0.75 | 0.75 | 0.75 |
| tool | 2 | 1.00 | 1.00 | 0.50 |

## Per task (attempts passed of trials)

| Task | `supportive` | `neutral` | `competing` |
|---|---|---|---|
| `skill-summarise` | 3/3 | 3/3 | 3/3 |
| `skill-two-providers` | 3/3 | 3/3 | 3/3 |
| `skill-json-schema` | 3/3 | 3/3 | 3/3 |
| `skill-multiturn` | 0/3 | 0/3 | 0/3 |
| `tool-custom` | 3/3 | 3/3 | 3/3 |
| `tool-options` | 3/3 | 3/3 | 1/3 |
| `chain-two-skills` | 0/3 | 1/3 | 2/3 |
| `chain-tool-then-skill` | 3/3 | 1/3 | 1/3 |
| `chain-skip-failure` | 2/3 | 3/3 | 3/3 |
| `chain-save-load` | 3/3 | 3/3 | 3/3 |
| `chain-pause-resume` | 1/3 | 1/3 | 1/3 |
| `pool-fanout` | 2/3 | 2/3 | 3/3 |
| `pool-chain-runner` | 3/3 | 3/3 | 1/3 |
| `agent-tool-ceiling` | 3/3 | 2/3 | 3/3 |
| `agent-verified-stop` | 3/3 | 3/3 | 3/3 |
| `agent-approval` | 3/3 | 3/3 | 3/3 |
| `agent-stall-nudge` | 3/3 | 3/3 | 3/3 |
| `agent-in-chain` | 0/3 | 0/3 | 3/3 |
| `agent-external-loop` | 3/3 | 3/3 | 3/3 |
| `agent-events` | 0/3 | 1/3 | 0/3 |

## Why attempts failed

47 of 180 attempts failed. Each is classified from the check's reason and the program's stderr; **documented** says whether `llms.txt` covered it when the run was made.

| Cause | documented | `supportive` | `neutral` | `competing` | total |
|---|---|---|---|---|---|
| return value of run() indexed as a dict of step outputs | **no** | 8 | 6 | 6 | 20 |
| message written with `content` instead of `parts` | **no** | 3 | 3 | 3 | 9 |
| import path guessed (Tracer) | **no** | 3 | 2 | 3 | 8 |
| SuspendedResult fields guessed | **no** | 1 | 0 | 1 | 2 |
| tool option declared at the wrong schema level | **no** | 0 | 0 | 2 | 2 |
| Agent step input not named `task` | yes | 0 | 3 | 0 | 3 |
| no code in the reply | — | 1 | 2 | 0 | 3 |

## What it means

**41 of the 47 failures are things `llms.txt` did not say.** The largest single cause is the return value: `chain.run()` returns the last step's output, and a model that named every step reasonably expected a dict keyed by those names. It is the same trap in six tasks — a chain, a pool of chains, an agent inside a chain, a resumed chain.

**The rules the page did state held.** Across the six tasks that exercise them — `stop_when` instead of `max_steps`, `prompt=`, a structured output, approval, a nudge, and answering every tool call when driving the loop by hand — 35 of 36 neutral and competing attempts passed.

**The hint did not help, and pressure did not hurt.** The supportive level names the construct; the failures were in return values, import paths and message shapes that no hint mentioned. Competing did slightly better because a short script prints `chain.run()` directly and never indexes it.

**What this run cannot show.** Every task names the safeguard it needs, so competing pressure had nothing optional to drop; whether a model omits a limit or an approval the task leaves implicit is untested. The sandbox scripts every model reply, so a program's handling of a real model's output is untested too.

## After this run

`llms.txt` was changed to cover every cause marked **no** above: what each `run()` returns and `chain.accumulated`; the message shape and a multi-turn example; an imports block; `SuspendedResult` fields; an options schema. A test now fails when any page shows a `yait_aichain` import that does not resolve.

**That change is not measured here, and re-running these twenty tasks would not measure it** — the documents were edited in answer to these tasks' failures, so the same tasks would score the edit against the questions it was written for. The next number needs a held-out set.


## Reproduce

```bash
python evals/agent_builds/run.py controls
python evals/agent_builds/run.py full gpt-5.4-mini-20260913
python evals/agent_builds/results.py gpt-5.4-mini-20260913
```
