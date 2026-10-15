# Observability

An agent is a model inside a harness. The model proposes; the harness checks,
authorises, runs, records and returns what came back. Everything the harness
does happens at a boundary between steps, and every boundary can be watched
from outside the run — without changing the run.

Two channels, for two readers:

- **logs** are for a person diagnosing — text, through Python's `logging`;
- **events** are for a program — structured records a UI, a tracer or a
  database consumes.

```python
from yait_aichain import Tracer

tracer = Tracer()
agent = Agent(model, tools=[…], hooks=[tracer])
agent.run("…")
[(e.type, e.name, e.cost) for e in tracer.events]
```

→ Every parameter: [API: Control](../api/control.md)

---

## Logs: the library emits, you decide where

The library writes through named loggers — `yait_aichain.agent`, `.chain`,
`.skills` — and attaches no handler of its own, so it is silent until your
application decides where its records go: the console, a file, CloudWatch.
`verbose=1` on an agent is the shortcut that attaches a console handler. Keys
and auth headers are never logged.

A log line is for reading. To drive behaviour — a status in a chat window, a
trace row in a database — use events, which carry structure rather than text.

## Events: every boundary, in order

A hook is any callable given an `Event`, passed as `hooks=[…]` to a Skill, a
Chain, a Pool or an Agent. A hook that raises is logged and never breaks the run,
and a hook cannot change what the run does: observation only.

| Event | From | Carries |
|---|---|---|
| `run.started` · `run.finished` | Agent, Chain, Pool | the task and mode; the verdict, usage and cost |
| `step.started` · `step.ended` | Chain, Pool | one step and its kind; for a Pool, one item, its index as the step |
| `tool_call.started` · `tool_call.ended` | Agent | one call: its id, its arguments, its raw result or error; each item of a `pool` call as a child with `parent_call_id`, its arguments cut at 8 KB (`complete: false`) and no result, at most 256 per call |
| `text.started` · `text.delta` · `text.ended` | Agent, while streaming | the answer as it is written |
| `approval.requested` · `approval.decided` | Agent | a gated call, before someone answers and after |
| `llm_call.started` · `llm_call.ended` | Skill | the model, tokens, cost, duration |

Every event of a run carries its `run_id`, including the model calls a Skill
makes inside an agent: two runs writing into one stream can be told apart
afterwards.

Three bases save writing a hook from scratch. `Hook` dispatches an event to the
method of the same name — `"tool_call.started"` reaches
`tool_call_started(event)` — so a subclass implements only the boundaries it
cares about. `Tracer` keeps every event in `.events`; `LoggingTracer` logs each
one to a `logging` logger. The library ships no tracing dependency:
OpenTelemetry, file and database sinks are a hook in your application.

## Enough to rebuild a turn

A tool call's event carries the tool's **own** result, by value — not the text
the model was shown. Text loses things a program needs: units, column names,
the difference between *no rows* and *the tool refused*. A UI that cannot tell
a refusal from an empty answer draws the same empty chart for both.

By value rather than by handle, because a handle needs somewhere to live and
the process that issued it may be gone — the target is a serverless function.
So an event is not always small: a thousand-row result is about 90 KB. An
event's printed form leaves the payload out, so logs stay readable.

## Watching it happen

Hooks are pushed to you. `agent.stream(task)` is the same events pulled — the
same loop walked instead of exhausted, not a second implementation — so a run
behaves the same whichever way it is watched. The answer arrives as
`text.delta` pieces interleaved, in order, with the tool calls that came
before it; the result itself lands on `agent.last_result` when the stream
ends. Breaking out of the loop abandons the run: no result, no
`run.finished`.

`run()` emits no `text.*`, deliberately. It has nobody to show pieces to, and
not streaming keeps the fallback between models, which a stream must give up —
a second model's answer cannot continue the first one's sentence.

## What a result rests on

Every attempt goes into the run's **journal**, each with its outcome and the
kind of evidence behind it, and the asymmetry is the point:

| Evidence | Means |
|---|---|
| `check` | a fact the code established — a tool ran and returned, a permission was refused |
| `model_claim` | the model says so, and nothing verified it — a delegated worker's report is the usual case |

A failure is usually a `check`; a success is often only a claim. A run whose
successes are all claims has proven nothing, and the result says so:
`result.acceptance.kind` is `checked`, `claimed` or `unsupported`. It is
reported, not enforced — an unsupported answer is still returned, and you can
see that it is one.

An acceptance carries a fingerprint of its checked evidence. Gather the
evidence again later — re-run the checks, re-read the file — and
`still_holds()` says whether the result was accepted on facts that are still
true; `supersede(reason)` marks it as needing to be earned again after a
person intervened.

## A run that is stuck

Two ways a run goes nowhere, and they look different in the journal: it keeps
**failing** — nothing has moved for the last few attempts — or it keeps
**succeeding pointlessly**, re-issuing a call that returns cleanly every time.
`stalled(k)` and `repeating(k)` in `stop_when` watch for each and, when one
fires, tell the model in an ordinary, logged message, then let it go on.
Surfaced, not enforced: an agent circling a hard problem legitimately must not
be cut off, and a ceiling is still there for the one that never stops.

Only exact repetition counts. A looser rule — intents that merely resemble
each other — was measured on three real runs and rejected: a healthy binary
search restates "probe the midpoint" every turn and scored *less* alike than a
broken one.

## Beacons: a child speaks up

Anything inside a run — a tool, a delegated worker, a pool item — can signal
whoever is waiting on it with `beacon(kind, message)`. `blocker`, `question`
and `contract_change` demand attention: a Pool stops starting new items when
one arrives, instead of spending the rest of the fan-out on work the blocker
already made pointless. `contract`, `decision` and `fact` are recorded and
passed up. Beacons travel up through every level and land on `result.beacons`.

## After a crash

A Chain is written to its store after every step that finishes, so a process
that dies mid-run — a crash, a timeout — loses nothing finished: a new Chain on
the same store continues with `resume(run_id)`. An agent needs no equivalent:
its state is the conversation, which the caller holds.

---

▶ [`examples/36_observe.py`](../../examples/36_observe.py) ·
[`examples/35_policy.py`](../../examples/35_policy.py)
