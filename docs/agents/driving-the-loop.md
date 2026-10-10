# Driving the loop

`agent.run()` owns its loop: it asks the model, runs the tools, appends the
results and asks again until the run ends. Sometimes something else has to own
it:

- a **serverless function** that may run for minutes, not hours, and so takes
  one turn per invocation, keeping the conversation in a store between them;
- a **benchmark harness** that runs the tools itself, in its own environment;
- a **UI** that shows each decision to a person before anything happens.

For those, the same machinery is exposed in pieces. The agent decides; you
execute and append; it decides again.

```python
from yait_aichain.models import ToolCallRequest, tool_result_turn

messages = agent.opening(task)
state    = agent.new_state()
while True:
    decision = agent.step(messages, state)            # decides, runs nothing
    if not isinstance(decision, ToolCallRequest):
        answer = decision                             # a string is the answer
        break
    messages.append(decision.as_turn())
    for call in decision.calls:
        result, error = agent.execute(call, state)    # or run it your own way
        messages.append(tool_result_turn(call.id, error or result))
```

→ Every parameter: [API: Agent](../api/agent.md)

---

## Every call gets an answer

The one obligation a driver takes on: after a reply with tool calls, append a
result for **every** call before the next `step()`, failed ones included — the
error is the result. `step()` checks this before it asks anything, and raises
locally, naming the unanswered calls; `dangling_calls(messages)` lets a driver
ask first.

The check exists because providers disagree about a call nobody answered, and
in the worst way. Three of them reject the request. Google is looser. A
self-hosted OpenAI-compatible server validates nothing and passes it through,
so the model meets its own unanswered call and improvises — repeats it,
apologises, invents a result — differently every time. That is run-to-run
variance produced by the harness, not the model. With the check, the same input
gives the same outcome, and the failure happens on your machine, before the
network, with a message that points at your code instead of at a provider's
wording.

## Why the pieces are shaped this way

- **The conversation is the state.** `messages` is a list of plain dicts, and
  `state` holds only the counters the stop conditions read and, in
  `waterfall`, the plan. Both serialise as they are, so a turn can end in one
  process and the next begin in another.
- **`opening()` gives two messages**: the stable prefix — the instructions,
  the tools — and the task. Keeping the prefix the same from turn to turn is
  what lets a provider cache it. `ground_rules=True` adds the rule that a
  plain-text reply ends the run; `run()` sets it, because in its loop that is
  true, and a driver usually leaves it off, because in its loop a reply may be
  an ordinary turn.
- **`execute()` never raises.** It returns `(result, error)`, so a tool that
  fails becomes a message the model reads, the way it does inside `run()`.

## One turn per invocation

The serverless shape follows directly: an invocation loads the messages for a
run, calls `step()` once, executes what it asked for, appends, saves, and
returns. The next event — a timer, a queue message, the end of a slow tool —
starts the next invocation. The fifteen-minute limit of a function becomes
the size of a checkpoint instead of a ceiling on the run.

When the agent should wait for the world rather than for the next turn, it
does not need a driver at all: a `wait()` or `gate()` in its tools parks the
run, and `agent.resume(run_id, signal)` carries on.
→ [Pause and resume](../pipelines/pause-and-resume.md) · [From laptop to production](../production.md)
